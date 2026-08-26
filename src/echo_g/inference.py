# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from echo_g.config import ExperimentConfig
from echo_g.data import (
    LengthBucketBatchSampler,
    RobotSpeechDataset,
    collate_motion,
)
from echo_g.flow import sample_euler
from echo_g.model import SpeechGroundedDiT
from echo_g.training import resolve_device
from echo_g.utils import (
    atomic_json_save,
    atomic_torch_save,
    move_batch,
    sha256,
    stable_seed,
)

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample robot motion with ECHO-G SGDiT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Required only for legacy paper checkpoints without an embedded config",
    )
    parser.add_argument("--split", choices=["train", "val"], default="val")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0])
    parser.add_argument("--steps", type=int)
    parser.add_argument("--guidance-scale", type=float)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=3)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def load_model_and_config(
    checkpoint_path: Path,
    config_path: Path | None,
    device: torch.device,
) -> tuple[
    SpeechGroundedDiT,
    ExperimentConfig,
    torch.Tensor,
    torch.Tensor,
    dict[str, Any],
]:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    required = {"ema", "motion_mean", "motion_std", "experiment"}
    missing = sorted(required - checkpoint.keys())
    if missing:
        raise ValueError(f"{checkpoint_path}: missing checkpoint fields {missing}")
    experiment = checkpoint["experiment"]
    if "config" in experiment:
        config = ExperimentConfig.from_dict(experiment["config"])
        model = SpeechGroundedDiT(config.model)
    else:
        if config_path is None:
            raise ValueError("--config is required for a legacy paper checkpoint")
        config = ExperimentConfig.from_yaml(config_path)
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(f"{checkpoint_path}: missing legacy model_config")
        model = SpeechGroundedDiT.from_checkpoint_config(model_config)
        if model.config != config.model:
            raise ValueError("legacy checkpoint architecture does not match --config")
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.to(device).eval()
    mean = torch.as_tensor(checkpoint["motion_mean"]).float()
    std = torch.as_tensor(checkpoint["motion_std"]).float().clamp_min(1e-6)
    return model, config, mean, std, experiment


def validate_prediction(
    path: Path,
    expected_frames: int,
    expected_dimension: int,
    seed: int,
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    motion = torch.as_tensor(payload["robot_repr"])
    checks = {
        "shape": motion.shape == (expected_frames, expected_dimension),
        "finite": bool(torch.isfinite(motion).all()),
        "seed": int(payload.get("seed", -1)) == seed,
        "units": payload.get("representation_units") == "physical",
    }
    if not all(checks.values()):
        raise ValueError(f"{path}: incompatible existing prediction {checks}")


def export_predictions(
    checkpoint_path: Path,
    data_root: Path,
    output_dir: Path,
    device_name: str,
    split: str = "val",
    seeds: list[int] | None = None,
    steps: int | None = None,
    guidance_scale: float | None = None,
    limit: int = 0,
    batch_size: int = 3,
    num_workers: int = 4,
    overwrite: bool = False,
    config_path: Path | None = None,
) -> Path:
    device = resolve_device(device_name)
    model, config, mean, std, experiment = load_model_and_config(
        checkpoint_path, config_path, device
    )
    sample_seeds = [0] if seeds is None else seeds
    if not sample_seeds:
        raise ValueError("at least one sampling seed is required")
    integration_steps = steps or config.sampling.integration_steps
    cfg = guidance_scale if guidance_scale is not None else config.sampling.guidance_scale
    dataset = RobotSpeechDataset(
        data_root,
        config.data,
        config.model,
        split,
        mean,
        std,
        max_items=limit,
    )
    sampler = LengthBucketBatchSampler(
        dataset.lengths,
        batch_size,
        seed=0,
        bucket_multiplier=config.training.bucket_multiplier,
        shuffle=False,
        drop_last=False,
    )
    loader = DataLoader(
        dataset,
        batch_sampler=sampler,
        collate_fn=collate_motion,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
    )
    checkpoint_hash = sha256(checkpoint_path)
    mean_device = mean.to(device)
    std_device = std.to(device)
    produced = 0
    processed = 0

    for collated in loader:
        stems = collated["stems"]
        frame_counts = [int(value) for value in collated["motion_mask"].sum(dim=1)]
        batch = move_batch(collated, device)
        for sample_seed in sample_seeds:
            seed_root = output_dir / f"seed_{sample_seed:03d}"
            output_paths = [seed_root / f"{stem}.pt" for stem in stems]
            missing_rows: list[int] = []
            for row, output_path in enumerate(output_paths):
                if output_path.is_file() and not overwrite:
                    validate_prediction(
                        output_path,
                        frame_counts[row],
                        config.model.motion_dim,
                        sample_seed,
                    )
                else:
                    missing_rows.append(row)
            if not missing_rows:
                continue

            initial_noise = torch.zeros(
                len(stems),
                batch["audio"].shape[1],
                model.motion_dim,
                device=device,
            )
            for row, stem in enumerate(stems):
                generator = torch.Generator(device=device)
                generator.manual_seed(stable_seed(stem, sample_seed))
                initial_noise[row, : frame_counts[row]] = torch.randn(
                    frame_counts[row],
                    model.motion_dim,
                    generator=generator,
                    device=device,
                )
            normalized = sample_euler(
                model,
                batch,
                integration_steps,
                cfg,
                initial_noise=initial_noise,
            )
            for row in missing_rows:
                stem = stems[row]
                frames = frame_counts[row]
                robot_motion = normalized[row, :frames].float() * std_device + mean_device
                robot_motion = robot_motion.cpu().contiguous()
                if not torch.isfinite(robot_motion).all():
                    raise ValueError(f"{stem}: sampled motion contains non-finite values")
                atomic_torch_save(
                    {
                        "robot_repr": robot_motion,
                        "real_num_frames": frames,
                        "fps": float(config.data.fps),
                        "stem": stem,
                        "seed": sample_seed,
                        "representation_units": "physical",
                        "representation_schema": config.data.representation_schema,
                        "source_checkpoint_sha256": checkpoint_hash,
                    },
                    output_paths[row],
                )
                produced += 1
        processed += len(stems)
        if processed % 60 < len(stems) or processed == len(dataset):
            LOGGER.info("processed %d/%d utterances", processed, len(dataset))

    expected = len(dataset) * len(sample_seeds)
    manifest = {
        "schema": "echo-g-sgdit-inference-v1",
        "status": "complete",
        "checkpoint_sha256": checkpoint_hash,
        "training_schema": experiment.get("schema"),
        "split": split,
        "utterances": len(dataset),
        "seeds": sample_seeds,
        "expected_files": expected,
        "new_files": produced,
        "fps": config.data.fps,
        "representation_units": "physical",
        "representation_schema": config.data.representation_schema,
        "integration_steps": integration_steps,
        "guidance_scale": cfg,
    }
    marker = output_dir / "inference_complete.json"
    atomic_json_save(manifest, marker)
    LOGGER.info("physical robot-motion export complete: %d files under %s", expected, output_dir)
    return marker


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    export_predictions(
        args.checkpoint,
        args.data_root,
        args.output_dir,
        args.device,
        args.split,
        args.seeds,
        args.steps,
        args.guidance_scale,
        args.limit,
        args.batch_size,
        args.num_workers,
        args.overwrite,
        args.config,
    )


if __name__ == "__main__":
    main()
