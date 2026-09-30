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
import math
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from echo_g.config import ExperimentConfig, HumanInferenceConfig, load_inference_config
from echo_g.data import (
    ConditionDataset,
    LengthBucketBatchSampler,
    RobotSpeechDataset,
    collate_motion,
    load_aligned_lengths,
    read_stems,
)
from echo_g.flow import sample_euler
from echo_g.model import SpeechGroundedDiT
from echo_g.training import resolve_device
from echo_g.utils import (
    atomic_json_save,
    atomic_torch_save,
    configure_reproducibility,
    move_batch,
    sha256,
    stable_seed,
)
from echo_g.word_time_attention import WORD_TIME_CONFIG, WORD_TIME_SCHEMA

LOGGER = logging.getLogger(__name__)
TIME_CONVENTION = "signed_frame_minus_token_center_seconds"


def parse_args(condition_only: bool = False) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample robot motion with ECHO-G SGDiT")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--retarget-checkpoint", type=Path)
    parser.add_argument("--human-vae-checkpoint", type=Path)
    if condition_only:
        parser.add_argument("--condition-dir", type=Path, required=True)
        parser.add_argument("--stem-list", type=Path)
        parser.add_argument("--lengths-csv", type=Path)
        parser.add_argument("--condition-manifest", type=Path)
        parser.add_argument("--condition-manifest-sha256")
        parser.set_defaults(data_root=None)
    else:
        parser.add_argument("--data-root", type=Path, required=True)
        parser.set_defaults(
            condition_dir=None,
            stem_list=None,
            lengths_csv=None,
            condition_manifest=None,
            condition_manifest_sha256=None,
        )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        help="Required for checkpoints without an embedded package config",
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
    ExperimentConfig | HumanInferenceConfig,
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
        if config_path is not None and ExperimentConfig.from_yaml(config_path) != config:
            raise ValueError("--config differs from the checkpoint's embedded configuration")
        model = SpeechGroundedDiT(config.model)
    else:
        if config_path is None:
            raise ValueError("--config is required when the checkpoint has no package config")
        config = load_inference_config(config_path)
        model_config = checkpoint.get("model_config")
        if not isinstance(model_config, dict):
            raise ValueError(f"{checkpoint_path}: missing model_config")
        expected = {
            "conditioning": config.data.conditioning,
            "target": "human" if isinstance(config, HumanInferenceConfig) else "robot",
            "fps": config.data.fps,
            "max_frames": config.data.max_frames,
        }
        if config.model.architecture == "v2":
            expected.update(
                {
                    "wordtime_schema": WORD_TIME_SCHEMA,
                    "wordtime_mode": "qknorm_wordtime",
                    "wordtime_config": WORD_TIME_CONFIG,
                    "time_distance_convention": TIME_CONVENTION,
                    "max_text_tokens": config.data.max_text_tokens,
                }
            )
        for name, value in expected.items():
            if experiment.get(name) != value:
                raise ValueError(f"Checkpoint experiment.{name} differs from the configuration")
        model = SpeechGroundedDiT.from_checkpoint_config(model_config, config.model.architecture)
        if model.config != config.model:
            raise ValueError("checkpoint architecture does not match --config")
    model.load_state_dict(checkpoint["ema"], strict=True)
    model.to(device).eval()
    mean = torch.as_tensor(checkpoint["motion_mean"]).float().reshape(-1)
    std = torch.as_tensor(checkpoint["motion_std"]).float().reshape(-1)
    if (
        mean.shape != (model.motion_dim,)
        or std.shape != mean.shape
        or not torch.isfinite(mean).all()
        or not torch.isfinite(std).all()
        or bool((std <= 0).any())
    ):
        raise ValueError("Checkpoint has invalid motion normalization statistics")
    return model, config, mean, std, experiment


def validate_prediction(
    path: Path,
    expected_frames: int,
    expected_dimension: int,
    seed: int,
    identity: dict[str, Any],
) -> None:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    motion = torch.as_tensor(payload["robot_repr"])
    checks = {
        "shape": motion.shape == (expected_frames, expected_dimension),
        "finite": bool(torch.isfinite(motion).all()),
        "seed": int(payload.get("seed", -1)) == seed,
        "units": payload.get("representation_units") == "physical",
    }
    checks.update({key: payload.get(key) == value for key, value in identity.items()})
    if not all(checks.values()):
        raise ValueError(f"{path}: incompatible existing prediction {checks}")


def export_predictions(
    checkpoint_path: Path,
    data_root: Path | None,
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
    condition_dir: Path | None = None,
    stem_list: Path | None = None,
    lengths_csv: Path | None = None,
    condition_manifest: Path | None = None,
    condition_manifest_sha256: str | None = None,
    retarget_checkpoint: Path | None = None,
    human_vae_checkpoint: Path | None = None,
) -> Path:
    if batch_size < 1 or num_workers < 0 or limit < 0:
        raise ValueError("Invalid batch size, worker count, or limit")
    configure_reproducibility(0)
    device = resolve_device(device_name)
    model, config, mean, std, experiment = load_model_and_config(
        checkpoint_path, config_path, device
    )
    decoder = None
    decoder_metadata: dict[str, Any] = {}
    human_profile = isinstance(config, HumanInferenceConfig)
    if human_profile:
        if retarget_checkpoint is None or human_vae_checkpoint is None:
            raise ValueError(
                "HumanRetarget requires --retarget-checkpoint and --human-vae-checkpoint"
            )
        from echo_g.human_retarget.decoder import HumanRetargetDecoder

        decoder = HumanRetargetDecoder(retarget_checkpoint, human_vae_checkpoint, device)
        decoder_metadata = decoder.metadata()
    elif retarget_checkpoint is not None or human_vae_checkpoint is not None:
        raise ValueError("Decoder checkpoints are only valid with the HumanRetarget profile")
    output_dimension = 39 if human_profile else config.model.motion_dim
    time_convention = TIME_CONVENTION if config.model.architecture == "v2" else "none"
    requested_split = split if condition_dir is None else None
    if human_profile and data_root is not None:
        if condition_dir is not None:
            raise ValueError("Use either --condition-dir or --data-root")
        if split != "val":
            raise ValueError("HumanRetarget supports condition-only or validation inference")
        # No human targets or training statistics are needed for this pipeline.
        stem_list = data_root / config.data.val_split
        expected_split_hash = experiment.get("val_split_sha256")
        if expected_split_hash and sha256(stem_list) != expected_split_hash:
            raise ValueError("Dataset split differs from the checkpoint")
        condition_dir = data_root / config.data.condition_dir
        lengths_csv = (
            data_root / config.data.aligned_lengths_file
            if config.data.aligned_lengths_file
            else None
        )
        condition_manifest = (
            data_root / config.data.condition_manifest if config.data.condition_manifest else None
        )
        condition_manifest_sha256 = config.data.condition_manifest_sha256
        data_root = None
    sample_seeds = [0] if seeds is None else seeds
    if not sample_seeds:
        raise ValueError("at least one sampling seed is required")
    integration_steps = config.sampling.integration_steps if steps is None else steps
    cfg = guidance_scale if guidance_scale is not None else config.sampling.guidance_scale
    if integration_steps < 1 or not math.isfinite(cfg) or cfg < 0:
        raise ValueError("Invalid sampling steps or guidance scale")
    if len(sample_seeds) != len(set(sample_seeds)):
        raise ValueError("Duplicate sample seeds")
    if condition_dir is not None:
        if data_root is not None:
            raise ValueError("Use either --condition-dir or --data-root")
        stems = (
            read_stems(stem_list)
            if stem_list
            else sorted(p.stem for p in condition_dir.glob("*.pt"))
        )
        if limit:
            stems = stems[:limit]
        lengths = load_aligned_lengths(lengths_csv) if lengths_csv else None
        dataset = ConditionDataset(
            condition_dir,
            stems,
            config.data,
            config.model,
            lengths,
            condition_manifest,
            condition_manifest_sha256,
        )
    else:
        if data_root is None:
            raise ValueError("A data root or condition directory is required")
        dataset = RobotSpeechDataset(
            data_root,
            config.data,
            config.model,
            split,
            mean,
            std,
            max_items=limit,
        )
        expected_split_hash = experiment.get(f"{split}_split_sha256")
        if expected_split_hash and sha256(dataset.split_path) != expected_split_hash:
            raise ValueError(
                "Dataset split differs from the checkpoint; "
                "use condition-only inference for new data"
            )
    sampler = LengthBucketBatchSampler(
        dataset.lengths,
        batch_size,
        seed=0,
        bucket_multiplier=(
            config.training.bucket_multiplier if isinstance(config, ExperimentConfig) else 64
        ),
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
        condition_hashes = {stem: sha256(dataset.condition_root / f"{stem}.pt") for stem in stems}
        identities = {
            stem: {
                "source_checkpoint_sha256": checkpoint_hash,
                "source_condition_sha256": condition_hashes[stem],
                "integration_steps": integration_steps,
                "guidance_scale": cfg,
                "fps": float(config.data.fps),
                "representation_schema": config.data.representation_schema,
                "time_distance_convention": time_convention,
                **decoder_metadata,
                "conditioning": config.data.conditioning,
                "stem": stem,
            }
            for stem in stems
        }
        for sample_seed in sample_seeds:
            seed_root = output_dir / f"seed_{sample_seed:03d}"
            output_paths = [seed_root / f"{stem}.pt" for stem in stems]
            missing_rows: list[int] = []
            for row, output_path in enumerate(output_paths):
                if output_path.is_file() and not overwrite:
                    validate_prediction(
                        output_path,
                        frame_counts[row],
                        output_dimension,
                        sample_seed,
                        identities[stems[row]],
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
                if decoder is not None:
                    robot_motion = decoder.decode(robot_motion)
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
                        **identities[stem],
                    },
                    output_paths[row],
                )
                produced += 1
        processed += len(stems)
        if processed % 60 < len(stems) or processed == len(dataset):
            LOGGER.info("processed %d/%d utterances", processed, len(dataset))

    expected = len(dataset) * len(sample_seeds)
    manifest = {
        "schema": "echo-g-v2-inference-v1",
        "status": "complete",
        "checkpoint_sha256": checkpoint_hash,
        "training_schema": experiment.get("schema"),
        "split": requested_split,
        "time_distance_convention": time_convention,
        "wordtime_schema": WORD_TIME_SCHEMA if config.model.architecture == "v2" else None,
        "conditioning": config.data.conditioning,
        **decoder_metadata,
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


def main(condition_only: bool = False) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args(condition_only)
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
        args.condition_dir,
        args.stem_list,
        args.lengths_csv,
        args.condition_manifest,
        args.condition_manifest_sha256,
        args.retarget_checkpoint,
        args.human_vae_checkpoint,
    )


def infer_main() -> None:
    main(condition_only=True)


if __name__ == "__main__":
    main()
