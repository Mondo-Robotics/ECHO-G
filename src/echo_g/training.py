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
import torch.nn as nn
from torch.utils.data import DataLoader

from echo_g.config import ExperimentConfig
from echo_g.data import (
    LengthBucketBatchSampler,
    RobotSpeechDataset,
    collate_motion,
    load_motion_stats,
)
from echo_g.flow import flow_matching_loss
from echo_g.model import SpeechGroundedDiT
from echo_g.utils import (
    atomic_json_save,
    atomic_torch_save,
    configure_reproducibility,
    cpu_state,
    move_batch,
    sha256,
)
from echo_g.word_time_attention import WORD_TIME_CONFIG, WORD_TIME_SCHEMA

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the ECHO-G speech-grounded transformer")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-train", type=int, default=0)
    parser.add_argument("--max-val", type=int, default=0)
    parser.add_argument("--longest-first", action="store_true")
    return parser.parse_args()


def resolve_device(name: str) -> torch.device:
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_loaders(
    config: ExperimentConfig,
    data_root: Path,
    mean: torch.Tensor,
    std: torch.Tensor,
    device: torch.device,
    max_train: int,
    max_val: int,
    longest_first: bool,
) -> tuple[
    RobotSpeechDataset,
    RobotSpeechDataset,
    LengthBucketBatchSampler,
    DataLoader[dict[str, Any]],
    DataLoader[dict[str, Any]],
]:
    train_dataset = RobotSpeechDataset(
        data_root,
        config.data,
        config.model,
        "train",
        mean,
        std,
        max_train,
        longest_first,
    )
    val_dataset = RobotSpeechDataset(
        data_root,
        config.data,
        config.model,
        "val",
        mean,
        std,
        max_val,
        longest_first,
    )
    train_sampler = LengthBucketBatchSampler(
        train_dataset.lengths,
        config.training.batch_size,
        config.training.seed,
        config.training.bucket_multiplier,
        shuffle=True,
        drop_last=True,
    )
    val_sampler = LengthBucketBatchSampler(
        val_dataset.lengths,
        config.training.batch_size,
        config.training.seed,
        config.training.bucket_multiplier,
        shuffle=False,
        drop_last=True,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        collate_fn=collate_motion,
        num_workers=config.training.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=config.training.num_workers > 0,
    )
    val_workers = max(0, config.training.num_workers // 2)
    val_loader = DataLoader(
        val_dataset,
        batch_sampler=val_sampler,
        collate_fn=collate_motion,
        num_workers=val_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=val_workers > 0,
    )
    return train_dataset, val_dataset, train_sampler, train_loader, val_loader


@torch.no_grad()
def update_ema(ema: dict[str, torch.Tensor], model: nn.Module, decay: float) -> None:
    for name, value in model.state_dict().items():
        ema[name].mul_(decay).add_(value.detach(), alpha=1.0 - decay)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    ema: dict[str, torch.Tensor],
    loader: DataLoader[dict[str, Any]],
    device: torch.device,
    temporal_weight: float,
    max_batches: int,
) -> float:
    # Clone even on CPU: detach().cpu() can still alias the live model.
    backup = {name: value.detach().cpu().clone() for name, value in model.state_dict().items()}
    was_training = model.training
    model.load_state_dict(ema, strict=True)
    model.eval()
    total = 0.0
    batches = 0
    # manual_seed initializes every CUDA generator; preserve all of them as well.
    devices = list(range(torch.cuda.device_count())) if torch.cuda.is_available() else []
    try:
        with torch.random.fork_rng(devices=devices):
            torch.manual_seed(100_003)
            if device.type == "cuda":
                torch.cuda.manual_seed_all(100_003)
            for index, batch in enumerate(loader):
                if max_batches > 0 and index >= max_batches:
                    break
                batch = move_batch(batch, device)
                loss, _, _ = flow_matching_loss(model, batch, 0.0, temporal_weight)
                if not torch.isfinite(loss):
                    raise FloatingPointError(f"non-finite validation loss at batch {index}")
                total += float(loss.item())
                batches += 1
        if batches == 0:
            raise ValueError("validation loader produced no batches")
        return total / batches
    finally:
        model.load_state_dict(backup, strict=True)
        model.train(was_training)


def experiment_metadata(
    config: ExperimentConfig,
    data_root: Path,
    train_dataset: RobotSpeechDataset,
    val_dataset: RobotSpeechDataset,
) -> dict[str, Any]:
    stats_path = data_root / config.data.stats_file
    return {
        "schema": "echo-g-sgdit-experiment-v1",
        "config": config.to_dict(),
        "data_release": data_root.name,
        "train_split": config.data.train_split,
        "train_split_sha256": sha256(train_dataset.split_path),
        "val_split": config.data.val_split,
        "val_split_sha256": sha256(val_dataset.split_path),
        "stats_file": config.data.stats_file,
        "stats_sha256": sha256(stats_path),
        "train_clips": len(train_dataset),
        "val_clips": len(val_dataset),
        "precision": "fp32",
        "amp": False,
        "architecture": config.model.architecture,
        "conditioning": config.data.conditioning,
        "max_frames": config.data.max_frames,
        "max_text_tokens": config.data.max_text_tokens,
        "wordtime_schema": WORD_TIME_SCHEMA if config.model.architecture == "v2" else None,
        "wordtime_mode": "qknorm_wordtime" if config.model.architecture == "v2" else None,
        "wordtime_config": dict(WORD_TIME_CONFIG) if config.model.architecture == "v2" else None,
        "time_distance_convention": (
            "signed_frame_minus_token_center_seconds"
            if config.model.architecture == "v2"
            else "none"
        ),
        "validation_drop_last": True,
        "validation_loss_clips": (
            len(val_dataset) // config.training.batch_size * config.training.batch_size
        ),
    }


def inference_checkpoint(
    ema: dict[str, torch.Tensor],
    mean: torch.Tensor,
    std: torch.Tensor,
    experiment: dict[str, Any],
    global_step: int,
    validation: float,
) -> dict[str, Any]:
    return {
        "schema": "echo-g-sgdit-checkpoint-v1",
        "global_step": global_step,
        "validation": validation,
        "ema": cpu_state(ema),
        "motion_mean": mean.cpu(),
        "motion_std": std.cpu(),
        "experiment": experiment,
    }


def save_resume_checkpoint(
    path: Path,
    inference: dict[str, Any],
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    epoch: int,
    next_batch_index: int,
    best_validation: float,
) -> None:
    payload = {
        **inference,
        "model": cpu_state(model.state_dict()),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "next_batch_index": next_batch_index,
        "best_validation": best_validation,
        "torch_rng_state": torch.get_rng_state(),
        "cuda_rng_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }
    atomic_torch_save(payload, path)


def run_training(
    config: ExperimentConfig,
    data_root: Path,
    output_dir: Path,
    device_name: str,
    resume: Path | None = None,
    max_train: int = 0,
    max_val: int = 0,
    longest_first: bool = False,
) -> Path:
    config.validate()
    if config.model.motion_dim == 136:
        raise ValueError("HumanRetarget is inference-only in this release")
    configure_reproducibility(config.training.seed)
    device = resolve_device(device_name)
    output_dir.mkdir(parents=True, exist_ok=True)
    mean, std = load_motion_stats(data_root / config.data.stats_file, config.model.motion_dim)
    train_dataset, val_dataset, train_sampler, train_loader, val_loader = build_loaders(
        config,
        data_root,
        mean,
        std,
        device,
        max_train,
        max_val,
        longest_first,
    )
    model = SpeechGroundedDiT(config.model).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.training.max_optimizer_steps, eta_min=0.0
    )
    ema = {name: value.detach().clone() for name, value in model.state_dict().items()}
    experiment = experiment_metadata(config, data_root, train_dataset, val_dataset)
    atomic_json_save(experiment, output_dir / "experiment.json")

    epoch = 0
    next_batch_index = 0
    global_step = 0
    best_validation = float("inf")
    if resume is not None:
        checkpoint = torch.load(resume, map_location="cpu", weights_only=False)
        previous = checkpoint["experiment"]
        for key in ("train_split_sha256", "val_split_sha256", "stats_sha256"):
            if previous.get(key) != experiment.get(key):
                raise ValueError(
                    f"resume mismatch for {key}: {previous.get(key)} != {experiment.get(key)}"
                )
        if previous.get("config") != experiment.get("config"):
            raise ValueError("resume checkpoint uses a different experiment configuration")
        model.load_state_dict(checkpoint["model"], strict=True)
        ema = {name: value.to(device) for name, value in checkpoint["ema"].items()}
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        epoch = int(checkpoint["epoch"])
        next_batch_index = int(checkpoint["next_batch_index"])
        global_step = int(checkpoint["global_step"])
        best_validation = float(checkpoint.get("best_validation", best_validation))
        torch.set_rng_state(checkpoint["torch_rng_state"])
        if device.type == "cuda" and checkpoint.get("cuda_rng_state"):
            torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state"])
        LOGGER.info(
            "resumed %s at epoch=%d batch=%d global_step=%d",
            resume,
            epoch,
            next_batch_index,
            global_step,
        )

    parameters = sum(parameter.numel() for parameter in model.parameters())
    LOGGER.info(
        "train=%d val=%d params=%.2fM batch=%d accumulation=%d max_steps=%d device=%s",
        len(train_dataset),
        len(val_dataset),
        parameters / 1e6,
        config.training.batch_size,
        config.training.gradient_accumulation,
        config.training.max_optimizer_steps,
        device,
    )
    optimizer.zero_grad(set_to_none=True)
    accumulation = 0
    reached_step_limit = global_step >= config.training.max_optimizer_steps
    last_validation = float("nan")
    # Recreating an iterator midway through an epoch consumes a new CPU base
    # seed. The interrupted run already consumed it; Dataset loading is purely
    # deterministic, so preserve the training RNG during that reconstruction.
    preserve_iterator_rng = resume is not None and (
        next_batch_index > 0 or (train_loader.persistent_workers and epoch > 0)
    )

    while epoch < config.training.max_epochs and not reached_step_limit:
        train_sampler.set_epoch(epoch)
        model.train()
        epoch_losses = torch.zeros(3, dtype=torch.float64)
        epoch_batches = 0
        if preserve_iterator_rng:
            with torch.random.fork_rng(devices=[]):
                train_iterator = iter(train_loader)
            preserve_iterator_rng = False
        else:
            train_iterator = iter(train_loader)
        for batch_index, batch in enumerate(train_iterator):
            if batch_index < next_batch_index:
                continue
            batch = move_batch(batch, device)
            total, flow_loss, temporal_loss = flow_matching_loss(
                model,
                batch,
                config.training.condition_dropout,
                config.training.temporal_loss_weight,
            )
            if not torch.isfinite(total):
                raise FloatingPointError(
                    f"non-finite loss at epoch={epoch} batch={batch_index} stems={batch['stems']}"
                )
            (total / config.training.gradient_accumulation).backward()
            accumulation += 1
            epoch_losses += torch.tensor(
                [total.item(), flow_loss.item(), temporal_loss.item()], dtype=torch.float64
            )
            epoch_batches += 1
            if accumulation < config.training.gradient_accumulation:
                continue

            gradient_norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), config.training.gradient_clip, error_if_nonfinite=False
            )
            if not torch.isfinite(gradient_norm):
                raise FloatingPointError(
                    f"non-finite gradient at epoch={epoch} batch={batch_index}"
                )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            update_ema(ema, model, config.training.ema_decay)
            accumulation = 0
            global_step += 1
            resume_epoch = epoch
            resume_batch = batch_index + 1
            if resume_batch >= len(train_loader):
                resume_epoch += 1
                resume_batch = 0

            if global_step % config.training.log_every_steps == 0:
                LOGGER.info(
                    "epoch=%d batch=%d global_step=%d/%d loss=%.6f grad=%.4f lr=%.3e",
                    epoch,
                    batch_index,
                    global_step,
                    config.training.max_optimizer_steps,
                    total.item(),
                    gradient_norm.item(),
                    scheduler.get_last_lr()[0],
                )

            should_evaluate = (
                config.training.evaluate_every_steps > 0
                and global_step % config.training.evaluate_every_steps == 0
            )
            if should_evaluate:
                last_validation = evaluate(
                    model,
                    ema,
                    val_loader,
                    device,
                    config.training.temporal_loss_weight,
                    config.training.validation_batches,
                )
                LOGGER.info("global_step=%d ema_validation=%.6f", global_step, last_validation)

            should_save = (
                config.training.save_every_steps > 0
                and global_step % config.training.save_every_steps == 0
            )
            if should_save or should_evaluate:
                snapshot = inference_checkpoint(
                    ema,
                    mean,
                    std,
                    experiment,
                    global_step,
                    last_validation,
                )
                if should_evaluate and last_validation < best_validation:
                    best_validation = last_validation
                    snapshot["validation"] = best_validation
                    atomic_torch_save(snapshot, output_dir / "best.pt")
                save_resume_checkpoint(
                    output_dir / "latest.pt",
                    snapshot,
                    model,
                    optimizer,
                    scheduler,
                    resume_epoch,
                    resume_batch,
                    best_validation,
                )
                if should_save and config.training.keep_step_checkpoints:
                    atomic_torch_save(snapshot, output_dir / f"step_{global_step:06d}.pt")

            if global_step >= config.training.max_optimizer_steps:
                reached_step_limit = True
                epoch = resume_epoch
                next_batch_index = resume_batch
                break
        if reached_step_limit:
            break
        averaged = epoch_losses / max(epoch_batches, 1)
        LOGGER.info(
            "epoch=%d mean_total=%.6f mean_flow=%.6f mean_temporal=%.6f",
            epoch,
            averaged[0],
            averaged[1],
            averaged[2],
        )
        epoch += 1
        next_batch_index = 0

    if not reached_step_limit:
        raise RuntimeError(
            f"max_epochs={config.training.max_epochs} reached at global_step={global_step}; "
            f"expected {config.training.max_optimizer_steps}"
        )

    last_validation = evaluate(
        model,
        ema,
        val_loader,
        device,
        config.training.temporal_loss_weight,
        config.training.validation_batches,
    )
    final_snapshot = inference_checkpoint(
        ema,
        mean,
        std,
        experiment,
        global_step,
        last_validation,
    )
    if last_validation < best_validation:
        best_validation = last_validation
        atomic_torch_save(final_snapshot, output_dir / "best.pt")
    atomic_torch_save(final_snapshot, output_dir / "final.pt")
    save_resume_checkpoint(
        output_dir / "latest.pt",
        final_snapshot,
        model,
        optimizer,
        scheduler,
        epoch,
        next_batch_index,
        best_validation,
    )
    atomic_json_save(
        {
            "status": "complete",
            "global_step": global_step,
            "final_validation": last_validation,
            "best_validation": best_validation,
        },
        output_dir / "complete.json",
    )
    LOGGER.info(
        "training complete at global_step=%d final_val=%.6f best_val=%.6f",
        global_step,
        last_validation,
        best_validation,
    )
    return output_dir / "best.pt"


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    if config.model.motion_dim != 39:
        raise ValueError(
            "Training supports direct robot motion only; HumanRetarget is inference-only"
        )
    run_training(
        config,
        args.data_root,
        args.output_dir,
        args.device,
        args.resume,
        args.max_train,
        args.max_val,
        args.longest_first,
    )


if __name__ == "__main__":
    main()
