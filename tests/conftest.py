# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import csv
from pathlib import Path

import pytest
import torch

from echo_g.config import (
    DataConfig,
    ExperimentConfig,
    ModelConfig,
    SamplingConfig,
    TrainingConfig,
)


@pytest.fixture
def synthetic_release(tmp_path: Path) -> tuple[Path, ExperimentConfig]:
    root = tmp_path / "synthetic_release"
    condition_root = root / "condition_30fps"
    motion_root = root / "motion_39d_30fps"
    split_root = root / "splits"
    stats_root = root / "stats"
    audit_root = root / "audit/condition"
    for path in (condition_root, motion_root, split_root, stats_root, audit_root):
        path.mkdir(parents=True, exist_ok=True)

    train_stems = [f"train_{index:02d}" for index in range(4)]
    val_stems = [f"val_{index:02d}" for index in range(2)]
    all_stems = train_stems + val_stems
    lengths: dict[str, int] = {}
    for index, stem in enumerate(all_stems):
        frames = 5 + index
        lengths[stem] = frames
        generator = torch.Generator().manual_seed(index)
        torch.save(
            {
                "audio_features": torch.randn(frames, 8, generator=generator),
                "text_tokens": torch.randn(3, 12, generator=generator),
                "text_pooled": torch.randn(12, generator=generator),
                "token_times": torch.tensor([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]]),
                "has_word_timing": True,
            },
            condition_root / f"{stem}.pt",
        )
        torch.save(
            {
                "robot_repr": torch.randn(frames, 5, generator=generator),
                "real_num_frames": frames,
                "fps": 30.0,
            },
            motion_root / f"{stem}.pt",
        )
    (split_root / "train.txt").write_text("\n".join(train_stems) + "\n")
    (split_root / "val.txt").write_text("\n".join(val_stems) + "\n")
    torch.save({"mean": torch.zeros(5), "std": torch.ones(5)}, stats_root / "train.pt")
    with (audit_root / "lengths.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["stem", "aligned_frames", "error"])
        writer.writeheader()
        for stem in all_stems:
            writer.writerow({"stem": stem, "aligned_frames": lengths[stem], "error": ""})

    config = ExperimentConfig(
        data=DataConfig(
            max_frames=16,
            condition_dir="condition_30fps",
            motion_dir="motion_39d_30fps",
            train_split="splits/train.txt",
            val_split="splits/val.txt",
            stats_file="stats/train.pt",
            aligned_lengths_file="audit/condition/lengths.csv",
            representation_schema="synthetic-robot-v1",
        ),
        model=ModelConfig(
            motion_dim=5,
            audio_dim=8,
            text_dim=12,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            feedforward_dim=64,
        ),
        training=TrainingConfig(
            batch_size=2,
            gradient_accumulation=1,
            max_optimizer_steps=2,
            max_epochs=2,
            learning_rate=1e-3,
            num_workers=0,
            bucket_multiplier=2,
            log_every_steps=1,
            evaluate_every_steps=1,
            save_every_steps=1,
        ),
        sampling=SamplingConfig(integration_steps=2, guidance_scale=1.0),
    )
    return root, config
