# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

from pathlib import Path

import torch

from echo_g.conditions import token_time_spans
from echo_g.config import ExperimentConfig
from echo_g.data import RobotSpeechDataset, collate_motion, load_motion_stats


def test_dataset_and_padding(
    synthetic_release: tuple[Path, ExperimentConfig],
) -> None:
    root, config = synthetic_release
    mean, std = load_motion_stats(root / config.data.stats_file, config.model.motion_dim)
    dataset = RobotSpeechDataset(root, config.data, config.model, "train", mean, std)
    batch = collate_motion([dataset[0], dataset[1]])
    assert batch["motion"].shape == (2, 6, 5)
    assert batch["audio"].shape == (2, 6, 8)
    assert batch["text"].shape == (2, 3, 12)
    assert batch["motion_mask"].sum(dim=1).tolist() == [5, 6]
    assert torch.isfinite(batch["time_distance"]).all()


def test_word_timestamps_follow_token_offsets() -> None:
    transcript = "hello world"
    words = (
        {"text": "hello", "start": 0.1, "end": 0.4},
        {"text": "world", "start": 0.6, "end": 1.0},
    )
    spans = token_time_spans(transcript, [(0, 5), (5, 6), (6, 11)], words)
    assert torch.allclose(spans[0], torch.tensor([0.1, 0.4]))
    assert torch.allclose(spans[1], torch.tensor([0.4, 0.4]))
    assert torch.allclose(spans[2], torch.tensor([0.6, 1.0]))
