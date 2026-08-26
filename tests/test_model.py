# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import torch

from echo_g.config import ModelConfig
from echo_g.model import SpeechGroundedDiT


def test_sgdit_accepts_variable_sequence_lengths() -> None:
    config = ModelConfig(
        motion_dim=5,
        audio_dim=8,
        text_dim=12,
        hidden_dim=32,
        num_layers=2,
        num_heads=4,
        feedforward_dim=64,
    )
    model = SpeechGroundedDiT(config).eval()
    for frames in (7, 19):
        output = model(
            torch.randn(2, frames, 5),
            torch.rand(2),
            torch.randn(2, frames, 8),
            torch.randn(2, 3, 12),
            torch.ones(2, frames, dtype=torch.bool),
            torch.zeros(2, 3, dtype=torch.bool),
            torch.rand(2, frames, 3),
        )
        assert output.shape == (2, frames, 5)
        assert torch.isfinite(output).all()
