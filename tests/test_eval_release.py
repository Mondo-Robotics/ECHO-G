# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from echo_g.evaluation.g1_kinematics import G1_BFS_PARENTS
from echo_g.evaluation.g1_motion_cls import (
    Jerk,
    Multimodality,
    SRGRMass,
    load_g1_ae,
    load_repr,
    map2latent_feature,
)
from echo_g.evaluation.robot_fgd.motion_encoder import VAESKConv
from echo_g.evaluation.rotation import matrix_to_rotation_6d


def test_mm20_matches_explicit_unordered_pairs_and_rejects_missing_tail() -> None:
    random = np.random.default_rng(18)
    samples = [random.normal(size=(5, 30, 3)) for _ in range(20)]
    expected = np.mean([np.abs(a - b).sum(axis=(1, 2)).mean() for a, b in combinations(samples, 2)])
    metric = Multimodality(expected_runs=20)
    pair_mean, literal = metric.update(samples)
    assert pair_mean == pytest.approx(expected)
    assert metric.avg() == pytest.approx(expected)
    assert literal == pytest.approx(expected * 190 / 200)
    samples[-1] = samples[-1][:-1]
    with pytest.raises(ValueError, match="exactly equal"):
        metric.update(samples)


def test_weighted_jerk_uses_valid_difference_count() -> None:
    def trajectory(frames: int, jerk: float) -> np.ndarray:
        t = np.arange(frames, dtype=np.float64) / 30
        values = np.zeros((frames, 30, 3))
        values[..., 0] = (jerk * t**3 / 6)[:, None]
        return values

    metric = Jerk(30)
    for frames, jerk in [(8, 2), (18, 6)]:
        predicted = trajectory(frames, jerk)
        metric.update(predicted, np.zeros_like(predicted))
    clip_equal, _ = metric.summary()
    weighted, ground_truth = metric.summary_length_weighted()
    assert clip_equal["mean"] == pytest.approx(4)
    assert weighted["mean"] == pytest.approx(5)
    assert ground_truth["mean"] == 0


def test_srgr_normalizes_evaluated_semantic_mass() -> None:
    metric = SRGRMass()
    truth = np.zeros((2, 30, 3))
    prediction = truth.copy()
    prediction[1, :, 0] = 0.2
    metric.run(prediction, truth, np.array([0.25, 0.75]))
    assert metric.avg() == pytest.approx(0.25)
    metric.reset()
    metric.run(truth, truth, np.array([0.25, 0.75]))
    assert metric.avg() == 1


def test_physical_motion_metadata_is_enforced(tmp_path: Path) -> None:
    path = tmp_path / "sample.pt"
    torch.save(
        {"robot_repr": torch.zeros(8, 39), "representation_units": "normalized", "fps": 30}, path
    )
    with pytest.raises(ValueError, match="physical"):
        load_repr(str(path), 30)
    torch.save(
        {"robot_repr": torch.zeros(8, 39), "representation_units": "physical", "fps": 25}, path
    )
    with pytest.raises(ValueError, match="fps"):
        load_repr(str(path), 30)


def test_fgd_checkpoint_load_and_feature_path_need_no_parent_repository(tmp_path: Path) -> None:
    config = SimpleNamespace(
        vae_layer=4,
        vae_grow=[2, 2, 2, 2],
        variational=False,
        vae_test_dim=29,
        vae_length=192,
        channel_base=2,
    )
    model = VAESKConv(config, topology=G1_BFS_PARENTS).eval()
    checkpoint = tmp_path / "synthetic_encoder.bin"
    torch.save(
        {
            "model_state": model.state_dict(),
            "vae_layer": 4,
            "vae_grow": [2, 2, 2, 2],
            "vae_test_dim": 29,
            "vae_length": 192,
            "channel_base": 2,
            "input_mode": "pure2",
            "input_normalization": "none",
            "fps": 30,
        },
        checkpoint,
    )
    loaded, loaded_config = load_g1_ae(str(checkpoint), "cpu", 30)
    motion = torch.zeros(64, 39)
    motion[:, :6] = matrix_to_rotation_6d(torch.eye(3))
    features = map2latent_feature(loaded, loaded_config, motion.numpy(), "cpu")
    assert features is not None
    assert features.shape[1] == 192
    assert np.isfinite(features).all()
    assert not loaded.training
