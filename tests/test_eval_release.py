# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import json
import sys
import wave
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from echo_g.evaluation import g1_motion_cls as evaluator
from echo_g.evaluation.g1_kinematics import G1_BFS_PARENTS
from echo_g.evaluation.g1_motion_cls import (
    Jerk,
    Multimodality,
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


@pytest.fixture
def synthetic_encoder(tmp_path: Path) -> Path:
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
    return checkpoint


def test_fgd_checkpoint_load_and_feature_path_need_no_parent_repository(
    synthetic_encoder: Path,
) -> None:
    loaded, loaded_config = load_g1_ae(str(synthetic_encoder), "cpu", 30)
    motion = torch.zeros(64, 39)
    motion[:, :6] = matrix_to_rotation_6d(torch.eye(3))
    features = map2latent_feature(loaded, loaded_config, motion.numpy(), "cpu")
    assert features is not None
    assert features.shape[1] == 192
    assert np.isfinite(features).all()
    assert not loaded.training


def test_public_benchmark_metrics_and_required_audio(
    tmp_path: Path,
    synthetic_encoder: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("librosa")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for directory in ["pred", "ref", "audio"]:
        (tmp_path / directory).mkdir()
    mmae = tmp_path / "mmae.npy"
    np.save(mmae, np.full(30, 0.1, dtype=np.float32))
    random = np.random.default_rng(42)
    for index, frames in enumerate([64, 96]):
        stem = f"clip_{index}"
        times = torch.arange(frames) / 30
        motion = torch.zeros(frames, 39)
        motion[:, :6] = matrix_to_rotation_6d(torch.eye(3))
        motion[:, 10:] = 0.15 * torch.sin(times[:, None] * torch.arange(1, 30)[None, :])
        prediction = motion.clone()
        prediction[:, 10:] *= 0.9
        for directory, values in [("ref", motion), ("pred", prediction)]:
            torch.save(
                {"robot_repr": values, "representation_units": "physical", "fps": 30},
                tmp_path / directory / f"{stem}.pt",
            )
        for run in range(20):
            run_dir = tmp_path / "mm20" / f"run_{run:03d}"
            run_dir.mkdir(parents=True, exist_ok=True)
            sampled = prediction.clone()
            sampled[:, 10:] += torch.from_numpy(
                random.normal(0, 0.005, (frames, 29)).astype(np.float32)
            )
            torch.save({"robot_repr": sampled, "fps": 30}, run_dir / f"{stem}.pt")
        waveform = np.zeros(round(frames / 30 * 16000), dtype=np.int16)
        for offset in range(3200, len(waveform) - 1600, 6400):
            waveform[offset : offset + 800] = (
                20000 * np.sin(np.arange(800) * 2 * np.pi * 880 / 16000)
            ).astype(np.int16)
        with wave.open(str(tmp_path / "audio" / f"{stem}.wav"), "wb") as handle:
            handle.setnchannels(1)
            handle.setsampwidth(2)
            handle.setframerate(16000)
            handle.writeframes(waveform.tobytes())

    args = [
        "echo-g-eval",
        "--pred-dir",
        str(tmp_path / "pred"),
        "--ref-dir",
        str(tmp_path / "ref"),
        "--wav-dir",
        str(tmp_path / "audio"),
        "--mmae-file",
        str(mmae),
        "--g1-ae-ckpt",
        str(synthetic_encoder),
        "--require-all-stems",
        "--require-equal-lengths",
        "--enable-foot-metrics",
        "--multimodality-root",
        str(tmp_path / "mm20"),
        "--require-all-multimodality",
        "--out",
        str(tmp_path / "result.json"),
    ]
    monkeypatch.setattr(sys, "argv", args)
    evaluator.main()
    public = json.loads((tmp_path / "result.json").read_text())
    assert public["n_clips"] == 2
    assert not any(key.lower().startswith("srgr") for key in public["params"])
    assert not any(key.startswith("SRGR") for key in public["metrics"])
    metrics = public["metrics"]
    assert np.isfinite(metrics["FGD"])
    assert metrics["BA"]["n"] == 2
    assert metrics["Jerk_length_weighted"]["mean"] > 0
    assert metrics["Multimodality"] > 0
    assert metrics["Multimodality_status"] == "COMPLETE"
    assert metrics["foot_ground_error"]["n"] == 2
    assert metrics["contact_sliding_speed"]["n"] == 2

    (tmp_path / "audio" / "clip_0.wav").unlink()
    with pytest.raises(FileNotFoundError, match="BA audio is missing"):
        evaluator.main()
