# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import shutil
from pathlib import Path

import pytest
import torch

from echo_g.config import ExperimentConfig
from echo_g.data import ConditionDataset, collate_motion, decode_condition
from echo_g.inference import export_predictions
from echo_g.model import SpeechGroundedDiT


def checkpoint(path: Path, config: ExperimentConfig) -> Path:
    model = SpeechGroundedDiT(config.model)
    torch.save(
        {
            "ema": model.state_dict(),
            "motion_mean": torch.zeros(config.model.motion_dim),
            "motion_std": torch.ones(config.model.motion_dim),
            "experiment": {"config": config.to_dict()},
        },
        path,
    )
    return path


def test_signed_time_and_no_ground_truth_dataset(
    synthetic_release: tuple[Path, ExperimentConfig],
    tmp_path: Path,
) -> None:
    root, config = synthetic_release
    conditions = tmp_path / "conditions_only"
    conditions.mkdir()
    shutil.copy2(root / "condition_30fps/val_00.pt", conditions / "val_00.pt")
    dataset = ConditionDataset(conditions, ["val_00"], config.data, config.model)
    batch = collate_motion([dataset[0]])
    distance = batch["time_distance"]
    assert distance[0, 0, 2] < 0
    assert distance[0, -1, 0] > 0
    expected = torch.arange(9).float()[:, None] / 30 - torch.tensor([0.05, 0.15, 0.25])
    torch.testing.assert_close(distance[0], expected)
    weights = checkpoint(tmp_path / "weights.pt", config)
    output = tmp_path / "predictions"
    export_predictions(weights, None, output, "cpu", condition_dir=conditions, num_workers=0)
    result = torch.load(output / "seed_000/val_00.pt", weights_only=True)
    assert result["robot_repr"].shape == (9, 5)
    assert result["time_distance_convention"].startswith("signed_")
    assert len(result["source_condition_sha256"]) == 64
    export_predictions(weights, None, output, "cpu", condition_dir=conditions, num_workers=0)
    changed = torch.load(weights, weights_only=True)
    changed["ema"]["output.bias"][0] += 1
    torch.save(changed, weights)
    with pytest.raises(ValueError, match="incompatible existing prediction"):
        export_predictions(weights, None, output, "cpu", condition_dir=conditions, num_workers=0)


def test_reused_prediction_rejects_changed_conditions_and_solver(
    synthetic_release: tuple[Path, ExperimentConfig],
    tmp_path: Path,
) -> None:
    root, config = synthetic_release
    weights = checkpoint(tmp_path / "weights.pt", config)
    output = tmp_path / "predictions"
    export_predictions(weights, root, output, "cpu", num_workers=0)
    with pytest.raises(ValueError, match="incompatible existing prediction"):
        export_predictions(weights, root, output, "cpu", num_workers=0, steps=3)
    condition_path = root / "condition_30fps/val_00.pt"
    payload = torch.load(condition_path, weights_only=True)
    payload["audio_features"] += 0.1
    torch.save(payload, condition_path)
    with pytest.raises(ValueError, match="incompatible existing prediction"):
        export_predictions(weights, root, output, "cpu", num_workers=0)


@pytest.mark.parametrize("defect", ["truncated", "overflow", "missing_time", "bad_time", "nan"])
def test_condition_integrity(
    synthetic_release: tuple[Path, ExperimentConfig],
    defect: str,
) -> None:
    root, config = synthetic_release
    payload = torch.load(root / "condition_30fps/val_00.pt", weights_only=True)
    if defect == "truncated":
        payload["n_tokens"] = 150
    elif defect == "overflow":
        payload["text_tokens"] = torch.zeros(257, 12)
    elif defect == "missing_time":
        payload["has_word_timing"] = False
    elif defect == "bad_time":
        payload["token_times"][0] = torch.tensor([1.0, 0.0])
    elif defect == "nan":
        payload["audio_features"][0, 0] = float("nan")
    with pytest.raises(ValueError):
        decode_condition(payload, "example", config.data, config.model)
