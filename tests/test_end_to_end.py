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

from echo_g.config import ExperimentConfig
from echo_g.inference import export_predictions
from echo_g.training import run_training


def test_cpu_training_and_sampling(
    synthetic_release: tuple[Path, ExperimentConfig], tmp_path: Path
) -> None:
    data_root, config = synthetic_release
    run_dir = tmp_path / "run"
    checkpoint = run_training(config, data_root, run_dir, "cpu")
    assert checkpoint.is_file()
    assert (run_dir / "complete.json").is_file()

    prediction_root = tmp_path / "predictions"
    marker = export_predictions(
        checkpoint,
        data_root,
        prediction_root,
        "cpu",
        split="val",
        seeds=[7],
        batch_size=2,
        num_workers=0,
    )
    assert marker.is_file()
    predictions = sorted((prediction_root / "seed_007").glob("*.pt"))
    assert len(predictions) == 2
    payload = torch.load(predictions[0], map_location="cpu", weights_only=True)
    assert payload["representation_units"] == "physical"
    assert payload["robot_repr"].shape[-1] == config.model.motion_dim
    assert torch.isfinite(payload["robot_repr"]).all()
