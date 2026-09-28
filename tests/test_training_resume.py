# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from echo_g import training
from echo_g.config import ExperimentConfig
from echo_g.data import load_motion_stats
from echo_g.model import SpeechGroundedDiT


class InterruptedTraining(RuntimeError):
    pass


def _assert_equal(left: Any, right: Any) -> None:
    if isinstance(left, torch.Tensor):
        assert torch.equal(left, right)
    elif isinstance(left, dict):
        assert left.keys() == right.keys()
        for key in left:
            _assert_equal(left[key], right[key])
    elif isinstance(left, (list, tuple)):
        assert len(left) == len(right)
        for first, second in zip(left, right, strict=True):
            _assert_equal(first, second)
    else:
        assert left == right


@pytest.mark.parametrize(("accumulation", "interrupt_step"), [(1, 1), (1, 2), (3, 1)])
def test_resume_matches_uninterrupted_training(
    synthetic_release: tuple[Path, ExperimentConfig],
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    accumulation: int,
    interrupt_step: int,
) -> None:
    data_root, base_config = synthetic_release
    config = replace(
        base_config,
        training=replace(
            base_config.training,
            max_optimizer_steps=4,
            max_epochs=8,
            gradient_accumulation=accumulation,
        ),
    )
    full_dir, resumed_dir = tmp_path / "full", tmp_path / "resumed"
    training.run_training(config, data_root, full_dir, "cpu")
    original_save = training.save_resume_checkpoint

    def save_then_interrupt(*args: Any, **kwargs: Any) -> None:
        original_save(*args, **kwargs)
        if args[1]["global_step"] == interrupt_step:
            raise InterruptedTraining()

    monkeypatch.setattr(training, "save_resume_checkpoint", save_then_interrupt)
    with pytest.raises(InterruptedTraining):
        training.run_training(config, data_root, resumed_dir, "cpu")
    monkeypatch.setattr(training, "save_resume_checkpoint", original_save)
    training.run_training(config, data_root, resumed_dir, "cpu", resume=resumed_dir / "latest.pt")
    full = torch.load(full_dir / "latest.pt", weights_only=True)
    resumed = torch.load(resumed_dir / "latest.pt", weights_only=True)
    for key in (
        "model",
        "ema",
        "optimizer",
        "scheduler",
        "torch_rng_state",
        "global_step",
        "validation",
        "best_validation",
        "epoch",
        "next_batch_index",
    ):
        _assert_equal(full[key], resumed[key])
    full_best = torch.load(full_dir / "best.pt", weights_only=True)
    resumed_best = torch.load(resumed_dir / "best.pt", weights_only=True)
    _assert_equal(full_best, resumed_best)


def test_evaluation_restores_cpu_model_rng_and_mode(
    synthetic_release: tuple[Path, ExperimentConfig],
) -> None:
    data_root, config = synthetic_release
    mean, std = load_motion_stats(data_root / config.data.stats_file, config.model.motion_dim)
    *_, val_loader = training.build_loaders(
        config, data_root, mean, std, torch.device("cpu"), 0, 0, False
    )
    model = SpeechGroundedDiT(config.model).train()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    ema = {key: value + 0.02 for key, value in before.items()}
    rng = torch.get_rng_state().clone()
    value = training.evaluate(model, ema, val_loader, torch.device("cpu"), 0.5, 0)
    assert value > 0 and model.training
    _assert_equal(before, model.state_dict())
    assert torch.equal(rng, torch.get_rng_state())
    model.eval()
    with pytest.raises(ValueError, match="no batches"):
        training.evaluate(model, ema, [], torch.device("cpu"), 0.5, 0)
    assert not model.training
    _assert_equal(before, model.state_dict())


def test_historical_validation_drop_last_is_explicit(
    synthetic_release: tuple[Path, ExperimentConfig],
) -> None:
    data_root, config = synthetic_release
    mean, std = load_motion_stats(data_root / config.data.stats_file, config.model.motion_dim)
    train, val, sampler, train_loader, val_loader = training.build_loaders(
        config,
        data_root,
        mean,
        std,
        torch.device("cpu"),
        0,
        0,
        False,
    )
    assert sampler.drop_last and val_loader.batch_sampler.drop_last
    metadata = training.experiment_metadata(config, data_root, train, val)
    assert metadata["validation_drop_last"]
    assert metadata["precision"] == "fp32" and not metadata["amp"]
    assert metadata["time_distance_convention"] == "signed_frame_minus_token_center_seconds"
    assert metadata["wordtime_mode"] == "qknorm_wordtime"
