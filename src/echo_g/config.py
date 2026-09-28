# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any, TypeVar

import yaml


@dataclass(frozen=True)
class DataConfig:
    fps: int = 30
    max_frames: int = 600
    max_text_tokens: int = 256
    condition_dir: str = "condition_30fps"
    motion_dir: str = "motion_39d_30fps"
    train_split: str = "splits/train_drop.txt"
    val_split: str = "splits/val_common.txt"
    stats_file: str = "stats/drop_train.pt"
    aligned_lengths_file: str | None = "audit/condition/per_clip_condition_motion_audit.csv"
    representation_schema: str = "projecthermes-g1-39d-standard-v2"
    conditioning: str = "audio-text"

    def validate(self) -> None:
        if self.fps <= 0:
            raise ValueError("data.fps must be positive")
        if self.max_frames < 2:
            raise ValueError("data.max_frames must be at least 2")
        if not 1 <= self.max_text_tokens <= 256:
            raise ValueError("data.max_text_tokens must be in [1, 256]")
        if self.max_frames > 600:
            raise ValueError("V2 data.max_frames cannot exceed 600")
        if self.conditioning not in {"audio-text", "text-only"}:
            raise ValueError("data.conditioning must be audio-text or text-only")


@dataclass(frozen=True)
class ModelConfig:
    motion_dim: int = 39
    audio_dim: int = 1024
    text_dim: int = 2560
    hidden_dim: int = 768
    num_layers: int = 12
    num_heads: int = 8
    feedforward_dim: int = 2048
    dropout: float = 0.0
    architecture: str = "v2"
    position_encoding: str = "learned"
    max_t: int = 608
    max_text_tokens: int = 256

    def validate(self) -> None:
        if self.architecture != "v2" or self.position_encoding != "learned":
            raise ValueError("this branch supports V2 with learned position encoding only")
        if not 1 <= self.max_text_tokens <= 256:
            raise ValueError("model.max_text_tokens must be in [1, 256]")
        dimensions = {
            "max_t": self.max_t,
            "motion_dim": self.motion_dim,
            "audio_dim": self.audio_dim,
            "text_dim": self.text_dim,
            "hidden_dim": self.hidden_dim,
            "num_layers": self.num_layers,
            "num_heads": self.num_heads,
            "feedforward_dim": self.feedforward_dim,
        }
        invalid = [name for name, value in dimensions.items() if value <= 0]
        if invalid:
            raise ValueError(f"model dimensions must be positive: {invalid}")
        if self.hidden_dim % self.num_heads:
            raise ValueError("model.hidden_dim must be divisible by model.num_heads")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("model.dropout must be in [0, 1)")


@dataclass(frozen=True)
class TrainingConfig:
    batch_size: int = 3
    gradient_accumulation: int = 16
    max_optimizer_steps: int = 63_000
    max_epochs: int = 400
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    condition_dropout: float = 0.1
    temporal_loss_weight: float = 0.5
    ema_decay: float = 0.999
    gradient_clip: float = 1.0
    seed: int = 20_260_825
    num_workers: int = 8
    bucket_multiplier: int = 64
    log_every_steps: int = 50
    evaluate_every_steps: int = 5_000
    save_every_steps: int = 5_000
    validation_batches: int = 0
    keep_step_checkpoints: bool = False

    def validate(self) -> None:
        positive = {
            "batch_size": self.batch_size,
            "gradient_accumulation": self.gradient_accumulation,
            "max_optimizer_steps": self.max_optimizer_steps,
            "max_epochs": self.max_epochs,
            "learning_rate": self.learning_rate,
            "ema_decay": self.ema_decay,
            "gradient_clip": self.gradient_clip,
            "bucket_multiplier": self.bucket_multiplier,
        }
        invalid = [name for name, value in positive.items() if value <= 0]
        if invalid:
            raise ValueError(f"training values must be positive: {invalid}")
        if not 0.0 <= self.condition_dropout < 1.0:
            raise ValueError("training.condition_dropout must be in [0, 1)")
        if not 0.0 < self.ema_decay <= 1.0:
            raise ValueError("training.ema_decay must be in (0, 1]")
        if self.num_workers < 0:
            raise ValueError("training.num_workers cannot be negative")


@dataclass(frozen=True)
class SamplingConfig:
    integration_steps: int = 8
    guidance_scale: float = 1.0

    def validate(self) -> None:
        if self.integration_steps <= 0:
            raise ValueError("sampling.integration_steps must be positive")
        if self.guidance_scale < 0:
            raise ValueError("sampling.guidance_scale cannot be negative")


ConfigSection = TypeVar("ConfigSection", DataConfig, ModelConfig, TrainingConfig, SamplingConfig)


def _section(section_type: type[ConfigSection], values: dict[str, Any]) -> ConfigSection:
    allowed = {field.name for field in fields(section_type)}
    unknown = sorted(set(values) - allowed)
    if unknown:
        raise ValueError(f"unknown {section_type.__name__} keys: {unknown}")
    return section_type(**values)


@dataclass(frozen=True)
class ExperimentConfig:
    data: DataConfig
    model: ModelConfig
    training: TrainingConfig
    sampling: SamplingConfig

    @classmethod
    def from_yaml(cls, path: Path) -> ExperimentConfig:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError(f"{path}: expected a YAML mapping")
        expected = {"data", "model", "training", "sampling"}
        missing = sorted(expected - payload.keys())
        unknown = sorted(payload.keys() - expected)
        if missing or unknown:
            raise ValueError(f"{path}: missing={missing}, unknown={unknown}")
        config = cls(
            data=_section(DataConfig, payload["data"]),
            model=_section(ModelConfig, payload["model"]),
            training=_section(TrainingConfig, payload["training"]),
            sampling=_section(SamplingConfig, payload["sampling"]),
        )
        config.validate()
        return config

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> ExperimentConfig:
        config = cls(
            data=_section(DataConfig, payload["data"]),
            model=_section(ModelConfig, payload["model"]),
            training=_section(TrainingConfig, payload["training"]),
            sampling=_section(SamplingConfig, payload["sampling"]),
        )
        config.validate()
        return config

    def validate(self) -> None:
        self.data.validate()
        self.model.validate()
        self.training.validate()
        self.sampling.validate()
        if self.data.max_frames > self.model.max_t:
            raise ValueError("data.max_frames exceeds model.max_t")
        if self.data.max_text_tokens > self.model.max_text_tokens:
            raise ValueError("data.max_text_tokens exceeds model.max_text_tokens")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
