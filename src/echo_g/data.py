# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import csv
import math
import os
import random
import re
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset, Sampler

from echo_g.condition_provenance import (
    FrozenConditionManifest,
    load_condition_manifest,
    validate_condition_provenance,
)
from echo_g.config import DataConfig, ModelConfig


def read_stems(path: Path) -> list[str]:
    stems = [line.strip() for line in path.read_text(encoding="utf-8").splitlines()]
    stems = [stem for stem in stems if stem]
    if not stems:
        raise ValueError(f"{path}: split is empty")
    if len(stems) != len(set(stems)):
        raise ValueError(f"{path}: split contains duplicate stems")
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", stem) for stem in stems):
        raise ValueError(f"{path}: split contains an unsafe stem")
    return stems


def available_stems(path: Path) -> set[str]:
    if not path.is_dir():
        raise FileNotFoundError(path)
    return {
        entry.name[:-3]
        for entry in os.scandir(path)
        if entry.is_file() and entry.name.endswith(".pt")
    }


def load_aligned_lengths(path: Path) -> dict[str, int]:
    lengths: dict[str, int] = {}
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            stem = row.get("stem", "")
            if not stem:
                raise ValueError(f"{path}: row without a stem")
            if row.get("error"):
                raise ValueError(f"{path}: audited error for {stem}: {row['error']}")
            if stem in lengths:
                raise ValueError(f"{path}: duplicate audited stem {stem}")
            lengths[stem] = int(row["aligned_frames"])
            if lengths[stem] < 2:
                raise ValueError(f"{path}: invalid aligned length for {stem}")
    if not lengths:
        raise ValueError(f"{path}: no aligned lengths")
    return lengths


def load_motion_stats(path: Path, motion_dim: int) -> tuple[torch.Tensor, torch.Tensor]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, dict) or "mean" not in payload or "std" not in payload:
        raise ValueError(f"{path}: expected a mapping containing mean and std")
    mean = torch.as_tensor(payload["mean"]).float().reshape(-1)
    std = torch.as_tensor(payload["std"]).float().reshape(-1)
    if mean.shape != (motion_dim,) or std.shape != (motion_dim,):
        raise ValueError(f"{path}: expected {motion_dim}D statistics, got {mean.shape}/{std.shape}")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
        raise ValueError(f"{path}: statistics contain non-finite values")
    if torch.any(std <= 0):
        raise ValueError(f"{path}: standard deviations must be positive")
    return mean, std


def _audio_features(payload: dict[str, Any]) -> torch.Tensor:
    key = "audio_features" if "audio_features" in payload else "audio_4fps"
    if key not in payload:
        raise KeyError("condition payload has no audio_features")
    return torch.as_tensor(payload[key]).float()


def decode_condition(
    payload: dict[str, Any],
    stem: str,
    config: DataConfig,
    model_config: ModelConfig,
    frames: int | None = None,
    *,
    condition_path: Path | None = None,
    condition_manifest: FrozenConditionManifest | None = None,
) -> dict[str, Any]:
    audio = _audio_features(payload)
    if audio.ndim != 2 or audio.shape[1] != model_config.audio_dim:
        raise ValueError(f"{stem}: invalid audio shape {tuple(audio.shape)}")
    if float(payload.get("fps", payload.get("latent_fps", config.fps))) != config.fps:
        raise ValueError(f"{stem}: condition frame rate differs from the configuration")
    frames = len(audio) if frames is None else frames
    if not 2 <= frames <= config.max_frames or frames > len(audio):
        raise ValueError(f"{stem}: input must fit {config.max_frames} frames; split explicitly")
    if not torch.isfinite(audio).all():
        raise ValueError(f"{stem}: non-finite audio features")
    text = torch.as_tensor(payload["text_tokens"]).float()
    if (
        text.ndim != 2
        or text.shape[1] != model_config.text_dim
        or not 1 <= len(text) <= config.max_text_tokens
    ):
        raise ValueError(
            f"{stem}: expected complete text with at most {config.max_text_tokens} tokens"
        )
    token_count = payload.get("n_tokens")
    if type(token_count) is not int or token_count != len(text):
        raise ValueError(f"{stem}: text cache was truncated; regenerate complete text conditions")
    if not payload.get("has_word_timing") or "token_times" not in payload:
        raise ValueError(f"{stem}: ECHO-G requires word-derived token timestamps")
    times = torch.as_tensor(payload["token_times"]).float()
    if (
        times.shape != (len(text), 2)
        or not torch.isfinite(times).all()
        or bool((times < 0).any())
        or bool((times[:, 1] < times[:, 0]).any())
    ):
        raise ValueError(f"{stem}: invalid token timestamps")
    if not torch.isfinite(text).all():
        raise ValueError(f"{stem}: non-finite text features")
    validate_condition_provenance(payload, stem, condition_path, condition_manifest)
    audio = audio[:frames]
    if config.conditioning == "text-only":
        audio = torch.zeros_like(audio)
    return {
        "audio": audio,
        "text": text,
        "frames": frames,
        "stem": stem,
        "frame_times": torch.arange(frames, dtype=torch.float32) / config.fps,
        "token_centers": times.mean(dim=-1),
        "has_timing": True,
    }


class ConditionDataset(Dataset[dict[str, Any]]):
    """Inference conditions without target motion files or dataset statistics."""

    def __init__(
        self,
        condition_dir: Path,
        stems: list[str],
        config: DataConfig,
        model_config: ModelConfig,
        lengths: dict[str, int] | None = None,
        condition_manifest: Path | None = None,
        condition_manifest_sha256: str | None = None,
    ) -> None:
        self.condition_root = condition_dir
        self.condition_manifest = load_condition_manifest(
            condition_manifest, condition_manifest_sha256
        )
        self.stems = stems
        self.config = config
        self.model_config = model_config
        self.lengths: list[int] = []
        if not stems or len(stems) != len(set(stems)):
            raise ValueError("Expected nonempty, unique condition stems")
        for stem in stems:
            if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", stem):
                raise ValueError(f"Unsafe stem: {stem!r}")
            payload = torch.load(
                condition_dir / f"{stem}.pt", map_location="cpu", weights_only=True
            )
            frames = None if lengths is None else min(lengths[stem], config.max_frames)
            sample = decode_condition(
                payload,
                stem,
                config,
                model_config,
                frames,
                condition_path=condition_dir / f"{stem}.pt",
                condition_manifest=self.condition_manifest,
            )
            self.lengths.append(sample["frames"])

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, Any]:
        stem = self.stems[index]
        payload = torch.load(
            self.condition_root / f"{stem}.pt", map_location="cpu", weights_only=True
        )
        sample = decode_condition(
            payload,
            stem,
            self.config,
            self.model_config,
            self.lengths[index],
            condition_path=self.condition_root / f"{stem}.pt",
            condition_manifest=self.condition_manifest,
        )
        # The shared batch allocator uses this tensor; sampling never consumes it.
        sample["motion"] = torch.zeros(sample["frames"], self.model_config.motion_dim)
        return sample


class RobotSpeechDataset(Dataset[dict[str, Any]]):
    def __init__(
        self,
        root: Path,
        config: DataConfig,
        model_config: ModelConfig,
        split: str,
        mean: torch.Tensor,
        std: torch.Tensor,
        max_items: int = 0,
        longest_first: bool = False,
    ) -> None:
        if split not in {"train", "val"}:
            raise ValueError(f"unsupported split: {split}")
        self.root = root
        self.config = config
        self.model_config = model_config
        self.mean = mean.float()
        self.std = std.float().clamp_min(1e-6)
        self.condition_root = root / config.condition_dir
        self.condition_manifest = load_condition_manifest(
            config.condition_manifest, config.condition_manifest_sha256, root
        )
        self.motion_root = root / config.motion_dir
        relative_split = config.train_split if split == "train" else config.val_split
        self.split_path = root / relative_split
        stems = read_stems(self.split_path)

        condition_stems = available_stems(self.condition_root)
        motion_stems = available_stems(self.motion_root)
        missing_condition = [stem for stem in stems if stem not in condition_stems]
        missing_motion = [stem for stem in stems if stem not in motion_stems]
        if missing_condition or missing_motion:
            raise FileNotFoundError(
                "unpaired dataset entries: "
                f"conditions={missing_condition[:5]}, motions={missing_motion[:5]}"
            )

        audited_lengths: dict[str, int] | None = None
        if config.aligned_lengths_file is not None:
            audited_lengths = load_aligned_lengths(root / config.aligned_lengths_file)
            missing_lengths = [stem for stem in stems if stem not in audited_lengths]
            if missing_lengths:
                raise FileNotFoundError(
                    f"{len(missing_lengths)} split entries lack audited lengths; "
                    f"first={missing_lengths[:5]}"
                )
        self.audited_lengths = audited_lengths

        if longest_first:
            if audited_lengths is None:
                raise ValueError("longest_first requires data.aligned_lengths_file")
            stems.sort(key=audited_lengths.__getitem__, reverse=True)
        if max_items > 0:
            stems = stems[:max_items]
        self.stems = stems
        if audited_lengths is None:
            self.lengths = [self._inspect_length(stem) for stem in stems]
        else:
            self.lengths = [min(audited_lengths[stem], config.max_frames) for stem in stems]

    def _inspect_length(self, stem: str) -> int:
        condition = torch.load(
            self.condition_root / f"{stem}.pt", map_location="cpu", weights_only=True
        )
        motion = torch.load(self.motion_root / f"{stem}.pt", map_location="cpu", weights_only=True)
        audio = _audio_features(condition)
        robot = torch.as_tensor(motion["robot_repr"])
        real_frames = min(int(motion.get("real_num_frames", robot.shape[0])), robot.shape[0])
        return min(real_frames, audio.shape[0], self.config.max_frames)

    def __len__(self) -> int:
        return len(self.stems)

    def __getitem__(self, index: int) -> dict[str, Any]:
        stem = self.stems[index]
        condition = torch.load(
            self.condition_root / f"{stem}.pt", map_location="cpu", weights_only=True
        )
        motion_payload = torch.load(
            self.motion_root / f"{stem}.pt", map_location="cpu", weights_only=True
        )
        motion = torch.as_tensor(motion_payload["robot_repr"]).float()
        audio = _audio_features(condition)
        frames = self.lengths[index]
        real_frames = min(
            int(motion_payload.get("real_num_frames", motion.shape[0])), motion.shape[0]
        )
        if frames < 2:
            raise ValueError(f"{stem}: aligned sequence has fewer than two frames")
        if real_frames < frames or audio.shape[0] < frames:
            raise ValueError(
                f"{stem}: aligned length {frames} exceeds motion/audio "
                f"lengths {real_frames}/{audio.shape[0]}"
            )
        motion = motion[:frames]
        audio = audio[:frames]
        if motion.shape[-1] != self.model_config.motion_dim:
            raise ValueError(f"{stem}: invalid motion shape {tuple(motion.shape)}")
        if audio.shape[-1] != self.model_config.audio_dim:
            raise ValueError(f"{stem}: invalid audio shape {tuple(audio.shape)}")

        normalized_motion = (motion - self.mean) / self.std
        if not torch.isfinite(normalized_motion).all():
            raise ValueError(f"{stem}: normalized motion contains non-finite values")
        sample = decode_condition(
            condition,
            stem,
            self.config,
            self.model_config,
            frames,
            condition_path=self.condition_root / f"{stem}.pt",
            condition_manifest=self.condition_manifest,
        )
        sample["motion"] = normalized_motion
        return sample


class LengthBucketBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        lengths: list[int],
        batch_size: int,
        seed: int,
        bucket_multiplier: int,
        shuffle: bool,
        drop_last: bool = True,
    ) -> None:
        if batch_size <= 0 or bucket_multiplier <= 0:
            raise ValueError("batch_size and bucket_multiplier must be positive")
        self.lengths = lengths
        self.batch_size = batch_size
        self.seed = seed
        self.bucket_size = batch_size * bucket_multiplier
        self.shuffle = shuffle
        self.drop_last = drop_last
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        if self.drop_last:
            return len(self.lengths) // self.batch_size
        return math.ceil(len(self.lengths) / self.batch_size)

    def _batches(self, indices: list[int]) -> Iterator[list[int]]:
        for start in range(0, len(indices), self.batch_size):
            batch = indices[start : start + self.batch_size]
            if len(batch) == self.batch_size or not self.drop_last:
                yield batch

    def __iter__(self) -> Iterator[list[int]]:
        ordered = sorted(range(len(self.lengths)), key=self.lengths.__getitem__)
        if not self.shuffle:
            yield from self._batches(ordered)
            return
        generator = random.Random(self.seed + self.epoch)
        batches: list[list[int]] = []
        for start in range(0, len(ordered), self.bucket_size):
            bucket = ordered[start : start + self.bucket_size]
            generator.shuffle(bucket)
            batches.extend(self._batches(bucket))
        generator.shuffle(batches)
        yield from batches


def collate_motion(samples: list[dict[str, Any]]) -> dict[str, Any]:
    if not samples:
        raise ValueError("cannot collate an empty batch")
    batch_size = len(samples)
    max_frames = max(int(sample["frames"]) for sample in samples)
    max_tokens = max(sample["text"].shape[0] for sample in samples)
    audio_dim = samples[0]["audio"].shape[-1]
    text_dim = samples[0]["text"].shape[-1]
    motion_dim = samples[0]["motion"].shape[-1]
    audio = torch.zeros(batch_size, max_frames, audio_dim)
    text = torch.zeros(batch_size, max_tokens, text_dim)
    motion = torch.zeros(batch_size, max_frames, motion_dim)
    motion_mask = torch.zeros(batch_size, max_frames, dtype=torch.bool)
    text_padding_mask = torch.ones(batch_size, max_tokens, dtype=torch.bool)
    time_distance = torch.zeros(batch_size, max_frames, max_tokens)
    stems: list[str] = []
    for row, sample in enumerate(samples):
        frames = int(sample["frames"])
        tokens = sample["text"].shape[0]
        audio[row, :frames] = sample["audio"]
        text[row, :tokens] = sample["text"]
        motion[row, :frames] = sample["motion"]
        motion_mask[row, :frames] = True
        text_padding_mask[row, :tokens] = False
        if sample["has_timing"]:
            time_distance[row, :frames, :tokens] = (
                sample["frame_times"][:, None] - sample["token_centers"][None, :]
            )
        stems.append(sample["stem"])
    return {
        "audio": audio,
        "text": text,
        "motion": motion,
        "motion_mask": motion_mask,
        "text_padding_mask": text_padding_mask,
        "time_distance": time_distance,
        "stems": stems,
    }
