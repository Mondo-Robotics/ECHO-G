# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest
import torch

from echo_g.condition_provenance import (
    FROZEN_MANIFEST_SCHEMA,
    FrozenConditionManifest,
    make_text_provenance,
    tensor_sha256,
    token_time_spans,
)
from echo_g.config import DataConfig, ExperimentConfig, ModelConfig
from echo_g.data import ConditionDataset, RobotSpeechDataset, decode_condition
from echo_g.utils import sha256


def _complete_payload(count: int = 64) -> dict[str, Any]:
    words = [
        {"text": "a", "start": index / 100, "end": (index + 1) / 100} for index in range(count)
    ]
    transcript = " ".join(word["text"] for word in words)
    offsets = [[index * 2, index * 2 + 1] for index in range(count)]
    payload = {
        "audio_features": torch.zeros(30, 8),
        "text_tokens": torch.arange(count * 12).reshape(count, 12).float(),
        "token_times": token_time_spans(transcript, offsets, words).half(),
        "has_word_timing": True,
        "n_tokens": count,
        "canonical_transcript": transcript,
        "word_timestamps": words,
        "source_text_model": "test-encoder",
        "hidden_layer": -2,
        "encoder_dtype": "bfloat16",
    }
    payload["text_provenance"] = make_text_provenance(
        payload, list(range(count)), offsets, {"model": "test-encoder"}
    )
    return payload


def _manifest(path: Path, conditions: Path, stems: list[str]) -> str:
    entries = {}
    for stem in stems:
        condition_path = conditions / f"{stem}.pt"
        payload = torch.load(condition_path, weights_only=True)
        entries[stem] = {
            "condition_sha256": sha256(condition_path),
            "full_token_count": payload["n_tokens"],
            "text_tokens_sha256": tensor_sha256(payload["text_tokens"]),
            "token_times_sha256": tensor_sha256(payload["token_times"]),
        }
    path.write_text(
        json.dumps(
            {"schema": FROZEN_MANIFEST_SCHEMA, "entry_count": len(entries), "entries": entries}
        )
    )
    return sha256(path)


def test_old_main_64_token_cache_rejected_by_actual_dataset(tmp_path: Path) -> None:
    payload = _complete_payload(90)
    del payload["text_provenance"]
    payload["text_tokens"] = payload["text_tokens"][:64]
    payload["token_times"] = payload["token_times"][:64]
    payload["n_tokens"] = 64  # Exactly the old main extractor's post-truncation count.
    torch.save(payload, tmp_path / "example.pt")
    with pytest.raises(ValueError, match="missing complete-text provenance"):
        ConditionDataset(tmp_path, ["example"], DataConfig(), ModelConfig(audio_dim=8, text_dim=12))


def test_complete_64_tokens_valid_without_blanket_length_rejection(tmp_path: Path) -> None:
    payload = _complete_payload(64)
    torch.save(payload, tmp_path / "example.pt")
    dataset = ConditionDataset(
        tmp_path, ["example"], DataConfig(), ModelConfig(audio_dim=8, text_dim=12)
    )
    assert torch.equal(dataset[0]["text"], payload["text_tokens"])
    assert torch.equal(dataset[0]["token_centers"], payload["token_times"].float().mean(-1))


@pytest.mark.parametrize(
    "defect", ["clipped", "changed_word_time", "changed_features", "offsets", "encoder"]
)
def test_modern_cache_requires_independent_complete_provenance(defect: str) -> None:
    payload = _complete_payload(90)
    if defect == "clipped":
        payload["text_tokens"] = payload["text_tokens"][:64]
        payload["token_times"] = payload["token_times"][:64]
        payload["n_tokens"] = 64
    elif defect == "changed_word_time":
        payload["word_timestamps"][0]["end"] += 0.1
    elif defect == "changed_features":
        payload["text_tokens"][0, 0] += 1
    elif defect == "offsets":
        payload["text_provenance"]["token_offsets"][0] = [-1, 0]
    else:
        payload["source_text_model"] = "different-encoder"
    with pytest.raises(ValueError):
        decode_condition(payload, "example", DataConfig(), ModelConfig(audio_dim=8, text_dim=12))


def test_verified_frozen_cache_works_in_both_real_dataset_entrypoints(
    synthetic_release: tuple[Path, ExperimentConfig], tmp_path: Path
) -> None:
    root, config = synthetic_release
    condition_root = root / config.data.condition_dir
    stems = ["val_00", "val_01"]
    reference = []
    for stem in stems:
        path = condition_root / f"{stem}.pt"
        payload = torch.load(path, weights_only=True)
        reference.append(decode_condition(payload, stem, config.data, config.model))
        del payload["text_provenance"]
        torch.save(payload, path)
    manifest_path = tmp_path / "frozen.json"
    manifest_hash = _manifest(manifest_path, condition_root, stems)
    verified_config = replace(
        config.data, condition_manifest=str(manifest_path), condition_manifest_sha256=manifest_hash
    )
    paired = RobotSpeechDataset(
        root, verified_config, config.model, "val", torch.zeros(5), torch.ones(5)
    )
    independent = ConditionDataset(
        condition_root,
        stems,
        config.data,
        config.model,
        condition_manifest=manifest_path,
        condition_manifest_sha256=manifest_hash,
    )
    for index in range(2):
        for dataset in (paired, independent):
            actual = dataset[index]
            for key in ("audio", "text", "token_centers", "frame_times"):
                assert torch.equal(actual[key], reference[index][key])
    path = condition_root / "val_00.pt"
    payload = torch.load(path, weights_only=True)
    payload["text_tokens"][0, 0] += 1
    torch.save(payload, path)
    for dataset in (paired, independent):
        with pytest.raises(ValueError, match="audited complete-text identity"):
            dataset[0]


def test_explicit_manifest_cannot_be_bypassed_by_modern_provenance(tmp_path: Path) -> None:
    payload = _complete_payload()
    path = tmp_path / "example.pt"
    torch.save(payload, path)
    manifest = tmp_path / "frozen.json"
    expected = _manifest(manifest, tmp_path, ["example"])
    payload["audio_features"] += 0.1
    torch.save(payload, path)
    with pytest.raises(ValueError, match="audited complete-text identity"):
        ConditionDataset(
            tmp_path,
            ["example"],
            DataConfig(),
            ModelConfig(audio_dim=8, text_dim=12),
            condition_manifest=manifest,
            condition_manifest_sha256=expected,
        )


def test_manifest_itself_requires_published_digest(tmp_path: Path) -> None:
    path = tmp_path / "frozen.json"
    path.write_text("{}")
    with pytest.raises(ValueError, match="pinned release identity"):
        FrozenConditionManifest(path, "0" * 64)
    with pytest.raises(ValueError, match="pinned condition manifest SHA256"):
        FrozenConditionManifest(path, "")
