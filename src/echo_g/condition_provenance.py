# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any

import torch

from echo_g.utils import sha256

TEXT_PROVENANCE_SCHEMA = "echo-g-complete-text-provenance-v1"
FROZEN_MANIFEST_SCHEMA = "echo-g-frozen-condition-manifest-v1"
TIME_MAPPING = "character_offset_union_with_spaces_at_previous_word_end"


def tensor_sha256(tensor: torch.Tensor) -> str:
    return hashlib.sha256(tensor.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def token_time_spans(
    transcript: str,
    offsets: list[tuple[int, int]] | list[list[int]],
    words: tuple[dict[str, Any], ...] | list[dict[str, Any]],
) -> torch.Tensor:
    if not words:
        raise ValueError("ECHO-G requires word timestamps")
    canonical = " ".join(str(word.get("text", word.get("word", ""))) for word in words)
    if transcript != canonical:
        raise ValueError("Token offsets must refer to the canonical space-joined word transcript")
    character_times: list[tuple[float, float]] = []
    for index, word in enumerate(words):
        text = str(word.get("text", word.get("word", "")))
        start, end = float(word["start"]), float(word["end"])
        if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
            raise ValueError("Invalid word timestamp")
        character_times.extend((start, end) for _ in text)
        if index != len(words) - 1:
            character_times.append((end, end))
    spans: list[list[float]] = []
    for low, high in offsets:
        low, high = int(low), min(int(high), len(character_times))
        if low < 0:
            raise ValueError("Negative tokenizer character offset")
        covered = character_times[low:high]
        if high <= low or not covered:
            spans.append([0.0, 0.0])
        else:
            spans.append([min(value[0] for value in covered), max(value[1] for value in covered)])
    return torch.tensor(spans, dtype=torch.float32)


def make_text_provenance(
    payload: dict[str, Any],
    full_token_ids: list[int],
    full_token_offsets: list[list[int]],
    encoder_identity: dict[str, Any],
) -> dict[str, Any]:
    return {
        "schema": TEXT_PROVENANCE_SCHEMA,
        "full_token_count": len(full_token_ids),
        "tokenizer_truncation": False,
        "token_ids": full_token_ids,
        "token_offsets": full_token_offsets,
        "encoder_identity": encoder_identity,
        "token_time_mapping": TIME_MAPPING,
        "canonical_transcript_sha256": hashlib.sha256(
            payload["canonical_transcript"].encode("utf-8")
        ).hexdigest(),
        "text_tokens_sha256": tensor_sha256(payload["text_tokens"]),
        "token_times_sha256": tensor_sha256(payload["token_times"]),
    }


def validate_text_provenance(payload: dict[str, Any], stem: str) -> None:
    provenance = payload.get("text_provenance")
    if not isinstance(provenance, dict) or provenance.get("schema") != TEXT_PROVENANCE_SCHEMA:
        raise ValueError(
            f"{stem}: missing complete-text provenance; regenerate or use a verified manifest"
        )
    count = provenance.get("full_token_count")
    ids, offsets = provenance.get("token_ids"), provenance.get("token_offsets")
    text, times = payload.get("text_tokens"), payload.get("token_times")
    if (
        type(count) is not int
        or not 1 <= count <= 256
        or provenance.get("tokenizer_truncation") is not False
        or not isinstance(ids, list)
        or len(ids) != count
        or any(type(value) is not int or value < 0 for value in ids)
        or not isinstance(offsets, list)
        or len(offsets) != count
        or not isinstance(text, torch.Tensor)
        or text.ndim != 2
        or len(text) != count
        or type(payload.get("n_tokens")) is not int
        or payload["n_tokens"] != count
        or not isinstance(times, torch.Tensor)
        or times.shape != (count, 2)
    ):
        raise ValueError(f"{stem}: incomplete or truncated text provenance")
    transcript, words = payload.get("canonical_transcript"), payload.get("word_timestamps")
    encoder = provenance.get("encoder_identity")
    if (
        not isinstance(transcript, str)
        or not transcript
        or not isinstance(words, list)
        or not words
        or not isinstance(encoder, dict)
        or not isinstance(encoder.get("model"), str)
        or not encoder["model"]
        or encoder["model"] != payload.get("source_text_model")
        or payload.get("hidden_layer") != -2
        or payload.get("encoder_dtype") != "bfloat16"
        or provenance.get("token_time_mapping") != TIME_MAPPING
    ):
        raise ValueError(f"{stem}: invalid encoder or word-time provenance")
    for offset in offsets:
        if (
            not isinstance(offset, list)
            or len(offset) != 2
            or any(type(value) is not int for value in offset)
            or not 0 <= offset[0] <= offset[1] <= len(transcript)
        ):
            raise ValueError(f"{stem}: invalid complete-token character offsets")
    expected_times = token_time_spans(transcript, offsets, words).to(times.dtype)
    if not torch.equal(times.cpu(), expected_times):
        raise ValueError(f"{stem}: token timestamps do not match the original word mapping")
    checks = {
        "canonical_transcript_sha256": hashlib.sha256(transcript.encode("utf-8")).hexdigest(),
        "text_tokens_sha256": tensor_sha256(text),
        "token_times_sha256": tensor_sha256(times),
    }
    if any(provenance.get(key) != value for key, value in checks.items()):
        raise ValueError(f"{stem}: complete-text provenance digest mismatch")


class FrozenConditionManifest:
    def __init__(self, path: Path, expected_sha256: str) -> None:
        if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
            raise ValueError("A pinned condition manifest SHA256 is required")
        contents = path.read_bytes()
        if hashlib.sha256(contents).hexdigest() != expected_sha256:
            raise ValueError("Condition manifest SHA256 differs from the pinned release identity")
        payload = json.loads(contents)
        self.entries = payload.get("entries")
        if (
            payload.get("schema") != FROZEN_MANIFEST_SCHEMA
            or not isinstance(self.entries, dict)
            or not self.entries
            or payload.get("entry_count") != len(self.entries)
        ):
            raise ValueError("Invalid frozen condition manifest")
        self.sha256 = expected_sha256

    def verify(self, path: Path, stem: str, payload: dict[str, Any]) -> None:
        record = self.entries.get(stem)
        if not isinstance(record, dict):
            raise ValueError(f"{stem}: no audited complete-text condition identity")
        count = record.get("full_token_count")
        if (
            type(count) is not int
            or not 1 <= count <= 256
            or count != payload.get("n_tokens")
            or len(payload["text_tokens"]) != count
            or sha256(path) != record.get("condition_sha256")
            or tensor_sha256(payload["text_tokens"]) != record.get("text_tokens_sha256")
            or tensor_sha256(payload["token_times"]) != record.get("token_times_sha256")
        ):
            raise ValueError(
                f"{stem}: frozen condition differs from its audited complete-text identity"
            )


def load_condition_manifest(
    path: str | Path | None, expected_sha256: str | None, root: Path | None = None
) -> FrozenConditionManifest | None:
    if path is None and expected_sha256 is None:
        return None
    if path is None or expected_sha256 is None:
        raise ValueError(
            "condition_manifest and condition_manifest_sha256 must be supplied together"
        )
    resolved = Path(path)
    if root is not None and not resolved.is_absolute():
        resolved = root / resolved
    return FrozenConditionManifest(resolved, expected_sha256)


def validate_condition_provenance(
    payload: dict[str, Any],
    stem: str,
    condition_path: Path | None = None,
    manifest: FrozenConditionManifest | None = None,
) -> None:
    if manifest is not None:
        if condition_path is None:
            raise ValueError(f"{stem}: a condition path is required for manifest verification")
        manifest.verify(condition_path, stem, payload)
        if "text_provenance" in payload:
            validate_text_provenance(payload, stem)
    elif "text_provenance" in payload:
        validate_text_provenance(payload, stem)
    else:
        raise ValueError(
            f"{stem}: missing complete-text provenance; regenerate conditions or provide "
            "the audited condition manifest and its published SHA256"
        )
