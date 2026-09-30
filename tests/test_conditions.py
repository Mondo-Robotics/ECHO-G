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
from pathlib import Path
from types import ModuleType, SimpleNamespace
from typing import Any

import pytest
import torch

from echo_g.conditions import (
    Utterance,
    extract_audio_conditions,
    extract_text_conditions,
    load_manifest,
    merge_conditions,
    token_time_spans,
)
from echo_g.config import DataConfig, ModelConfig
from echo_g.data import decode_condition


def _utterance(tmp_path: Path) -> Utterance:
    path = tmp_path / "audio.wav"
    path.write_bytes(b"synthetic audio source identity")
    return Utterance(
        "example",
        path,
        "hi world",
        (
            {"text": "hi", "start": 0.0, "end": 0.2},
            {"text": "world", "start": 0.3, "end": 0.8},
        ),
    )


def _fake_encoders(
    monkeypatch: pytest.MonkeyPatch, count: int = 3, text_dimension: int = 2560
) -> dict[str, Any]:
    records: dict[str, Any] = {}

    class Tokenizer:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Tokenizer:
            return cls()

        def __call__(self, transcript: str, **kwargs: Any) -> dict[str, torch.Tensor]:
            records["tokenizer_kwargs"] = kwargs
            offsets = [(0, 2), (2, 3), (3, 8)] if count == 3 else [(0, 1)] * count
            return {
                "input_ids": torch.ones(1, count, dtype=torch.long),
                "offset_mapping": torch.tensor([offsets]),
            }

    class TextModel:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> TextModel:
            records["text_model_kwargs"] = kwargs
            return cls()

        def to(self, *args: Any) -> TextModel:
            return self

        def eval(self) -> TextModel:
            return self

        def __call__(self, **kwargs: Any) -> SimpleNamespace:
            chosen = (
                torch.linspace(-0.3, 0.7, count * text_dimension)
                .reshape(1, count, text_dimension)
                .to(torch.bfloat16)
            )
            records["selected_hidden"] = chosen
            wrong = torch.zeros_like(chosen)
            return SimpleNamespace(hidden_states=[wrong, chosen, wrong])

    class AudioModel(TextModel):
        def __call__(self, *args: Any, **kwargs: Any) -> SimpleNamespace:
            features = torch.arange(3).float()[None, :, None].expand(1, 3, 1024)
            return SimpleNamespace(hidden_states=[features])

    class Processor:
        @classmethod
        def from_pretrained(cls, *args: Any, **kwargs: Any) -> Processor:
            return cls()

        def __call__(self, waveform: Any, **kwargs: Any) -> SimpleNamespace:
            return SimpleNamespace(input_values=torch.zeros(1, 16000))

    transformers = ModuleType("transformers")
    transformers.AutoTokenizer = Tokenizer
    transformers.Qwen3_5ForConditionalGeneration = TextModel
    transformers.Wav2Vec2Processor = Processor
    transformers.Wav2Vec2ForCTC = AudioModel
    librosa = ModuleType("librosa")
    librosa.load = lambda *args, **kwargs: (torch.zeros(16000).numpy(), 16000)
    monkeypatch.setitem(sys.modules, "transformers", transformers)
    monkeypatch.setitem(sys.modules, "librosa", librosa)
    return records


def test_canonical_join_and_subword_space_punctuation_timing(tmp_path: Path) -> None:
    utterance = _utterance(tmp_path)
    manifest = tmp_path / "input.jsonl"
    manifest.write_text(
        json.dumps(
            {
                "stem": "example",
                "audio_path": "audio.wav",
                "transcript": "hi    world",
                "words": utterance.words,
            }
        )
        + "\n"
    )
    loaded = load_manifest(manifest)[0]
    assert loaded.transcript == "hi world"
    words = ({"text": "你", "start": 0.1, "end": 0.2}, {"text": "好!", "start": 0.5, "end": 0.8})
    times = token_time_spans("你 好!", [(0, 1), (1, 2), (1, 4), (3, 4), (0, 0)], words)
    torch.testing.assert_close(
        times,
        torch.tensor(
            [
                [0.1, 0.2],
                [0.2, 0.2],
                [0.2, 0.8],
                [0.5, 0.8],
                [0.0, 0.0],
            ]
        ),
    )
    with pytest.raises(ValueError, match="canonical"):
        token_time_spans("你好!", [(0, 1)], words)


def test_raw_encoder_contract_and_full_cache_merge(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    utterance = _utterance(tmp_path)
    records = _fake_encoders(monkeypatch)
    audio_dir, text_dir, merged_dir = tmp_path / "audio", tmp_path / "text", tmp_path / "merged"
    extract_audio_conditions([utterance], audio_dir, "audio-test", 30, "cpu", False, False)
    extract_text_conditions([utterance], text_dir, "text-test", -2, 256, "cpu", False, False)
    merge_conditions([utterance], audio_dir, text_dir, merged_dir, False)
    payload = torch.load(merged_dir / "example.pt", weights_only=True)
    assert payload["audio_features"].shape == (30, 1024)
    assert payload["audio_features"].dtype == torch.float16
    torch.testing.assert_close(
        payload["audio_features"][[0, -1], 0], torch.tensor([0.0, 2.0]).half()
    )
    assert records["text_model_kwargs"]["torch_dtype"] == torch.bfloat16
    assert records["tokenizer_kwargs"]["truncation"] is False
    assert torch.equal(payload["text_tokens"], records["selected_hidden"][0].float().half())
    assert torch.equal(payload["text_pooled"], records["selected_hidden"][0].float().mean(0).half())
    assert payload["n_tokens"] == 3 and payload["hidden_layer"] == -2
    torch.testing.assert_close(
        payload["token_times"], torch.tensor([[0, 0.2], [0.2, 0.2], [0.3, 0.8]]).half()
    )
    # A timestamp-only change must not reuse an old text component at merge.
    changed = Utterance(
        utterance.stem,
        utterance.audio_path,
        utterance.transcript,
        ({"text": "hi", "start": 0.0, "end": 0.1}, utterance.words[1]),
    )
    with pytest.raises(ValueError, match="timestamps"):
        merge_conditions([changed], audio_dir, text_dir, merged_dir, False)
    utterance.audio_path.write_bytes(b"different source")
    with pytest.raises(ValueError, match="source changed"):
        merge_conditions([utterance], audio_dir, text_dir, merged_dir, False)


@pytest.mark.parametrize(("count", "dimension"), [(257, 2560), (3, 128)])
def test_raw_text_rejects_overflow_and_wrong_encoder(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    count: int,
    dimension: int,
) -> None:
    utterance = _utterance(tmp_path)
    _fake_encoders(monkeypatch, count, dimension)
    with pytest.raises(ValueError):
        extract_text_conditions(
            [utterance], tmp_path / "text", "test", -2, 256, "cpu", False, False
        )
    assert not (tmp_path / "text/example.pt").exists()


@pytest.mark.parametrize(
    "field", ["token_times", "text_tokens", "audio_features", "duration", "fps"]
)
def test_merge_rejects_nonfinite_or_wrong_fps(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    field: str,
) -> None:
    utterance = _utterance(tmp_path)
    _fake_encoders(monkeypatch)
    audio_dir, text_dir = tmp_path / "audio", tmp_path / "text"
    extract_audio_conditions([utterance], audio_dir, "audio-test", 30, "cpu", False, False)
    extract_text_conditions([utterance], text_dir, "text-test", -2, 256, "cpu", False, False)
    target = text_dir if field in ("token_times", "text_tokens") else audio_dir
    path = target / "example.pt"
    payload = torch.load(path, weights_only=True)
    if isinstance(payload[field], torch.Tensor):
        payload[field][0, 0] = float("nan")
    else:
        payload[field] = 25 if field == "fps" else float("nan")
    torch.save(payload, path)
    with pytest.raises(ValueError):
        merge_conditions([utterance], audio_dir, text_dir, tmp_path / "merged", False)


@pytest.mark.parametrize("count", [None, True, 3.5, 100])
def test_cached_text_requires_full_integer_token_count(count: Any) -> None:
    payload = {
        "audio_features": torch.zeros(10, 8),
        "text_tokens": torch.zeros(3, 12),
        "token_times": torch.tensor([[0.0, 0.1], [0.1, 0.2], [0.2, 0.3]]),
        "has_word_timing": True,
        "fps": 30,
    }
    if count is not None:
        payload["n_tokens"] = count
    model = ModelConfig(audio_dim=8, text_dim=12)
    with pytest.raises(ValueError, match="token|truncated"):
        decode_condition(payload, "example", DataConfig(), model)
