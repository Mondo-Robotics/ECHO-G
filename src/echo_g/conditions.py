# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import argparse
import json
import logging
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from echo_g.condition_provenance import (
    make_text_provenance,
    token_time_spans,
    validate_text_provenance,
)
from echo_g.utils import atomic_torch_save, sha256

LOGGER = logging.getLogger(__name__)
SAFE_STEM = re.compile(r"^[A-Za-z0-9._-]+$")


@dataclass(frozen=True)
class Utterance:
    stem: str
    audio_path: Path
    transcript: str
    words: tuple[dict[str, Any], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract wav2vec and language-model conditions for ECHO-G"
    )
    parser.add_argument("--mode", required=True, choices=["audio", "text", "merge"])
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--audio-model")
    parser.add_argument("--text-model")
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--text-layer", type=int, default=-2)
    parser.add_argument("--max-text-tokens", type=int, default=256)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def load_manifest(path: Path, limit: int = 0) -> list[Utterance]:
    utterances: list[Utterance] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            payload = json.loads(line)
            stem = str(payload.get("stem", ""))
            if not SAFE_STEM.fullmatch(stem) or stem.startswith("."):
                raise ValueError(f"{path}:{line_number}: unsafe or empty stem {stem!r}")
            if stem in seen:
                raise ValueError(f"{path}:{line_number}: duplicate stem {stem}")
            audio_path = Path(str(payload.get("audio_path", "")))
            if not audio_path.is_absolute():
                audio_path = (path.parent / audio_path).resolve()
            transcript = str(payload.get("transcript", ""))
            words_payload = payload.get("words") or []
            if not isinstance(words_payload, list):
                raise ValueError(f"{path}:{line_number}: words must be a list")
            words = tuple(dict(word) for word in words_payload)
            if not words:
                raise ValueError(f"{path}:{line_number}: ECHO-G requires word timestamps")
            previous_start = -1.0
            for word in words:
                text = str(word.get("text", word.get("word", "")))
                start, end = float(word["start"]), float(word["end"])
                if (
                    not text
                    or not math.isfinite(start)
                    or not math.isfinite(end)
                    or start < previous_start
                    or start < 0
                    or end < start
                ):
                    raise ValueError(f"{path}:{line_number}: invalid word timestamps")
                word.update(text=text, start=start, end=end)
                previous_start = start
            # Match the canonical transcript used by the released encoders.
            transcript = " ".join(word["text"] for word in words)
            if not transcript:
                raise ValueError(f"{path}:{line_number}: transcript is empty")
            if not audio_path.is_file():
                raise FileNotFoundError(audio_path)
            seen.add(stem)
            utterances.append(Utterance(stem, audio_path, transcript, words))
            if limit > 0 and len(utterances) >= limit:
                break
    if not utterances:
        raise ValueError(f"{path}: no utterances")
    return utterances


def interpolate_features(features: torch.Tensor, output_frames: int) -> torch.Tensor:
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError(f"expected [1, frames, channels], got {features.shape}")
    output_frames = max(2, output_frames)
    values = features.transpose(1, 2)
    values = F.interpolate(values, size=output_frames, mode="linear", align_corners=True)
    return values.transpose(1, 2)[0]


def validate_audio_component(payload: dict[str, Any], stem: str) -> None:
    features = payload.get("audio_features")
    duration, fps = (
        float(payload.get("duration", float("nan"))),
        float(payload.get("fps", float("nan"))),
    )
    if not math.isfinite(duration) or not 0 < duration <= 20 or fps != 30:
        raise ValueError(f"{stem}: invalid audio duration or frame rate")
    frames = max(2, round(duration * fps))
    if (
        payload.get("stem") != stem
        or not isinstance(features, torch.Tensor)
        or features.shape != (frames, 1024)
        or not torch.isfinite(features).all()
    ):
        raise ValueError(f"{stem}: expected finite audio [{frames},1024]")


def validate_text_component(payload: dict[str, Any], stem: str) -> None:
    count, features, times = (
        payload.get("n_tokens"),
        payload.get("text_tokens"),
        payload.get("token_times"),
    )
    if (
        not isinstance(count, int)
        or isinstance(count, bool)
        or not 1 <= count <= 256
        or payload.get("stem") != stem
        or not payload.get("has_word_timing")
        or not isinstance(features, torch.Tensor)
        or features.shape != (count, 2560)
        or not torch.isfinite(features).all()
        or not isinstance(times, torch.Tensor)
        or times.shape != (count, 2)
        or not torch.isfinite(times).all()
        or bool((times < 0).any())
        or bool((times[:, 1] < times[:, 0]).any())
    ):
        raise ValueError(f"{stem}: invalid or truncated text/timestamp features")
    validate_text_provenance(payload, stem)
    pooled = payload.get("text_pooled")
    if pooled is not None and (
        not isinstance(pooled, torch.Tensor)
        or pooled.shape != (2560,)
        or not torch.isfinite(pooled).all()
    ):
        raise ValueError(f"{stem}: invalid pooled text")


def word_timing_records(utterance: Utterance) -> list[dict[str, Any]]:
    return [
        {
            "text": str(word.get("text", word.get("word", ""))),
            "start": float(word["start"]),
            "end": float(word["end"]),
        }
        for word in utterance.words
    ]


def validate_word_duration(utterance: Utterance, duration: float) -> None:
    ends = [float(word["end"]) for word in utterance.words]
    if not ends or not all(math.isfinite(end) for end in ends) or max(ends) > duration + 0.05:
        raise ValueError(f"{utterance.stem}: word timestamps exceed audio duration")


def extract_audio_conditions(
    utterances: list[Utterance],
    output_dir: Path,
    model_name: str,
    fps: float,
    device_name: str,
    trust_remote_code: bool,
    skip_existing: bool,
) -> None:
    try:
        import librosa
        from transformers import Wav2Vec2ForCTC, Wav2Vec2Processor
    except ImportError as error:
        raise RuntimeError("install ECHO-G with the preprocess extra") from error
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = Wav2Vec2Processor.from_pretrained(model_name, trust_remote_code=trust_remote_code)
    model = Wav2Vec2ForCTC.from_pretrained(model_name, trust_remote_code=trust_remote_code).to(
        device
    )
    model.eval()
    for index, utterance in enumerate(utterances, start=1):
        output_path = output_dir / f"{utterance.stem}.pt"
        source_hash = sha256(utterance.audio_path)
        if skip_existing and output_path.is_file():
            cached = torch.load(output_path, map_location="cpu", weights_only=True)
            if (
                cached.get("source_audio_sha256") != source_hash
                or cached.get("source_audio_model") != model_name
                or cached.get("fps") != fps
            ):
                raise ValueError(f"Stale cached audio: {output_path}")
            validate_audio_component(cached, utterance.stem)
            validate_word_duration(utterance, float(cached["duration"]))
            continue
        waveform, _ = librosa.load(utterance.audio_path, sr=16_000, mono=True)
        duration = len(waveform) / 16_000.0
        if not 0 < duration <= 20.0 or fps != 30:
            raise ValueError("Input audio must be at most 20 seconds, with target fps=30")
        validate_word_duration(utterance, duration)
        inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            result = model(input_values, output_hidden_states=True)
        hidden = result.hidden_states[-1].float()
        if (
            hidden.ndim != 3
            or hidden.shape[0] != 1
            or hidden.shape[1] < 1
            or hidden.shape[2] != 1024
            or not torch.isfinite(hidden).all()
        ):
            raise ValueError("Wav2Vec2 must return finite hidden features [1,T,1024]")
        features = interpolate_features(hidden, int(round(duration * fps))).cpu()
        payload = {
            "audio_features": features.half(),
            "source_audio_sha256": source_hash,
            "source_audio_model": model_name,
            "duration": duration,
            "fps": fps,
            "stem": utterance.stem,
            "audio_dim": features.shape[-1],
        }
        validate_audio_component(payload, utterance.stem)
        atomic_torch_save(payload, output_path)
        if index <= 5 or index % 25 == 0:
            LOGGER.info("audio %d/%d %s %s", index, len(utterances), utterance.stem, features.shape)


def extract_text_conditions(
    utterances: list[Utterance],
    output_dir: Path,
    model_name: str,
    layer: int,
    max_tokens: int,
    device_name: str,
    trust_remote_code: bool,
    skip_existing: bool,
) -> None:
    try:
        from transformers import AutoTokenizer, Qwen3_5ForConditionalGeneration
    except ImportError as error:
        raise RuntimeError("install ECHO-G with the preprocess extra") from error
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, use_fast=True
    )
    if not 1 <= max_tokens <= 256 or layer != -2:
        raise ValueError("Text extraction uses hidden layer -2 and at most 256 tokens")
    dtype = torch.bfloat16
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    for index, utterance in enumerate(utterances, start=1):
        output_path = output_dir / f"{utterance.stem}.pt"
        encoded = tokenizer(
            utterance.transcript,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=False,
        )
        count = int(encoded["input_ids"].shape[1])
        if not 1 <= count <= max_tokens:
            raise ValueError(
                f"{utterance.stem}: {count} tokens exceed capacity {max_tokens}; split explicitly"
            )
        full_token_ids = encoded["input_ids"][0].tolist()
        offsets = encoded.pop("offset_mapping")[0].tolist()
        times = token_time_spans(utterance.transcript, offsets, utterance.words)
        if skip_existing and output_path.is_file():
            cached = torch.load(output_path, map_location="cpu", weights_only=True)
            if (
                cached.get("canonical_transcript") != utterance.transcript
                or cached.get("source_text_model") != model_name
                or cached.get("n_tokens") != count
                or cached.get("hidden_layer") != -2
                or cached.get("encoder_dtype") != "bfloat16"
                or cached.get("word_timestamps") != word_timing_records(utterance)
                or not torch.equal(cached["token_times"], times.half())
            ):
                raise ValueError(f"Stale cached text: {output_path}")
            validate_text_component(cached, utterance.stem)
            continue
        input_ids = encoded["input_ids"].to(device)
        with torch.no_grad():
            result = model(
                input_ids=input_ids,
                attention_mask=torch.ones_like(input_ids),
                output_hidden_states=True,
            )
        hidden = result.hidden_states[layer][0].float()
        if hidden.shape != (count, 2560) or not torch.isfinite(hidden).all():
            raise ValueError(f"Qwen must return finite hidden features [{count},2560]")
        token_features = hidden.cpu().contiguous()
        payload = {
            "text_pooled": hidden.mean(dim=0).cpu().half(),
            "canonical_transcript": utterance.transcript,
            "source_text_model": model_name,
            "hidden_layer": -2,
            "encoder_dtype": "bfloat16",
            "word_timestamps": word_timing_records(utterance),
            "token_time_mapping": "character_offset_union_with_spaces_at_previous_word_end",
            "text_tokens": token_features.half(),
            "token_times": times.half(),
            "has_word_timing": bool(utterance.words),
            "stem": utterance.stem,
            "text_dim": token_features.shape[-1],
            "n_tokens": token_features.shape[0],
        }
        payload["text_provenance"] = make_text_provenance(
            payload,
            full_token_ids,
            offsets,
            {
                "model": model_name,
                "model_class": type(model).__name__,
                "tokenizer_class": type(tokenizer).__name__,
                "resolved_revision": getattr(getattr(model, "config", None), "_commit_hash", None),
            },
        )
        validate_text_component(payload, utterance.stem)
        atomic_torch_save(payload, output_path)
        if index <= 5 or index % 25 == 0:
            LOGGER.info(
                "text %d/%d %s %s",
                index,
                len(utterances),
                utterance.stem,
                token_features.shape,
            )


def merge_conditions(
    utterances: list[Utterance],
    audio_dir: Path,
    text_dir: Path,
    output_dir: Path,
    skip_existing: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, utterance in enumerate(utterances, start=1):
        output_path = output_dir / f"{utterance.stem}.pt"
        audio_path = audio_dir / f"{utterance.stem}.pt"
        text_path = text_dir / f"{utterance.stem}.pt"
        if not audio_path.is_file() or not text_path.is_file():
            raise FileNotFoundError(f"missing condition component for {utterance.stem}")
        audio = torch.load(audio_path, map_location="cpu", weights_only=True)
        text = torch.load(text_path, map_location="cpu", weights_only=True)
        validate_audio_component(audio, utterance.stem)
        validate_text_component(text, utterance.stem)
        if audio.get("source_audio_sha256") != sha256(utterance.audio_path):
            raise ValueError(f"{utterance.stem}: cached audio source changed")
        if (
            text.get("canonical_transcript") != utterance.transcript
            or text.get("word_timestamps") != word_timing_records(utterance)
            or text.get("hidden_layer") != -2
            or text.get("encoder_dtype") != "bfloat16"
        ):
            raise ValueError(f"{utterance.stem}: cached text or word timestamps changed")
        validate_word_duration(utterance, float(audio["duration"]))
        if float(text["token_times"].max()) > float(audio["duration"]) + 0.05:
            raise ValueError("Token timestamps exceed audio duration")
        merged = {**audio, **text, "stem": utterance.stem}
        if skip_existing and output_path.is_file():
            cached = torch.load(output_path, map_location="cpu", weights_only=True)
            for key, value in merged.items():
                equal = (
                    isinstance(cached.get(key), torch.Tensor) and torch.equal(cached[key], value)
                    if isinstance(value, torch.Tensor)
                    else cached.get(key) == value
                )
                if not equal:
                    raise ValueError(f"Stale merged condition field {key}: {output_path}")
        else:
            atomic_torch_save(merged, output_path)
        if index <= 5 or index % 100 == 0:
            LOGGER.info("merged %d/%d %s", index, len(utterances), utterance.stem)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    utterances = load_manifest(args.manifest, args.limit)
    audio_dir = args.output_dir / ".audio"
    text_dir = args.output_dir / ".text"
    if args.mode == "audio":
        if not args.audio_model:
            raise ValueError("--audio-model is required for --mode audio")
        extract_audio_conditions(
            utterances,
            audio_dir,
            args.audio_model,
            args.fps,
            args.device,
            args.trust_remote_code,
            args.skip_existing,
        )
    elif args.mode == "text":
        if not args.text_model:
            raise ValueError("--text-model is required for --mode text")
        extract_text_conditions(
            utterances,
            text_dir,
            args.text_model,
            args.text_layer,
            args.max_text_tokens,
            args.device,
            args.trust_remote_code,
            args.skip_existing,
        )
    else:
        merge_conditions(utterances, audio_dir, text_dir, args.output_dir, args.skip_existing)


if __name__ == "__main__":
    main()
