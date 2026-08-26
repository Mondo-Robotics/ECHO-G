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
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from echo_g.utils import atomic_torch_save

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
    parser.add_argument("--max-text-tokens", type=int, default=64)
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
            if not SAFE_STEM.fullmatch(stem):
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
            if not transcript and words:
                transcript = " ".join(str(word.get("text", word.get("word", ""))) for word in words)
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


def token_time_spans(
    transcript: str,
    offsets: list[tuple[int, int]] | list[list[int]],
    words: tuple[dict[str, Any], ...],
) -> torch.Tensor:
    if not words:
        return torch.zeros(len(offsets), 2, dtype=torch.float32)
    character_times: list[tuple[float, float] | None] = [None] * len(transcript)
    cursor = 0
    for word in words:
        text = str(word.get("text", word.get("word", "")))
        if not text:
            continue
        start_time = float(word["start"])
        end_time = float(word["end"])
        if end_time < start_time:
            raise ValueError(f"word has negative duration: {word}")
        character_start = word.get("char_start")
        character_end = word.get("char_end")
        if character_start is None or character_end is None:
            character_start = transcript.find(text, cursor)
            if character_start < 0:
                raise ValueError(f"word {text!r} cannot be aligned to transcript")
            character_end = character_start + len(text)
        character_start = int(character_start)
        character_end = int(character_end)
        if not 0 <= character_start < character_end <= len(transcript):
            raise ValueError(f"invalid character span for word {word}")
        for index in range(character_start, character_end):
            character_times[index] = (start_time, end_time)
        cursor = character_end

    spans: list[list[float]] = []
    for low, high in offsets:
        low = int(low)
        high = min(int(high), len(character_times))
        covered = [value for value in character_times[low:high] if value is not None]
        if high <= low or not covered:
            spans.append([0.0, 0.0])
        else:
            spans.append(
                [min(value[0] for value in covered), max(value[1] for value in covered)]
            )
    return torch.tensor(spans, dtype=torch.float32)


def interpolate_features(features: torch.Tensor, output_frames: int) -> torch.Tensor:
    if features.ndim != 3 or features.shape[0] != 1:
        raise ValueError(f"expected [1, frames, channels], got {features.shape}")
    output_frames = max(2, output_frames)
    values = features.transpose(1, 2)
    values = F.interpolate(values, size=output_frames, mode="linear", align_corners=True)
    return values.transpose(1, 2)[0]


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
        import soundfile as sound_file
        from scipy.signal import resample_poly
        from transformers import AutoModelForCTC, AutoProcessor
    except ImportError as error:
        raise RuntimeError("install ECHO-G with the preprocess extra") from error
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir.mkdir(parents=True, exist_ok=True)
    processor = AutoProcessor.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCTC.from_pretrained(
        model_name, trust_remote_code=trust_remote_code
    ).to(device)
    model.eval()
    for index, utterance in enumerate(utterances, start=1):
        output_path = output_dir / f"{utterance.stem}.pt"
        if skip_existing and output_path.is_file():
            continue
        waveform, sample_rate = sound_file.read(
            utterance.audio_path, dtype="float32", always_2d=True
        )
        waveform = waveform.mean(axis=1)
        if sample_rate != 16_000:
            waveform = resample_poly(waveform, 16_000, sample_rate)
        duration = len(waveform) / 16_000.0
        inputs = processor(waveform, sampling_rate=16_000, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            result = model(input_values, output_hidden_states=True)
        hidden = result.hidden_states[-1].float()
        features = interpolate_features(hidden, int(round(duration * fps))).cpu()
        atomic_torch_save(
            {
                "audio_features": features.half(),
                "duration": duration,
                "fps": fps,
                "stem": utterance.stem,
                "audio_dim": features.shape[-1],
            },
            output_path,
        )
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
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        raise RuntimeError("install ECHO-G with the preprocess extra") from error
    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, trust_remote_code=trust_remote_code, use_fast=True
    )
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=dtype,
        trust_remote_code=trust_remote_code,
    ).to(device)
    model.eval()
    for index, utterance in enumerate(utterances, start=1):
        output_path = output_dir / f"{utterance.stem}.pt"
        if skip_existing and output_path.is_file():
            continue
        encoded = tokenizer(
            utterance.transcript,
            return_tensors="pt",
            return_offsets_mapping=True,
            truncation=True,
            max_length=512,
        )
        offsets = encoded.pop("offset_mapping")[0].tolist()
        inputs = {key: value.to(device) for key, value in encoded.items()}
        with torch.no_grad():
            result = model(**inputs, output_hidden_states=True, use_cache=False)
        hidden = result.hidden_states[layer][0].float()
        token_features = hidden[:max_tokens].cpu().contiguous()
        times = token_time_spans(utterance.transcript, offsets, utterance.words)
        times = times[: token_features.shape[0]]
        atomic_torch_save(
            {
                "text_pooled": hidden.mean(dim=0).cpu().half(),
                "text_tokens": token_features.half(),
                "token_times": times.half(),
                "has_word_timing": bool(utterance.words),
                "stem": utterance.stem,
                "text_dim": token_features.shape[-1],
                "n_tokens": token_features.shape[0],
            },
            output_path,
        )
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
        if skip_existing and output_path.is_file():
            continue
        audio_path = audio_dir / f"{utterance.stem}.pt"
        text_path = text_dir / f"{utterance.stem}.pt"
        if not audio_path.is_file() or not text_path.is_file():
            raise FileNotFoundError(f"missing condition component for {utterance.stem}")
        audio = torch.load(audio_path, map_location="cpu", weights_only=True)
        text = torch.load(text_path, map_location="cpu", weights_only=True)
        atomic_torch_save({**audio, **text, "stem": utterance.stem}, output_path)
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
        merge_conditions(
            utterances, audio_dir, text_dir, args.output_dir, args.skip_existing
        )


if __name__ == "__main__":
    main()
