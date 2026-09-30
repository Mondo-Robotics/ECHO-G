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

import pytest
import torch

from echo_g.conditions import load_manifest
from echo_g.config import ExperimentConfig, HumanInferenceConfig, load_inference_config
from echo_g.data import collate_motion, decode_condition
from echo_g.human_retarget.decoder import align_root_rz90, retarget_arbitrary_length
from echo_g.model import SpeechGroundedDiT


def test_audio_only_needs_no_text_or_word_times(
    synthetic_release: tuple[Path, ExperimentConfig],
) -> None:
    root, config = synthetic_release
    payload = torch.load(root / "condition_30fps/val_00.pt", weights_only=True)
    payload = {"audio_features": payload["audio_features"], "fps": 30}
    data = replace(config.data, conditioning="audio-only")
    model = replace(config.model, architecture="audio_only")
    sample = decode_condition(payload, "audio", data, model)
    assert torch.equal(sample["audio"], payload["audio_features"].float())
    assert sample["text"].shape == (1, model.text_dim)
    assert not sample["text"].any()
    sample["motion"] = torch.zeros(sample["frames"], model.motion_dim)
    batch = collate_motion([sample])
    assert not batch["time_distance"].any()
    network = SpeechGroundedDiT(model)
    state = network.state_dict()
    assert "blocks.0.time_bias_strength" in state
    assert "blocks.0.qk_temperature_raw" not in state


def test_text_only_uses_explicit_duration_without_audio(
    synthetic_release: tuple[Path, ExperimentConfig],
) -> None:
    root, config = synthetic_release
    payload = torch.load(root / "condition_30fps/val_00.pt", weights_only=True)
    payload.pop("audio_features")
    payload["duration"] = 0.5
    sample = decode_condition(
        payload, "text", replace(config.data, conditioning="text-only"), config.model
    )
    assert sample["frames"] == 15
    assert not sample["audio"].any()
    assert torch.equal(sample["text"], payload["text_tokens"].float())
    payload["duration"] = 0.05
    with pytest.raises(ValueError, match="timestamps exceed"):
        decode_condition(
            payload, "text", replace(config.data, conditioning="text-only"), config.model
        )
    payload.pop("duration")
    with pytest.raises(ValueError, match="duration"):
        decode_condition(
            payload, "text", replace(config.data, conditioning="text-only"), config.model
        )


def test_modality_specific_raw_manifest(tmp_path: Path) -> None:
    wav = tmp_path / "sample.wav"
    wav.touch()
    manifest = tmp_path / "manifest.jsonl"
    manifest.write_text(json.dumps({"stem": "audio", "audio_path": "sample.wav"}) + "\n")
    assert load_manifest(manifest, mode="audio")[0].words == ()
    with pytest.raises(ValueError, match="word timestamps"):
        load_manifest(manifest)
    manifest.write_text(
        json.dumps(
            {
                "stem": "text",
                "duration": 1.0,
                "words": [{"text": "Hello", "start": 0.1, "end": 0.5}],
            }
        )
        + "\n"
    )
    assert load_manifest(manifest, mode="text")[0].audio_path is None


def test_human_config_cannot_be_loaded_as_training_recipe() -> None:
    path = Path(__file__).parents[1] / "configs/human_retarget_inference.yaml"
    assert isinstance(load_inference_config(path), HumanInferenceConfig)
    with pytest.raises(ValueError, match="missing"):
        ExperimentConfig.from_yaml(path)


@pytest.mark.parametrize("frames", [2, 61, 72, 99, 100, 119, 120, 121, 247, 599, 600])
def test_retarget_windowing_preserves_tail_and_order(frames: int) -> None:
    class IdentityRetarget(torch.nn.Module):
        def retarget(self, values: torch.Tensor) -> torch.Tensor:
            return values[..., :39]

    values = torch.arange(frames, dtype=torch.float32)[:, None].expand(-1, 136)
    output = retarget_arbitrary_length(IdentityRetarget(), values)
    assert output.shape == (frames, 39)
    torch.testing.assert_close(output, values[:, :39])


def test_root_alignment_changes_orientation_only() -> None:
    values = torch.arange(78, dtype=torch.float32).reshape(2, 39)
    values[:, :6] = torch.tensor([1, 0, 0, 0, 1, 0])
    output = align_root_rz90(values)
    expected = torch.tensor([0, -1, 0, 1, 0, 0], dtype=torch.float32).expand(2, -1)
    torch.testing.assert_close(output[:, :6], expected)
    assert torch.equal(output[:, 6:], values[:, 6:])
