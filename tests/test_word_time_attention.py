# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

from dataclasses import asdict

import pytest
import torch
import torch.nn.functional as F

from echo_g.config import ModelConfig
from echo_g.model import SpeechGroundedDiT, WordTimeCrossAttentionBlock
from echo_g.word_time_attention import WORD_TIME_CONFIG, attention_weights, block_parameters


def _small_model() -> SpeechGroundedDiT:
    return SpeechGroundedDiT(
        ModelConfig(
            motion_dim=5,
            audio_dim=8,
            text_dim=12,
            hidden_dim=32,
            num_layers=2,
            num_heads=4,
            feedforward_dim=64,
            max_t=32,
        )
    ).eval()


def _reference_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    padding: torch.Tensor,
    distance: torch.Tensor,
    parameters: dict[str, torch.Tensor],
) -> torch.Tensor:
    # Independent scalar-head reconstruction: softmax as stabilized exp/sum.
    result = torch.zeros(query.shape[:3] + (key.shape[2],), dtype=torch.float32)
    for row in range(query.shape[0]):
        valid = ~padding[row]
        if not valid.any():
            continue
        for head in range(query.shape[1]):
            q = query[row, head].float()
            k = key[row, head, valid].float()
            q = q / q.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            k = k / k.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            logits = (q @ k.T) * parameters["temperature"][head]
            global_exp = (logits - logits.max(dim=-1, keepdim=True).values).exp()
            global_probability = global_exp / global_exp.sum(dim=-1, keepdim=True)
            delta = distance[row, :, valid] - parameters["lag"][head]
            prior = -(delta / parameters["sigma"][head]).square() / 2
            local_logits = logits + prior.clamp_min(-12)
            local_exp = (local_logits - local_logits.max(dim=-1, keepdim=True).values).exp()
            local_probability = local_exp / local_exp.sum(dim=-1, keepdim=True)
            mixture = parameters["mix"][head] * prior.exp().max(dim=-1, keepdim=True).values
            result[row, head, :, valid] = (
                global_probability * (1 - mixture) + local_probability * mixture
            )
    return result


def test_attention_matches_independent_reference_and_cross_output() -> None:
    torch.manual_seed(13)
    block = WordTimeCrossAttentionBlock(16, 4, 32, 0.0).eval()
    with torch.no_grad():
        block.word_lag_raw.copy_(torch.tensor([-0.8, -0.2, 0.4, 1.1]))
    motion, text = torch.randn(2, 7, 16), torch.randn(2, 5, 16)
    padding = torch.tensor([[False, False, False, True, True], [False] * 5])
    distance = torch.linspace(-2, 2, 70).reshape(2, 7, 5)
    query = block.query(motion).reshape(2, 7, 4, 4).transpose(1, 2)
    key = block.key(text).reshape(2, 5, 4, 4).transpose(1, 2)
    value = block.value(text).reshape(2, 5, 4, 4).transpose(1, 2)
    parameters = block_parameters(block)
    weights, info = attention_weights(query, key, padding, distance, **parameters, diagnostics=True)
    reference = _reference_attention(query, key, padding, distance, parameters)
    torch.testing.assert_close(weights, reference, atol=2e-7, rtol=1e-6)
    expected = F.linear(
        (reference @ value.float()).transpose(1, 2).reshape(2, 7, 16),
        block.cross_output.weight,
        block.cross_output.bias,
    )
    actual = block.cross_attention(motion, text, padding, distance)
    torch.testing.assert_close(actual, expected, atol=2e-7, rtol=1e-6)
    assert actual.abs().sum() > 0
    assert info["effective_mix"].max() <= 0.5
    assert torch.all(weights >= 0.5 * info["global"] - 1e-7)
    actual.square().sum().backward()
    for name in ("qk_temperature_raw", "word_sigma_raw", "word_lag_raw", "word_mix_raw"):
        gradient = getattr(block, name).grad
        assert gradient is not None and torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_signed_lag_direction_and_all_padding_safety() -> None:
    query, key = torch.zeros(1, 1, 1, 2), torch.zeros(1, 1, 2, 2)
    distance = torch.tensor([[[-0.25, 0.25]]])
    parameters = dict(
        temperature=torch.tensor([8.0]),
        sigma=torch.tensor([0.25]),
        lag=torch.tensor([0.25]),
        mix=torch.tensor([0.5]),
    )
    padding = torch.zeros(1, 2, dtype=torch.bool)
    weights, _ = attention_weights(query, key, padding, distance, **parameters)
    assert weights[0, 0, 0, 1] > weights[0, 0, 0, 0]
    absolute, _ = attention_weights(query, key, padding, distance.abs(), **parameters)
    torch.testing.assert_close(absolute, torch.full_like(absolute, 0.5))
    masked, _ = attention_weights(query, key, ~padding, distance, **parameters)
    assert torch.isfinite(masked).all() and torch.count_nonzero(masked) == 0


def test_original_checkpoint_names_and_strict_roundtrip() -> None:
    model = _small_model()
    original_config = dict(
        motion_dim=5,
        audio_dim=8,
        text_dim=12,
        hidden=32,
        layers=2,
        heads=4,
        ff_dim=64,
        max_t=32,
        pos_enc="learned",
        dropout=0.0,
    )
    restored = SpeechGroundedDiT.from_checkpoint_config(original_config)
    restored.load_state_dict(model.state_dict(), strict=True)
    assert restored.config == model.config
    assert model.state_dict()["position"].shape == (1, 32, 32)
    assert "blocks.0.word_lag_raw" in model.state_dict()
    assert not any("time_bias_strength" in name for name in model.state_dict())
    config_restored = SpeechGroundedDiT.from_checkpoint_config(asdict(model.config))
    config_restored.load_state_dict(model.state_dict(), strict=True)
    legacy_state = dict(model.state_dict())
    legacy_state.pop("blocks.0.word_mix_raw")
    legacy_state["blocks.0.time_bias_strength"] = torch.tensor(0.5)
    with pytest.raises(RuntimeError):
        restored.load_state_dict(legacy_state, strict=True)
    with pytest.raises(ValueError, match="pos_enc"):
        SpeechGroundedDiT.from_checkpoint_config({**original_config, "pos_enc": "sinusoidal"})
    with pytest.raises(ValueError, match="architecture"):
        SpeechGroundedDiT.from_checkpoint_config({"hidden_dim": 32})


def test_nonzero_model_output_uses_time_and_respects_capacity() -> None:
    torch.manual_seed(37)
    model = _small_model()
    # Open zero-initialized gates/output so this test exercises cross-attention.
    with torch.no_grad():
        model.output.weight.normal_(std=0.2)
        for block in model.blocks:
            block.adaptive_norm[-1].bias[5 * model.hidden : 6 * model.hidden].fill_(1.0)
            block.word_lag_raw.fill_(0.7)
    arguments = [
        torch.randn(1, 7, 5),
        torch.rand(1),
        torch.randn(1, 7, 8),
        torch.randn(1, 3, 12),
        torch.ones(1, 7, dtype=torch.bool),
        torch.zeros(1, 3, dtype=torch.bool),
        torch.arange(7)[None, :, None] / 30 - torch.tensor([0.0, 0.5, 1.0]),
    ]
    output = model(*arguments)
    reversed_time = model(*arguments[:-1], -arguments[-1])
    assert output.abs().sum() > 0
    assert (output - reversed_time).abs().max() > 1e-5
    with pytest.raises(ValueError, match="capacity"):
        model.positional_encoding(33, torch.device("cpu"), torch.float32)
    arguments[3] = torch.zeros(1, 257, 12)
    with pytest.raises(ValueError, match="token"):
        model(*arguments)


def test_v2_default_contract() -> None:
    config = ModelConfig()
    config.validate()
    assert config.max_t == 608 and config.max_text_tokens == 256
    assert config.position_encoding == "learned" and config.architecture == "v2"
    assert WORD_TIME_CONFIG["max_abs_lag_seconds"] == 0.5
