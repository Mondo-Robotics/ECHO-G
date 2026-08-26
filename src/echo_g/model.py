# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from echo_g.config import ModelConfig


def timestep_embedding(timestep: torch.Tensor, dimension: int) -> torch.Tensor:
    half = dimension // 2
    frequencies = torch.exp(
        -math.log(10_000.0)
        * torch.arange(half, device=timestep.device, dtype=torch.float32)
        / max(half, 1)
    )
    angles = timestep.float()[:, None] * frequencies[None] * 1_000.0
    embedding = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


def sinusoidal_position_encoding(
    frames: int,
    dimension: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if frames <= 0:
        raise ValueError("frames must be positive")
    positions = torch.arange(frames, device=device, dtype=torch.float32)[:, None]
    even_indices = torch.arange(0, dimension, 2, device=device, dtype=torch.float32)
    divisor = torch.exp(even_indices * (-math.log(10_000.0) / dimension))
    angles = positions * divisor
    encoding = torch.zeros(frames, dimension, device=device, dtype=torch.float32)
    encoding[:, 0::2] = torch.sin(angles)
    encoding[:, 1::2] = torch.cos(angles[:, : encoding[:, 1::2].shape[1]])
    return encoding.unsqueeze(0).to(dtype=dtype)


def modulate(values: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return values * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


class TimedCrossAttentionBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, feedforward: int, dropout: float) -> None:
        super().__init__()
        if hidden % heads:
            raise ValueError("hidden dimension must be divisible by attention heads")
        self.heads = heads
        self.head_dim = hidden // heads
        self.self_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.self_attention = nn.MultiheadAttention(
            hidden, heads, dropout=dropout, batch_first=True
        )
        self.cross_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.query = nn.Linear(hidden, hidden)
        self.key = nn.Linear(hidden, hidden)
        self.value = nn.Linear(hidden, hidden)
        self.cross_output = nn.Linear(hidden, hidden)
        self.time_bias_strength = nn.Parameter(torch.tensor(0.5))
        self.mlp_norm = nn.LayerNorm(hidden, elementwise_affine=False, eps=1e-6)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, feedforward),
            nn.GELU(),
            nn.Linear(feedforward, hidden),
        )
        self.adaptive_norm = nn.Sequential(nn.SiLU(), nn.Linear(hidden, 9 * hidden))
        nn.init.zeros_(self.adaptive_norm[-1].weight)
        nn.init.zeros_(self.adaptive_norm[-1].bias)

    def cross_attention(
        self,
        motion: torch.Tensor,
        text: torch.Tensor,
        text_padding_mask: torch.Tensor,
        time_distance: torch.Tensor,
    ) -> torch.Tensor:
        batch, frames, hidden = motion.shape
        tokens = text.shape[1]
        query = self.query(motion).view(batch, frames, self.heads, self.head_dim)
        query = query.transpose(1, 2)
        key = self.key(text).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        value = self.value(text).view(batch, tokens, self.heads, self.head_dim).transpose(1, 2)
        logits = torch.matmul(query.float(), key.float().transpose(-1, -2))
        logits = logits / math.sqrt(self.head_dim)
        time_bias = -F.softplus(self.time_bias_strength.float()) * time_distance.float()
        logits = logits + time_bias.unsqueeze(1)
        logits = logits.masked_fill(text_padding_mask[:, None, None, :], float("-inf"))
        attention = torch.softmax(logits, dim=-1)
        attended = torch.matmul(attention, value.float()).to(motion.dtype)
        attended = attended.transpose(1, 2).reshape(batch, frames, hidden)
        return self.cross_output(attended)

    def forward(
        self,
        values: torch.Tensor,
        condition: torch.Tensor,
        text: torch.Tensor,
        motion_padding_mask: torch.Tensor,
        text_padding_mask: torch.Tensor,
        time_distance: torch.Tensor,
    ) -> torch.Tensor:
        parameters = self.adaptive_norm(condition).chunk(9, dim=-1)
        self_shift, self_scale, self_gate = parameters[:3]
        cross_shift, cross_scale, cross_gate = parameters[3:6]
        mlp_shift, mlp_scale, mlp_gate = parameters[6:]

        hidden = modulate(self.self_norm(values), self_shift, self_scale)
        attended, _ = self.self_attention(
            hidden,
            hidden,
            hidden,
            key_padding_mask=motion_padding_mask,
            need_weights=False,
        )
        values = values + self_gate.unsqueeze(1) * attended
        hidden = modulate(self.cross_norm(values), cross_shift, cross_scale)
        crossed = self.cross_attention(hidden, text, text_padding_mask, time_distance)
        values = values + cross_gate.unsqueeze(1) * crossed
        hidden = modulate(self.mlp_norm(values), mlp_shift, mlp_scale)
        return values + mlp_gate.unsqueeze(1) * self.mlp(hidden)


class SpeechGroundedDiT(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.motion_dim = config.motion_dim
        self.hidden = config.hidden_dim
        self.motion_projection = nn.Linear(config.motion_dim, config.hidden_dim)
        self.audio_norm = nn.LayerNorm(config.audio_dim)
        self.audio_projection = nn.Linear(config.audio_dim, config.hidden_dim)
        self.text_norm = nn.LayerNorm(config.text_dim)
        self.text_projection = nn.Linear(config.text_dim, config.hidden_dim)
        self.timestep_mlp = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.blocks = nn.ModuleList(
            [
                TimedCrossAttentionBlock(
                    config.hidden_dim,
                    config.num_heads,
                    config.feedforward_dim,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(
            config.hidden_dim, elementwise_affine=False, eps=1e-6
        )
        self.final_adaptive_norm = nn.Sequential(
            nn.SiLU(), nn.Linear(config.hidden_dim, 2 * config.hidden_dim)
        )
        nn.init.zeros_(self.final_adaptive_norm[-1].weight)
        nn.init.zeros_(self.final_adaptive_norm[-1].bias)
        self.output = nn.Linear(config.hidden_dim, config.motion_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)

    @classmethod
    def from_checkpoint_config(cls, payload: dict[str, Any]) -> SpeechGroundedDiT:
        if "hidden" in payload:
            position_encoding = payload.get("pos_enc", "sinusoidal")
            config = ModelConfig(
                motion_dim=int(payload["motion_dim"]),
                audio_dim=int(payload.get("audio_dim", 1024)),
                text_dim=int(payload.get("text_dim", 2560)),
                hidden_dim=int(payload["hidden"]),
                num_layers=int(payload["layers"]),
                num_heads=int(payload["heads"]),
                feedforward_dim=int(payload["ff_dim"]),
                dropout=float(payload.get("dropout", 0.0)),
                position_encoding=str(position_encoding),
            )
        else:
            config = ModelConfig(**payload)
        return cls(config)

    def positional_encoding(
        self, frames: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        return sinusoidal_position_encoding(frames, self.hidden, device, dtype)

    def forward(
        self,
        noised_motion: torch.Tensor,
        flow_time: torch.Tensor,
        audio: torch.Tensor,
        text_tokens: torch.Tensor,
        motion_mask: torch.Tensor,
        text_padding_mask: torch.Tensor,
        time_distance: torch.Tensor,
    ) -> torch.Tensor:
        frames = noised_motion.shape[1]
        hidden = (
            self.motion_projection(noised_motion)
            + self.audio_projection(self.audio_norm(audio))
            + self.positional_encoding(frames, noised_motion.device, noised_motion.dtype)
        )
        condition = self.timestep_mlp(timestep_embedding(flow_time, self.hidden))
        text = self.text_projection(self.text_norm(text_tokens))
        motion_padding_mask = ~motion_mask
        for block in self.blocks:
            hidden = block(
                hidden,
                condition,
                text,
                motion_padding_mask,
                text_padding_mask,
                time_distance,
            )
        shift, scale = self.final_adaptive_norm(condition).chunk(2, dim=-1)
        hidden = modulate(self.final_norm(hidden), shift, scale)
        return self.output(hidden)
