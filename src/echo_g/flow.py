# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn

from echo_g.model import SpeechGroundedDiT


def apply_condition_dropout(
    audio: torch.Tensor,
    text: torch.Tensor,
    text_padding_mask: torch.Tensor,
    time_distance: torch.Tensor,
    probability: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if probability <= 0:
        return audio, text, text_padding_mask, time_distance
    dropped = torch.rand(audio.shape[0], device=audio.device) < probability
    if not dropped.any():
        return audio, text, text_padding_mask, time_distance
    audio = audio.masked_fill(dropped[:, None, None], 0.0)
    text = text.masked_fill(dropped[:, None, None], 0.0)
    text_padding_mask = text_padding_mask.clone()
    text_padding_mask[dropped] = True
    text_padding_mask[dropped, 0] = False
    time_distance = time_distance.masked_fill(dropped[:, None, None], 0.0)
    return audio, text, text_padding_mask, time_distance


def flow_matching_loss(
    model: nn.Module,
    batch: dict[str, Any],
    condition_dropout: float,
    temporal_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    target = batch["motion"]
    audio, text, text_padding_mask, time_distance = apply_condition_dropout(
        batch["audio"],
        batch["text"],
        batch["text_padding_mask"],
        batch["time_distance"],
        condition_dropout,
    )
    noise = torch.randn_like(target)
    flow_time = torch.rand(target.shape[0], device=target.device)
    flow_time_expanded = flow_time[:, None, None]
    noised_motion = flow_time_expanded * target + (1.0 - flow_time_expanded) * noise
    target_velocity = target - noise
    predicted_velocity = model(
        noised_motion,
        flow_time,
        audio,
        text,
        batch["motion_mask"],
        text_padding_mask,
        time_distance,
    )

    valid = batch["motion_mask"].unsqueeze(-1).to(predicted_velocity.dtype)
    denominator = valid.sum().clamp_min(1.0) * target.shape[-1]
    flow_loss = ((predicted_velocity - target_velocity).square() * valid).sum()
    flow_loss = flow_loss / denominator

    adjacent = batch["motion_mask"][:, 1:] & batch["motion_mask"][:, :-1]
    adjacent = adjacent.unsqueeze(-1).to(predicted_velocity.dtype)
    predicted_delta = predicted_velocity[:, 1:] - predicted_velocity[:, :-1]
    target_delta = target_velocity[:, 1:] - target_velocity[:, :-1]
    temporal_denominator = adjacent.sum().clamp_min(1.0) * target.shape[-1]
    temporal_loss = ((predicted_delta - target_delta).square() * adjacent).sum()
    temporal_loss = temporal_loss / temporal_denominator
    total = flow_loss + temporal_weight * temporal_loss
    return total, flow_loss, temporal_loss


@torch.no_grad()
def sample_euler(
    model: SpeechGroundedDiT,
    batch: dict[str, Any],
    steps: int,
    guidance_scale: float,
    initial_noise: torch.Tensor | None = None,
) -> torch.Tensor:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if guidance_scale < 0:
        raise ValueError("guidance_scale cannot be negative")
    audio = batch["audio"]
    text = batch["text"]
    motion_mask = batch["motion_mask"]
    text_padding_mask = batch["text_padding_mask"]
    time_distance = batch["time_distance"]
    expected_shape = (audio.shape[0], audio.shape[1], model.motion_dim)
    sample = initial_noise
    if sample is None:
        sample = torch.randn(expected_shape, device=audio.device, dtype=audio.dtype)
    if sample.shape != expected_shape:
        raise ValueError(f"initial_noise has shape {sample.shape}, expected {expected_shape}")

    step_size = 1.0 / steps
    for index in range(steps):
        flow_time = torch.full(
            (audio.shape[0],), index * step_size, device=audio.device, dtype=torch.float32
        )
        conditional = model(
            sample,
            flow_time,
            audio,
            text,
            motion_mask,
            text_padding_mask,
            time_distance,
        )
        if guidance_scale == 1.0:
            velocity = conditional
        else:
            unconditional_padding = torch.ones_like(text_padding_mask)
            unconditional_padding[:, 0] = False
            unconditional = model(
                sample,
                flow_time,
                torch.zeros_like(audio),
                torch.zeros_like(text),
                motion_mask,
                unconditional_padding,
                torch.zeros_like(time_distance),
            )
            velocity = unconditional + guidance_scale * (conditional - unconditional)
        sample = sample + step_size * velocity
    return sample
