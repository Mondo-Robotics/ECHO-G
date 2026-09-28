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
import torch.nn.functional as F

V2_SCHEMA = "release30-bounded-qk-global-local-wordtime-v2.1"
V2_CONFIG = {
    "temperature_min": 1.0,
    "temperature_max": 16.0,
    "temperature_init": 8.0,
    "sigma_min_seconds": 0.25,
    "sigma_max_seconds": 2.0,
    "sigma_init_seconds": 0.75,
    "max_abs_lag_seconds": 0.5,
    "max_mix": 0.5,
    "init_mix": 0.1,
    "max_log_penalty": 12.0,
    "normalize_eps": 1e-6,
}


def initial_logit(value: float, lower: float, upper: float) -> float:
    proportion = (value - lower) / (upper - lower)
    if not 0 < proportion < 1:
        raise ValueError("initial value must lie strictly inside bounds")
    return math.log(proportion / (1 - proportion))


def bounded(raw: torch.Tensor, lower: float, upper: float) -> torch.Tensor:
    return lower + (upper - lower) * raw.float().sigmoid()


def block_parameters(block: Any) -> dict[str, torch.Tensor]:
    return {
        "temperature": bounded(
            block.qk_temperature_raw, V2_CONFIG["temperature_min"], V2_CONFIG["temperature_max"]
        ),
        "sigma": bounded(
            block.word_sigma_raw, V2_CONFIG["sigma_min_seconds"], V2_CONFIG["sigma_max_seconds"]
        ),
        "lag": V2_CONFIG["max_abs_lag_seconds"] * block.word_lag_raw.float().tanh(),
        "mix": V2_CONFIG["max_mix"] * block.word_mix_raw.float().sigmoid(),
    }


def masked_softmax(logits: torch.Tensor, padding: torch.Tensor) -> torch.Tensor:
    mask = padding[:, None, None, :]
    has_keys = (~mask).any(dim=-1, keepdim=True)
    safe = torch.where(has_keys, logits.masked_fill(mask, -torch.inf), 0.0)
    return safe.softmax(dim=-1).masked_fill(mask, 0.0)


def attention_weights(
    query: torch.Tensor,
    key: torch.Tensor,
    padding: torch.Tensor,
    signed_distance: torch.Tensor,
    temperature: torch.Tensor,
    sigma: torch.Tensor,
    lag: torch.Tensor,
    mix: torch.Tensor,
    diagnostics: bool = False,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if query.ndim != 4 or key.ndim != 4 or key.shape[2] == 0:
        raise ValueError("Q/K must be [B,H,T/N,D] with at least one key")
    if signed_distance.shape != (query.shape[0], query.shape[2], key.shape[2]):
        raise ValueError("signed time distances must be [B,T,N]")
    q = F.normalize(query.float(), dim=-1, eps=V2_CONFIG["normalize_eps"])
    k = F.normalize(key.float(), dim=-1, eps=V2_CONFIG["normalize_eps"])
    logits = (q @ k.transpose(-1, -2)) * temperature[None, :, None, None]
    global_weights = masked_softmax(logits, padding)
    standardized = (signed_distance[:, None].float() - lag[None, :, None, None]) / sigma[
        None, :, None, None
    ]
    log_prior = -0.5 * standardized.square()
    local_weights = masked_softmax(
        logits + log_prior.clamp_min(-V2_CONFIG["max_log_penalty"]), padding
    )
    affinity = log_prior.exp().masked_fill(padding[:, None, None, :], 0.0)
    support = affinity.amax(dim=-1, keepdim=True)
    effective_mix = mix[None, :, None, None] * support
    weights = (1 - effective_mix) * global_weights + effective_mix * local_weights
    information = (
        {
            "logits": logits,
            "global": global_weights,
            "local": local_weights,
            "effective_mix": effective_mix,
            "support": support,
        }
        if diagnostics
        else {}
    )
    return weights, information
