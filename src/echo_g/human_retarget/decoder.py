# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from echo_g.human_retarget.networks import (
    MotionVAE,
    MotionVAEConfig,
    RetargetVAE,
    RetargetVAEConfig,
)

LOGGER = logging.getLogger(__name__)
SCHEMA = "echo-g-human-retarget-decoder-v1"
RETARGET_CHECKPOINT_SHA256 = "6b8a95c11c3e301921e56c4520811a6efade9950f42f7a416aa36f95ccc931c8"
HUMAN_VAE_CHECKPOINT_SHA256 = "928a159381564489482d072a203b91918c656cc28047b273c691c74132e4fb62"
ALIGNMENT = "left_multiply_base_orientation_rz_positive_90_degrees"
ALIGNMENT_SCOPE = "base_orientation_6d_only_physical"
RETARGET_CONFIG = {
    "latent_dim": 256,
    "hidden_dim": 512,
    "num_layers": 4,
    "num_heads": 8,
    "ff_dim": 1024,
    "dropout": 0.1,
    "num_frames": 120,
    "fps": 30.0,
    "chunk_len": 20,
    "robot_motion_dim": 39,
    "activation": "gelu",
    "split_latent": True,
    "split_decoder_hard": True,
    "latent_dim_upper": 192,
    "latent_dim_lower": 64,
    "use_latent_adapter": False,
    "adapter_hidden_dim": 512,
}
HUMAN_VAE_CONFIG = {
    "motion_dim": 136,
    "latent_dim": 256,
    "hidden_dim": 512,
    "num_layers": 4,
    "num_heads": 8,
    "ff_dim": 1024,
    "dropout": 0.1,
    "num_frames": 480,
    "fps": 30.0,
    "activation": "gelu",
    "use_chunk_latent": True,
    "chunk_len": 20,
    "use_joint_mixer": False,
    "split_latent": True,
    "latent_dim_upper": 192,
    "latent_dim_lower": 64,
    "split_decoder_hard": True,
}


def frozen_metadata() -> dict[str, Any]:
    return {
        "human_retarget_decoder_schema": SCHEMA,
        "retarget_checkpoint_sha256": RETARGET_CHECKPOINT_SHA256,
        "human_vae_checkpoint_sha256": HUMAN_VAE_CHECKPOINT_SHA256,
        "retarget_window": 120,
        "retarget_chunk_size": 20,
        "retarget_overlap": 20,
        "robot_root_orientation_alignment": ALIGNMENT,
        "robot_root_orientation_alignment_degrees": 90.0,
        "robot_root_orientation_alignment_scope": ALIGNMENT_SCOPE,
        "human_robot_alignment_applied": True,
    }


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _checkpoint(path: Path, expected_sha256: str, config: dict[str, Any]) -> dict[str, Any]:
    if not path.is_file() or sha256(path) != expected_sha256:
        raise ValueError(f"frozen decoder checkpoint SHA256 mismatch: {path.name}")
    payload = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("config"), dict)
        or payload["config"].get("model") != config
    ):
        raise ValueError("frozen decoder architecture configuration mismatch")
    state = payload.get("model_state_dict")
    if not isinstance(state, dict) or not state:
        raise ValueError("missing decoder model_state_dict")
    for name, value in state.items():
        if (
            not isinstance(value, torch.Tensor)
            or not bool(torch.isfinite(value).all())
            or (value.is_floating_point() and value.dtype != torch.float32)
        ):
            raise ValueError(f"decoder weights must be finite FP32 tensors: {name}")
    if sha256(path) != expected_sha256:
        raise ValueError("decoder checkpoint changed while loading")
    return payload


def _statistics(payload: dict[str, Any], device: torch.device) -> dict[str, torch.Tensor]:
    stats = {}
    for name, dimension in (
        ("human_mean", 136),
        ("human_std", 136),
        ("robot_mean", 39),
        ("robot_std", 39),
    ):
        value = payload.get(name)
        if (
            not isinstance(value, torch.Tensor)
            or value.shape != (dimension,)
            or value.dtype != torch.float32
            or not bool(torch.isfinite(value).all())
            or (name.endswith("_std") and not bool((value > 0).all()))
        ):
            raise ValueError(f"invalid frozen retarget normalization: {name}")
        stats[name] = value.float().to(device)
    stats["human_std"] = stats["human_std"].clamp_min(1e-2)
    return stats


@torch.no_grad()
def retarget_arbitrary_length(
    model: torch.nn.Module,
    normalized_human: torch.Tensor,
    window: int = 120,
    overlap: int = 20,
    chunk_size: int = 20,
) -> torch.Tensor:
    # Same operation order as the frozen exporter, including its final window.
    frames = normalized_human.shape[0]
    if (
        normalized_human.ndim != 2
        or normalized_human.shape[1] != 136
        or frames < 1
        or not bool(torch.isfinite(normalized_human).all())
    ):
        raise ValueError("retarget input must be finite nonempty [T,136]")
    if window <= 0 or chunk_size <= 0:
        raise ValueError("retarget window and chunk size must be positive")
    if frames <= window:
        padding = (-frames) % chunk_size
        padded = normalized_human
        if padding:
            padded = torch.cat([padded, padded[-1:].expand(padding, -1)], dim=0)
        return model.retarget(padded.unsqueeze(0))[0, :frames]
    if overlap <= 0 or overlap >= window or (window - overlap) % chunk_size:
        raise ValueError("retarget overlap must give a positive chunk-aligned stride")
    starts = list(range(0, frames - window + 1, window - overlap))
    final_start = frames - window
    if starts[-1] != final_start:
        starts.append(final_start)
    output: torch.Tensor | None = None
    weights: torch.Tensor | None = None
    for start in starts:
        prediction = model.retarget(normalized_human[start : start + window].unsqueeze(0))[0]
        if output is None or weights is None:
            output = torch.zeros(
                frames, prediction.shape[-1], device=prediction.device, dtype=prediction.dtype
            )
            weights = torch.zeros(frames, 1, device=prediction.device, dtype=prediction.dtype)
        blend = torch.ones(window, 1, device=prediction.device, dtype=prediction.dtype)
        if start > 0:
            blend[:overlap] = torch.linspace(
                0, 1, overlap, device=prediction.device, dtype=prediction.dtype
            )[:, None]
        if start + window < frames:
            blend[-overlap:] = torch.linspace(
                1, 0, overlap, device=prediction.device, dtype=prediction.dtype
            )[:, None]
        output[start : start + window] += prediction * blend
        weights[start : start + window] += blend
    if output is None or weights is None:
        raise RuntimeError("retarget window coverage is empty")
    return output / weights.clamp_min(1e-6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


@torch.no_grad()
def align_root_rz90(physical_robot: torch.Tensor) -> torch.Tensor:
    if (
        physical_robot.ndim != 2
        or physical_robot.shape[1] != 39
        or len(physical_robot) < 1
        or physical_robot.dtype != torch.float32
        or not bool(torch.isfinite(physical_robot).all())
    ):
        raise ValueError("orientation correction requires finite FP32 physical [T,39]")
    rotation = rotation_6d_to_matrix(physical_robot[:, :6])
    rz90 = torch.tensor(
        [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=physical_robot.dtype,
        device=physical_robot.device,
    )
    corrected = physical_robot.clone()
    # The frozen rotation implementation stores the first two ROWS in 6D.
    corrected[:, :6] = (rz90 @ rotation)[..., :2, :].clone().reshape(-1, 6)
    return corrected.contiguous()


class HumanRetargetDecoder:
    def __init__(
        self, retarget_checkpoint: Path, human_vae_checkpoint: Path, device: torch.device
    ) -> None:
        self.device = torch.device(device)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("the frozen decoder supports CPU or CUDA")
        LOGGER.info("Loading frozen Human MotionVAE and RetargetVAE")
        human = _checkpoint(
            Path(human_vae_checkpoint), HUMAN_VAE_CHECKPOINT_SHA256, HUMAN_VAE_CONFIG
        )
        retarget = _checkpoint(
            Path(retarget_checkpoint), RETARGET_CHECKPOINT_SHA256, RETARGET_CONFIG
        )
        human_vae = MotionVAE(MotionVAEConfig(**HUMAN_VAE_CONFIG))
        human_vae.load_state_dict(human["model_state_dict"], strict=True)
        self.model = RetargetVAE(human_vae=human_vae, config=RetargetVAEConfig(**RETARGET_CONFIG))
        # Includes the encoder updated during RetargetVAE training. Loading only
        # robot_decoder would incorrectly retain the original Human VAE encoder.
        self.model.load_state_dict(retarget["model_state_dict"], strict=True)
        # Preserve the original parameter flags: human VAE False, robot decoder
        # True. With no_grad inference, changing the latter still selects a
        # numerically different PyTorch path for short sequences on CUDA.
        self.model.to(device=self.device, dtype=torch.float32).eval()
        self.stats = _statistics(retarget, self.device)

    def metadata(self) -> dict[str, Any]:
        return frozen_metadata()

    @torch.no_grad()
    def decode_raw(self, physical_human: torch.Tensor) -> torch.Tensor:
        if (
            not isinstance(physical_human, torch.Tensor)
            or physical_human.ndim != 2
            or physical_human.shape[1] != 136
            or len(physical_human) < 1
            or not physical_human.is_floating_point()
            or not bool(torch.isfinite(physical_human).all())
        ):
            raise ValueError("decoder requires finite physical human motion [T,136]")
        self.model.eval()
        with torch.autocast(device_type=self.device.type, enabled=False):
            physical = physical_human.to(device=self.device, dtype=torch.float32)
            normalized = (physical - self.stats["human_mean"]) / self.stats["human_std"]
            robot = retarget_arbitrary_length(self.model, normalized, 120, 20, 20)
            output = (robot * (self.stats["robot_std"] + 1e-8) + self.stats["robot_mean"]).float()
        if output.shape != (len(physical_human), 39) or not bool(torch.isfinite(output).all()):
            raise ValueError("decoder lost frame coverage or produced invalid robot motion")
        return output

    @torch.no_grad()
    def decode(self, physical_human: torch.Tensor) -> torch.Tensor:
        # Historical benchmark correction runs on CPU after saving raw output.
        # Keep that arithmetic location for exact regression, then restore device.
        raw = self.decode_raw(physical_human)
        return align_root_rz90(raw.cpu()).to(self.device)
