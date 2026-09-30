# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).


from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


# Architecture constructors and inference operations extracted from the frozen sources.
# Training loops, losses, stochastic VAE sampling, and dataset imports are omitted.


@dataclass
class MotionVAEConfig:
    """Configuration for Motion VAE model architecture."""

    # Input/output dimensions
    motion_dim: int = 136  # root_6d(6) + yaw_delta(1) + root_vel(3) + body_6d(126)
    latent_dim: int = 256  # VAE latent space dimension

    # Transformer architecture
    hidden_dim: int = 512  # Transformer hidden dimension
    num_layers: int = 4  # Number of transformer layers
    num_heads: int = 8  # Number of attention heads
    ff_dim: int = 1024  # Feed-forward dimension
    dropout: float = 0.1  # Dropout rate

    # Sequence parameters
    num_frames: int = 120  # Number of frames per clip (4s @ 30fps)
    fps: float = 30.0  # Frame rate

    # Activation
    activation: str = "gelu"

    # Chunk latent: encode T frames as chunks each with own z
    use_chunk_latent: bool = False
    chunk_len: int = 20  # frames per chunk (120 / 20 = 6 chunks)

    # Joint-aware spatial modeling
    use_joint_mixer: bool = False

    # Split latent: decouple upper-body and lower-body into separate
    # latent sub-spaces. When enabled, the encoder emits two (mu, logvar) heads
    # of sizes latent_dim_upper and latent_dim_lower; they're concatenated at
    # decode time so the downstream DiT still sees a single (B, C, latent_dim)
    # latent (latent_dim == latent_dim_upper + latent_dim_lower).
    split_latent: bool = False
    latent_dim_upper: int = 192
    latent_dim_lower: int = 64

    # when split_decoder_hard=True AND split_latent=True, replace the
    # single decoder with two parallel branches. Upper branch takes only
    # z_upper and outputs only upper-body body_pose dims. Lower branch takes
    # only z_lower and outputs only root + lower-body dims. This is a strict
    # information cut — z_lower literally cannot affect upper-body output
    # and vice versa, unlike the soft KL prior in plain split_latent mode.
    split_decoder_hard: bool = False


@dataclass
class RetargetVAEConfig:
    """Architecture configuration for the robot decoder branch."""

    # Must match the human VAE's latent space
    latent_dim: int = 256
    hidden_dim: int = 512
    num_layers: int = 4
    num_heads: int = 8
    ff_dim: int = 1024
    dropout: float = 0.1

    # Sequence parameters (must match human VAE: T=480, chunk=20)
    num_frames: int = 480
    fps: float = 30.0
    chunk_len: int = 20

    # Robot output dimension
    robot_motion_dim: int = 39

    # Activation (match human VAE)
    activation: str = "gelu"

    # Split latent (match human VAE hard-split decoder).
    # When enabled, z is treated as [z_upper || z_lower] and decoded by
    # two independent Transformer branches with strict information cut.
    split_latent: bool = False
    split_decoder_hard: bool = False
    latent_dim_upper: int = 192
    latent_dim_lower: int = 64

    # Latent adapter
    use_latent_adapter: bool = False
    adapter_hidden_dim: int = 512


SMPL_PARENTS = [
    -1,  # 0: pelvis (root)
    0,  # 1: left_hip -> pelvis
    0,  # 2: right_hip -> pelvis
    0,  # 3: spine1 -> pelvis
    1,  # 4: left_knee -> left_hip
    2,  # 5: right_knee -> right_hip
    3,  # 6: spine2 -> spine1
    4,  # 7: left_ankle -> left_knee
    5,  # 8: right_ankle -> right_knee
    6,  # 9: spine3 -> spine2
    7,  # 10: left_foot -> left_ankle
    8,  # 11: right_foot -> right_ankle
    9,  # 12: neck -> spine3
    9,  # 13: left_collar -> spine3
    9,  # 14: right_collar -> spine3
    12,  # 15: head -> neck
    13,  # 16: left_shoulder -> left_collar
    14,  # 17: right_shoulder -> right_collar
    16,  # 18: left_elbow -> left_shoulder
    17,  # 19: right_elbow -> right_shoulder
    18,  # 20: left_wrist -> left_elbow
    19,  # 21: right_wrist -> right_elbow
]


def _build_body_adjacency() -> torch.Tensor:
    """Build normalized 21x21 adjacency for SMPL body joints (excluding root)."""
    num_joints = 21
    adj = torch.zeros(num_joints, num_joints)

    for j in range(1, 22):
        parent = SMPL_PARENTS[j]
        if parent >= 1:
            joint_idx = j - 1
            parent_idx = parent - 1
            adj[joint_idx, parent_idx] = 1.0
            adj[parent_idx, joint_idx] = 1.0

    adj = adj + torch.eye(num_joints)
    deg = adj.sum(dim=1)
    deg_inv_sqrt = deg.pow(-0.5)
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

    return deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for transformer."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)  # (max_len, 1, d_model)

        self.pe: torch.Tensor
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding.

        Args:
            x: Input tensor, shape (T, B, D).

        Returns:
            Tensor with positional encoding added.
        """
        pe = cast(torch.Tensor, self.pe)
        x = x + pe[: x.shape[0], :]
        return self.dropout(x)


class MotionEncoder(nn.Module):
    """Transformer encoder for motion sequences.

    Maps motion sequence to latent distribution parameters (mu, logvar).
    Uses learnable mu/sigma query tokens (following ACTOR).
    """

    def __init__(self, config: MotionVAEConfig) -> None:
        super().__init__()
        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.motion_dim, config.hidden_dim)

        # Positional encoding. `max_len` is intentionally generous so we can
        # hot-start from a short-sequence checkpoint (num_frames=120) into a
        # long-sequence config (num_frames=480+) without re-allocating the
        # sinusoidal buffer.
        self.pos_encoder = PositionalEncoding(
            config.hidden_dim, config.dropout, max_len=max(config.num_frames + 10, 1024)
        )

        # Learnable query tokens for mu and sigma. When split_latent is enabled
        # we use 4 query tokens (mu_upper, sigma_upper, mu_lower, sigma_lower)
        # so the transformer can attend to different motion aspects for each
        # branch — rather than forcing one shared representation to carry both.
        self.mu_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        self.sigma_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
        if config.split_latent:
            self.mu_lower_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))
            self.sigma_lower_query = nn.Parameter(torch.randn(1, 1, config.hidden_dim))

        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=False,  # (T, B, D) format
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=config.num_layers)

        # Output projections to latent space. Under split_latent, the full
        # latent_dim = latent_dim_upper + latent_dim_lower is produced by two
        # separate Linear heads and concatenated. This keeps downstream DiT /
        # inference code unchanged — they still see a (B, C, latent_dim) tensor
        # — while exposing the structural split to the VAE loss.
        if config.split_latent:
            assert config.latent_dim == config.latent_dim_upper + config.latent_dim_lower, (
                f"split_latent requires latent_dim ({config.latent_dim}) == "
                f"latent_dim_upper ({config.latent_dim_upper}) + "
                f"latent_dim_lower ({config.latent_dim_lower})"
            )
            self.mu_proj = nn.Linear(config.hidden_dim, config.latent_dim_upper)
            self.logvar_proj = nn.Linear(config.hidden_dim, config.latent_dim_upper)
            self.mu_lower_proj = nn.Linear(config.hidden_dim, config.latent_dim_lower)
            self.logvar_lower_proj = nn.Linear(config.hidden_dim, config.latent_dim_lower)
        else:
            self.mu_proj = nn.Linear(config.hidden_dim, config.latent_dim)
            self.logvar_proj = nn.Linear(config.hidden_dim, config.latent_dim)

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode motion sequence to latent distribution.

        Args:
            motion: Motion sequence, shape (B, T, D).

        Returns:
            mu: Mean of latent distribution, shape (B, latent_dim).
            logvar: Log variance of latent distribution, shape (B, latent_dim).
        """
        B, T, D = motion.shape

        # Project input: (B, T, D) -> (B, T, hidden_dim)
        x = self.input_proj(motion)

        # Transpose to (T, B, hidden_dim) for transformer
        x = x.transpose(0, 1)

        # Prepend mu and sigma query tokens. Split-latent uses 4 queries so
        # the transformer can route separate representations to upper vs lower.
        mu_q = self.mu_query.expand(1, B, -1)  # (1, B, hidden_dim)
        sigma_q = self.sigma_query.expand(1, B, -1)  # (1, B, hidden_dim)
        if self.config.split_latent:
            mu_l_q = self.mu_lower_query.expand(1, B, -1)
            sigma_l_q = self.sigma_lower_query.expand(1, B, -1)
            x = torch.cat([mu_q, sigma_q, mu_l_q, sigma_l_q, x], dim=0)  # (T+4, B, hidden_dim)
        else:
            x = torch.cat([mu_q, sigma_q, x], dim=0)  # (T+2, B, hidden_dim)

        # Add positional encoding
        x = self.pos_encoder(x)

        # Transformer encoding
        x = self.transformer(x)

        # Extract mu and logvar from query positions. Under split_latent we
        # read 4 query slots and concatenate upper / lower halves, returning
        # a single (B, latent_dim) tensor for backward-compat with callers.
        if self.config.split_latent:
            mu_upper = self.mu_proj(x[0])  # (B, latent_dim_upper)
            logvar_upper = self.logvar_proj(x[1])  # (B, latent_dim_upper)
            mu_lower = self.mu_lower_proj(x[2])  # (B, latent_dim_lower)
            logvar_lower = self.logvar_lower_proj(x[3])
            mu = torch.cat([mu_upper, mu_lower], dim=-1)
            logvar = torch.cat([logvar_upper, logvar_lower], dim=-1)
        else:
            mu = self.mu_proj(x[0])  # (B, latent_dim)
            logvar = self.logvar_proj(x[1])  # (B, latent_dim)

        return mu, logvar


class MotionDecoder(nn.Module):
    """Transformer decoder for motion sequences.

    Maps latent vector to motion sequence using cross-attention.
    """

    def __init__(self, config: MotionVAEConfig) -> None:
        super().__init__()
        self.config = config

        # Latent projection
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        # Learnable time queries (one for each frame)
        self.time_queries = nn.Parameter(torch.randn(config.num_frames, 1, config.hidden_dim))

        # Positional encoding for time queries. See MotionEncoder for why
        # max_len is generous.
        self.pos_encoder = PositionalEncoding(
            config.hidden_dim, config.dropout, max_len=max(config.num_frames + 10, 1024)
        )

        # Transformer decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=False,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)

        # Output projection
        self.output_proj = nn.Linear(config.hidden_dim, config.motion_dim)

    def forward(self, z: torch.Tensor, num_frames: int | None = None) -> torch.Tensor:
        """Decode latent vector to motion sequence.

        Args:
            z: Latent vector, shape (B, latent_dim).
            num_frames: Number of frames to generate. If None, uses config default.

        Returns:
            Motion sequence, shape (B, T, D).
        """
        B = z.shape[0]
        T = num_frames or self.config.num_frames

        # Project latent: (B, latent_dim) -> (B, hidden_dim)
        memory = self.latent_proj(z)
        memory = memory.unsqueeze(0)  # (1, B, hidden_dim)

        # Create time queries
        if T == self.config.num_frames:
            tgt = self.time_queries.expand(-1, B, -1)  # (T, B, hidden_dim)
        else:
            # Interpolate time queries for different lengths
            tgt = (
                F.interpolate(
                    self.time_queries.permute(1, 2, 0),  # (1, hidden_dim, num_frames)
                    size=T,
                    mode="linear",
                    align_corners=True,
                )
                .permute(2, 0, 1)
                .expand(-1, B, -1)
            )  # (T, B, hidden_dim)

        # Add positional encoding
        tgt = self.pos_encoder(tgt)

        # Transformer decoding
        output = self.transformer(tgt, memory)  # (T, B, hidden_dim)

        # Project to motion space
        output = self.output_proj(output)  # (T, B, motion_dim)

        # Transpose back to (B, T, D)
        output = output.transpose(0, 1)

        return output


class ChunkLatentEncoder(nn.Module):
    """Wraps MotionEncoder to produce per-chunk latent distributions."""

    def __init__(self, config: MotionVAEConfig) -> None:
        super().__init__()
        if config.chunk_len <= 0:
            raise ValueError(f"chunk_len must be positive, got {config.chunk_len}")

        self.chunk_len = config.chunk_len
        if config.num_frames % self.chunk_len != 0:
            raise ValueError(
                f"num_frames ({config.num_frames}) must be divisible by "
                f"chunk_len ({self.chunk_len})"
            )

        chunk_config = MotionVAEConfig(
            motion_dim=config.motion_dim,
            latent_dim=config.latent_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            ff_dim=config.ff_dim,
            dropout=config.dropout,
            num_frames=self.chunk_len,
            fps=config.fps,
            activation=config.activation,
            split_latent=config.split_latent,
            latent_dim_upper=config.latent_dim_upper,
            latent_dim_lower=config.latent_dim_lower,
        )
        self.encoder = MotionEncoder(chunk_config)

    def forward(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode motion into per-chunk latent params.

        Args:
            motion: Motion sequence, shape (B, T, D).

        Returns:
            mu: Mean of latent distribution, shape (B, C, latent_dim).
            logvar: Log variance, shape (B, C, latent_dim).
        """
        B, T, D = motion.shape
        num_chunks = T // self.chunk_len
        if num_chunks <= 0:
            raise ValueError(f"Input length T={T} must be at least chunk_len={self.chunk_len}")

        chunks = motion[:, : num_chunks * self.chunk_len].reshape(B * num_chunks, self.chunk_len, D)
        mu, logvar = self.encoder(chunks)

        return mu.reshape(B, num_chunks, -1), logvar.reshape(B, num_chunks, -1)


class ChunkLatentDecoder(nn.Module):
    """Decoder that cross-attends time queries to chunk latent memory tokens."""

    def __init__(self, config: MotionVAEConfig) -> None:
        super().__init__()
        self.config = config
        if config.chunk_len <= 0:
            raise ValueError(f"chunk_len must be positive, got {config.chunk_len}")

        self.chunk_len = config.chunk_len
        if config.num_frames % self.chunk_len != 0:
            raise ValueError(
                f"num_frames ({config.num_frames}) must be divisible by "
                f"chunk_len ({self.chunk_len})"
            )

        self.num_chunks = config.num_frames // self.chunk_len

        # hard-split decoder. When config.split_latent is True, build
        # TWO parallel branches — upper-body and lower-body. The upper branch
        # projects only z_upper (dims [0:latent_dim_upper]) and outputs only
        # the motion dims 10 + j*6 for j ∈ upper-body joint indices (+ no
        # root). The lower branch projects only z_lower and outputs the root
        # dims [0:10] and lower-body body_pose dims. Outputs are stitched.
        # This gives a TRUE information bottleneck: z_lower literally cannot
        # affect upper-body output because it doesn't feed that branch.
        self.hard_split = bool(
            getattr(config, "split_latent", False) and getattr(config, "split_decoder_hard", False)
        )

        # Dimension-index tables (SMPL): lower body joints = hips, knees,
        # ankles, feet (body_pose indices 0,1,3,4,6,7,9,10). Upper body =
        # rest of the 21 body_pose joints. motion_repr layout:
        #   [0:10]       root_orient_6d(6) + yaw_delta(1) + root_vel_local(3)
        #   [10:136]     body_pose_6d (21 joints × 6)
        if self.hard_split:
            lower_joints = [0, 1, 3, 4, 6, 7, 9, 10]
            upper_joints = [i for i in range(21) if i not in lower_joints]
            lower_dims: list[int] = list(range(0, 10))
            for j in lower_joints:
                lower_dims.extend(range(10 + j * 6, 10 + (j + 1) * 6))
            upper_dims: list[int] = []
            for j in upper_joints:
                upper_dims.extend(range(10 + j * 6, 10 + (j + 1) * 6))
            assert len(lower_dims) + len(upper_dims) == config.motion_dim, (
                f"dim split sums must equal motion_dim={config.motion_dim}"
            )
            self.register_buffer(
                "lower_out_idx",
                torch.tensor(lower_dims, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "upper_out_idx",
                torch.tensor(upper_dims, dtype=torch.long),
                persistent=False,
            )

            # Two parallel transformer decoders, each seeing only its own latent.
            self.latent_proj_upper = nn.Linear(config.latent_dim_upper, config.hidden_dim)
            self.latent_proj_lower = nn.Linear(config.latent_dim_lower, config.hidden_dim)
            # Each branch has its own chunk_pos_encoding + time_queries to
            # enable hot-start from a non-split checkpoint below.
            self.chunk_pos_encoding_upper = nn.Parameter(
                torch.randn(self.num_chunks, 1, config.hidden_dim)
            )
            self.chunk_pos_encoding_lower = nn.Parameter(
                torch.randn(self.num_chunks, 1, config.hidden_dim)
            )
            self.time_queries_upper = nn.Parameter(
                torch.randn(config.num_frames, 1, config.hidden_dim)
            )
            self.time_queries_lower = nn.Parameter(
                torch.randn(config.num_frames, 1, config.hidden_dim)
            )
            self.pos_encoder = PositionalEncoding(
                config.hidden_dim, config.dropout, max_len=max(config.num_frames + 10, 1024)
            )
            dec_layer_u = nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=False,
            )
            dec_layer_l = nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=False,
            )
            self.transformer_upper = nn.TransformerDecoder(
                dec_layer_u, num_layers=config.num_layers
            )
            self.transformer_lower = nn.TransformerDecoder(
                dec_layer_l, num_layers=config.num_layers
            )
            self.output_proj_upper = nn.Linear(config.hidden_dim, len(upper_dims))
            self.output_proj_lower = nn.Linear(config.hidden_dim, len(lower_dims))
            # Keep placeholders so old state dicts partially-load without crash.
            self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)
            self.chunk_pos_encoding = nn.Parameter(
                torch.randn(self.num_chunks, 1, config.hidden_dim)
            )
            self.time_queries = nn.Parameter(torch.randn(config.num_frames, 1, config.hidden_dim))
            dec_layer = nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=False,
            )
            self.transformer = nn.TransformerDecoder(dec_layer, num_layers=config.num_layers)
            self.output_proj = nn.Linear(config.hidden_dim, config.motion_dim)
            return

        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        self.chunk_pos_encoding = nn.Parameter(torch.randn(self.num_chunks, 1, config.hidden_dim))

        self.time_queries = nn.Parameter(torch.randn(config.num_frames, 1, config.hidden_dim))

        # Generous max_len so the sinusoidal buffer covers hot-starts from
        # shorter checkpoints without re-allocation.
        self.pos_encoder = PositionalEncoding(
            config.hidden_dim, config.dropout, max_len=max(config.num_frames + 10, 1024)
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=False,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)

        self.output_proj = nn.Linear(config.hidden_dim, config.motion_dim)

    def _get_time_queries(self, T: int, batch_size: int) -> torch.Tensor:
        if T == self.config.num_frames:
            return self.time_queries.expand(-1, batch_size, -1)

        return (
            F.interpolate(
                self.time_queries.permute(1, 2, 0),
                size=T,
                mode="linear",
                align_corners=True,
            )
            .permute(2, 0, 1)
            .expand(-1, batch_size, -1)
        )

    def _get_chunk_pos_encoding(self, num_chunks: int) -> torch.Tensor:
        if num_chunks == self.num_chunks:
            return self.chunk_pos_encoding

        return F.interpolate(
            self.chunk_pos_encoding.permute(1, 2, 0),
            size=num_chunks,
            mode="linear",
            align_corners=True,
        ).permute(2, 0, 1)

    def _interp_param(self, param: torch.Tensor, target_size: int) -> torch.Tensor:
        """Interpolate a learnable (L, 1, D) param along L to target_size."""
        if param.shape[0] == target_size:
            return param
        return F.interpolate(
            param.permute(1, 2, 0),
            size=target_size,
            mode="linear",
            align_corners=True,
        ).permute(2, 0, 1)

    def forward(self, z: torch.Tensor, num_frames: int | None = None) -> torch.Tensor:
        """Decode chunk latents to motion sequence.

        Args:
            z: Chunk latent vectors, shape (B, C, latent_dim).
            num_frames: Number of frames to generate. If None, uses config default.

        Returns:
            Motion sequence, shape (B, T, D).
        """
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape (B, C, latent_dim), got {tuple(z.shape)}")

        B = z.shape[0]
        T = num_frames or self.config.num_frames
        num_chunks = z.shape[1]

        if self.hard_split:
            # Split z -> upper and lower latent branches.
            z_upper = z[..., : self.config.latent_dim_upper]
            z_lower = z[..., self.config.latent_dim_upper :]
            # Upper branch
            mem_u = self.latent_proj_upper(z_upper).transpose(0, 1)
            mem_u = mem_u + self._interp_param(self.chunk_pos_encoding_upper, num_chunks)
            tq_u = self._interp_param(self.time_queries_upper, T).expand(-1, B, -1)
            tq_u = self.pos_encoder(tq_u)
            out_u = self.transformer_upper(tq_u, mem_u)
            out_u = self.output_proj_upper(out_u)  # (T, B, |upper_dims|)
            # Lower branch
            mem_l = self.latent_proj_lower(z_lower).transpose(0, 1)
            mem_l = mem_l + self._interp_param(self.chunk_pos_encoding_lower, num_chunks)
            tq_l = self._interp_param(self.time_queries_lower, T).expand(-1, B, -1)
            tq_l = self.pos_encoder(tq_l)
            out_l = self.transformer_lower(tq_l, mem_l)
            out_l = self.output_proj_lower(out_l)  # (T, B, |lower_dims|)
            # Scatter to full (T, B, motion_dim)
            full = torch.zeros(T, B, self.config.motion_dim, device=z.device, dtype=out_u.dtype)
            full[:, :, self.upper_out_idx] = out_u
            full[:, :, self.lower_out_idx] = out_l
            return full.transpose(0, 1)

        memory = self.latent_proj(z).transpose(0, 1)
        memory = memory + self._get_chunk_pos_encoding(num_chunks)

        tgt = self._get_time_queries(T, B)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(tgt, memory)
        output = self.output_proj(output)

        return output.transpose(0, 1)


class JointTopologyMixer(nn.Module):
    """Graph-based spatial mixer over body_pose_6d joints."""

    def __init__(self, joint_dim: int = 6, hidden_dim: int = 32, num_joints: int = 21) -> None:
        super().__init__()
        self.joint_dim = joint_dim
        self.num_joints = num_joints
        self.body_start = 10
        self.body_end = 136

        adj_norm = _build_body_adjacency()
        self.adj_norm: torch.Tensor
        self.register_buffer("adj_norm", adj_norm)

        self.mlp = nn.Sequential(
            nn.Linear(joint_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, joint_dim),
        )

    def forward(self, motion: torch.Tensor) -> torch.Tensor:
        """Apply topology-aware body joint mixing.

        Args:
            motion: Full motion representation, shape (B, T, 136).

        Returns:
            Motion tensor with mixed body_pose_6d, shape (B, T, 136).
        """
        B, T, D = motion.shape
        if D < self.body_end:
            raise ValueError(f"Expected motion dimension >= {self.body_end}, got {D}")

        body_pose = motion[:, :, self.body_start : self.body_end]
        expected_dim = self.num_joints * self.joint_dim
        if body_pose.shape[-1] != expected_dim:
            raise ValueError(f"Expected body pose dim {expected_dim}, got {body_pose.shape[-1]}")

        body_pose = body_pose.reshape(B * T, self.num_joints, self.joint_dim)
        aggregated = torch.matmul(self.adj_norm, body_pose)
        body_pose_out = body_pose + self.mlp(aggregated)
        body_pose_out = body_pose_out.reshape(B, T, expected_dim)

        out = motion.clone()
        out[:, :, self.body_start : self.body_end] = body_pose_out
        return out


class MotionVAE(nn.Module):
    def __init__(self, config: MotionVAEConfig) -> None:
        super().__init__()
        self.config = config

        self.joint_mixer = JointTopologyMixer() if config.use_joint_mixer else None

        if config.use_chunk_latent:
            self.encoder = ChunkLatentEncoder(config)
            self.decoder = ChunkLatentDecoder(config)
        else:
            self.encoder = MotionEncoder(config)
            self.decoder = MotionDecoder(config)

        logger.info(
            (
                "MotionVAE initialized: motion_dim=%d, latent_dim=%d, hidden_dim=%d, "
                "layers=%d, chunk_latent=%s, joint_mixer=%s"
            ),
            config.motion_dim,
            config.latent_dim,
            config.hidden_dim,
            config.num_layers,
            config.use_chunk_latent,
            config.use_joint_mixer,
        )

    def encode(self, motion: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Encode motion to latent distribution.

        Args:
            motion: Motion sequence, shape (B, T, D).

        Returns:
            mu: Mean, shape (B, latent_dim) or (B, C, latent_dim).
            logvar: Log variance, shape (B, latent_dim) or (B, C, latent_dim).
        """
        return self.encoder(motion)


class RobotChunkLatentDecoder(nn.Module):
    """Decoder that maps chunk latents to robot motion.

    Supports two modes:
      - Standard (single path): full latent → single Transformer → 39-dim output
      - Hard-split (dual path): z_upper → upper Transformer → 17-dim (waist+arms)
                                 z_lower → lower Transformer → 22-dim (root+legs)

    Hard-split mirrors the human ChunkLatentDecoder architecture,
    providing strict information bottleneck between upper/lower body.
    """

    def __init__(self, config: RetargetVAEConfig) -> None:
        super().__init__()
        self.config = config

        if config.chunk_len <= 0:
            raise ValueError(f"chunk_len must be positive, got {config.chunk_len}")
        if config.num_frames % config.chunk_len != 0:
            raise ValueError(
                f"num_frames ({config.num_frames}) must be divisible by "
                f"chunk_len ({config.chunk_len})"
            )

        self.chunk_len = config.chunk_len
        self.num_chunks = config.num_frames // config.chunk_len

        # Hard-split mode: two parallel decoder branches
        self.hard_split = bool(
            getattr(config, "split_latent", False) and getattr(config, "split_decoder_hard", False)
        )

        if self.hard_split:
            # Robot 39-dim output split:
            #   Lower branch: [0:10] root (base_orient_6d + yaw_delta + base_vel)
            #                 [10:22] left_leg(6) + right_leg(6)
            #   Upper branch: [22:25] waist(3)
            #                 [25:32] left_arm(7)
            #                 [32:39] right_arm(7)
            lower_dims = list(range(0, 22))  # 22 dims
            upper_dims = list(range(22, 39))  # 17 dims
            assert len(lower_dims) + len(upper_dims) == config.robot_motion_dim, (
                f"dim split sums must equal robot_motion_dim={config.robot_motion_dim}"
            )
            self.register_buffer(
                "lower_out_idx",
                torch.tensor(lower_dims, dtype=torch.long),
                persistent=False,
            )
            self.register_buffer(
                "upper_out_idx",
                torch.tensor(upper_dims, dtype=torch.long),
                persistent=False,
            )

            # Upper branch (z_upper → waist + arms)
            self.latent_proj_upper = nn.Linear(config.latent_dim_upper, config.hidden_dim)
            self.chunk_pos_encoding_upper = nn.Parameter(
                torch.randn(self.num_chunks, 1, config.hidden_dim)
            )
            self.time_queries_upper = nn.Parameter(
                torch.randn(config.num_frames, 1, config.hidden_dim)
            )
            dec_layer_u = nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=False,
            )
            self.transformer_upper = nn.TransformerDecoder(
                dec_layer_u, num_layers=config.num_layers
            )
            self.output_proj_upper = nn.Linear(config.hidden_dim, len(upper_dims))

            # Lower branch (z_lower → root + legs)
            self.latent_proj_lower = nn.Linear(config.latent_dim_lower, config.hidden_dim)
            self.chunk_pos_encoding_lower = nn.Parameter(
                torch.randn(self.num_chunks, 1, config.hidden_dim)
            )
            self.time_queries_lower = nn.Parameter(
                torch.randn(config.num_frames, 1, config.hidden_dim)
            )
            dec_layer_l = nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=False,
            )
            self.transformer_lower = nn.TransformerDecoder(
                dec_layer_l, num_layers=config.num_layers
            )
            self.output_proj_lower = nn.Linear(config.hidden_dim, len(lower_dims))

            # Shared positional encoding (sinusoidal, same for both branches)
            self.pos_encoder = PositionalEncoding(
                config.hidden_dim,
                config.dropout,
                max_len=max(config.num_frames + 10, 1024),
            )

            # Keep placeholders for backward-compatible state_dict loading
            self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)
            self.chunk_pos_encoding = nn.Parameter(
                torch.randn(self.num_chunks, 1, config.hidden_dim)
            )
            self.time_queries = nn.Parameter(torch.randn(config.num_frames, 1, config.hidden_dim))
            dec_layer = nn.TransformerDecoderLayer(
                d_model=config.hidden_dim,
                nhead=config.num_heads,
                dim_feedforward=config.ff_dim,
                dropout=config.dropout,
                activation=config.activation,
                batch_first=False,
            )
            self.transformer = nn.TransformerDecoder(dec_layer, num_layers=config.num_layers)
            self.output_proj = nn.Linear(config.hidden_dim, config.robot_motion_dim)
            return

        # Standard (non-split) path
        self.latent_proj = nn.Linear(config.latent_dim, config.hidden_dim)

        self.chunk_pos_encoding = nn.Parameter(torch.randn(self.num_chunks, 1, config.hidden_dim))

        self.time_queries = nn.Parameter(torch.randn(config.num_frames, 1, config.hidden_dim))

        self.pos_encoder = PositionalEncoding(
            config.hidden_dim,
            config.dropout,
            max_len=max(config.num_frames + 10, 1024),
        )

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.ff_dim,
            dropout=config.dropout,
            activation=config.activation,
            batch_first=False,
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=config.num_layers)

        self.output_proj = nn.Linear(config.hidden_dim, config.robot_motion_dim)

    def _interp_param(self, param: torch.Tensor, target_size: int) -> torch.Tensor:
        """Interpolate a learnable (L, 1, D) param along L to target_size."""
        if param.shape[0] == target_size:
            return param
        return F.interpolate(
            param.permute(1, 2, 0),
            size=target_size,
            mode="linear",
            align_corners=True,
        ).permute(2, 0, 1)

    def forward(self, z: torch.Tensor, num_frames: int | None = None) -> torch.Tensor:
        """Decode chunk latents to robot motion sequence.

        Args:
            z: Chunk latent vectors, shape (B, C, latent_dim).
            num_frames: Number of output frames. If None, uses config default.

        Returns:
            Robot motion sequence, shape (B, T, 39).
        """
        if z.ndim != 3:
            raise ValueError(f"Expected z with shape (B, C, latent_dim), got {tuple(z.shape)}")

        B = z.shape[0]
        T = num_frames or self.config.num_frames
        num_chunks = z.shape[1]

        if self.hard_split:
            # Split z into upper and lower latent branches
            z_upper = z[..., : self.config.latent_dim_upper]  # (B, C, 192)
            z_lower = z[..., self.config.latent_dim_upper :]  # (B, C, 64)

            # Upper branch: waist + arms → 17 dims
            mem_u = self.latent_proj_upper(z_upper).transpose(0, 1)  # (C, B, hidden)
            mem_u = mem_u + self._interp_param(self.chunk_pos_encoding_upper, num_chunks)
            tq_u = self._interp_param(self.time_queries_upper, T).expand(-1, B, -1)
            tq_u = self.pos_encoder(tq_u)
            out_u = self.transformer_upper(tq_u, mem_u)  # (T, B, hidden)
            out_u = self.output_proj_upper(out_u)  # (T, B, 17)

            # Lower branch: root + legs → 22 dims
            mem_l = self.latent_proj_lower(z_lower).transpose(0, 1)  # (C, B, hidden)
            mem_l = mem_l + self._interp_param(self.chunk_pos_encoding_lower, num_chunks)
            tq_l = self._interp_param(self.time_queries_lower, T).expand(-1, B, -1)
            tq_l = self.pos_encoder(tq_l)
            out_l = self.transformer_lower(tq_l, mem_l)  # (T, B, hidden)
            out_l = self.output_proj_lower(out_l)  # (T, B, 22)

            # Scatter to full robot motion (T, B, 39)
            full = torch.zeros(
                T, B, self.config.robot_motion_dim, device=z.device, dtype=out_u.dtype
            )
            full[:, :, self.upper_out_idx] = out_u
            full[:, :, self.lower_out_idx] = out_l
            return full.transpose(0, 1)  # (B, T, 39)

        # Standard (non-split) path
        memory = self.latent_proj(z).transpose(0, 1)  # (C, B, hidden_dim)
        memory = memory + self._interp_param(self.chunk_pos_encoding, num_chunks)

        tgt = self._interp_param(self.time_queries, T).expand(-1, B, -1)
        tgt = self.pos_encoder(tgt)

        output = self.transformer(tgt, memory)  # (T, B, hidden_dim)
        output = self.output_proj(output)  # (T, B, 39)

        return output.transpose(0, 1)  # (B, T, 39)


class LatentAdapter(nn.Module):
    """Residual MLP adapter for latent space transformation.

    Transforms z → z + MLP(z) with zero-initialized output layer
    so that it starts as an identity mapping.
    """

    def __init__(self, latent_dim: int = 256, hidden_dim: int = 512) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(latent_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        # Zero-init output layer so adapter starts as identity
        nn.init.zeros_(self.mlp[2].weight)
        nn.init.zeros_(self.mlp[2].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply residual adapter.

        Args:
            z: Latent vectors, shape (..., latent_dim).

        Returns:
            Adapted latent vectors, shape (..., latent_dim).
        """
        return z + self.mlp(z)


class SplitLatentAdapter(nn.Module):
    """Split-aware residual adapter that processes upper and lower latents independently.

    Each branch has its own MLP with zero-initialized output for identity start.
    This respects the hard-split information bottleneck — z_upper adapter
    cannot affect z_lower and vice versa.
    """

    def __init__(
        self,
        latent_dim_upper: int = 192,
        latent_dim_lower: int = 64,
        hidden_dim_upper: int = 512,
        hidden_dim_lower: int = 256,
    ) -> None:
        super().__init__()
        self.latent_dim_upper = latent_dim_upper

        self.mlp_upper = nn.Sequential(
            nn.Linear(latent_dim_upper, hidden_dim_upper),
            nn.GELU(),
            nn.Linear(hidden_dim_upper, latent_dim_upper),
        )
        nn.init.zeros_(self.mlp_upper[2].weight)
        nn.init.zeros_(self.mlp_upper[2].bias)

        self.mlp_lower = nn.Sequential(
            nn.Linear(latent_dim_lower, hidden_dim_lower),
            nn.GELU(),
            nn.Linear(hidden_dim_lower, latent_dim_lower),
        )
        nn.init.zeros_(self.mlp_lower[2].weight)
        nn.init.zeros_(self.mlp_lower[2].bias)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """Apply split residual adapter.

        Args:
            z: Concatenated latent vectors [z_upper || z_lower], shape (..., latent_dim).

        Returns:
            Adapted latent vectors, shape (..., latent_dim).
        """
        z_upper = z[..., : self.latent_dim_upper]
        z_lower = z[..., self.latent_dim_upper :]
        z_upper = z_upper + self.mlp_upper(z_upper)
        z_lower = z_lower + self.mlp_lower(z_lower)
        return torch.cat([z_upper, z_lower], dim=-1)


class RetargetVAE(nn.Module):
    def __init__(
        self,
        human_vae: MotionVAE,
        config: RetargetVAEConfig,
    ) -> None:
        super().__init__()

        # Frozen human VAE (encoder + decoder for monitoring)
        self.human_vae = human_vae
        for param in self.human_vae.parameters():
            param.requires_grad = False

        # Trainable robot decoder
        self.robot_decoder = RobotChunkLatentDecoder(config)

        # Optional latent adapter (split-aware when hard-split enabled)
        self.latent_adapter: LatentAdapter | SplitLatentAdapter | None = None
        if config.use_latent_adapter:
            if config.split_latent and config.split_decoder_hard:
                self.latent_adapter = SplitLatentAdapter(
                    latent_dim_upper=config.latent_dim_upper,
                    latent_dim_lower=config.latent_dim_lower,
                    hidden_dim_upper=config.adapter_hidden_dim,
                    hidden_dim_lower=min(config.adapter_hidden_dim, 256),
                )
            else:
                self.latent_adapter = LatentAdapter(
                    latent_dim=config.latent_dim,
                    hidden_dim=config.adapter_hidden_dim,
                )

        self.config = config

        # Log parameter counts
        robot_params = sum(p.numel() for p in self.robot_decoder.parameters())
        adapter_params = (
            sum(p.numel() for p in self.latent_adapter.parameters()) if self.latent_adapter else 0
        )
        frozen_params = sum(p.numel() for p in self.human_vae.parameters())
        logger.info(
            "RetargetVAE: robot_decoder=%.2fM, adapter=%.2fM, frozen_human=%.2fM",
            robot_params / 1e6,
            adapter_params / 1e6,
            frozen_params / 1e6,
        )

    @torch.no_grad()
    def retarget(self, human_motion: torch.Tensor) -> torch.Tensor:
        """Deterministic retargeting (inference mode).

        Args:
            human_motion: Normalized human motion, shape (B, T, 136).

        Returns:
            Robot motion, shape (B, T, 39).
        """
        self.eval()
        mu, _ = self.human_vae.encode(human_motion)

        z_robot = mu
        if self.latent_adapter is not None:
            z_robot = self.latent_adapter(z_robot)

        return self.robot_decoder(z_robot, human_motion.shape[1])
