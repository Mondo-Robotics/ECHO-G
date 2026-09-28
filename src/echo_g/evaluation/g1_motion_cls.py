# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

"""G1 motion-generation benchmark — CLASS-BASED metrics (EMAGE mertic.py style).

Same numerics as scripts/eval_g1_motion.py, restructured so each benchmark is a stateful metric
CLASS mirroring PantoMatrix/emage_evaltools/mertic.py: __init__(config) -> update/run/compute
(per clip) -> compute()/avg() (final scalar) -> reset(). The per-clip loop fans each clip out to
every evaluator; results are gathered at the end via .compute()/.avg().

Metrics (all on pre-stored robot_repr[T,39] .pt for pred + reference; no model is run):
  FGD          : Fréchet distance on the learned G1 skeleton-conv AE latent (--g1-ae-ckpt REQUIRED,
                 docs/BENCHMARK.md). EMAGE kernel: scipy-sqrtm with a fixed 1e-6 covariance
                 offset inside sqrtm and no clamp.
  Div          : EMAGE L1div semantics: WITHIN-clip per-frame MAD and frame-weighted. Prediction
                 and corresponding GT are both reported; their absolute gap is the lower-is-better
                 comparison metric. FK uses zero translation and identity global orientation.
  Multimodality: average pairwise L1 distance among N independently sampled motions for the same
                 audio (Audio2Gestures uses N=20). Evaluated in pose-only G1 FK-position space,
                 then averaged equally over complete clips. Requires --multimodality-root.
  BA           : beat_consistency — per-body motion beats (speed minima ∩ moving) vs audio onsets,
                 Gaussian σ=0.3s, audio->motion by default, averaged over upper bodies. Prediction
                 and corresponding GT are both reported; their absolute gap is the lower-is-better
                 comparison metric. FK uses zero translation and identity global orientation.
  SRGR         : semantic-mass-normalized per-frame/per-body FK-position recall. The numerator is
                 semantic-weighted position hits; the denominator is the maximum possible semantic
                 hit mass on the frames actually evaluated, so a perfect prediction is always 1.
                 Its FK uses zero translation and identity global orientation. Default threshold is
                 0.1 m, matching EMAGE's numeric default.
  jerk         : OMG body_jerk_mean on world-space G1 body positions: mean 3rd finite-difference
                 magnitude * fps^3 (m/s^3). Prediction and corresponding GT are both reported;
                 their absolute gap is the lower-is-better comparison metric. Reports both the
                 historical clip-equal aggregation and a length-weighted aggregation using T-3.
  foot_ground_error / contact_sliding_speed : OPTIONAL (--enable-foot-metrics), grounded-stance FK.

Div/BA/SRGR/Multimodality use zero translation and identity root orientation. Jerk reconstructs
the world root trajectory from yaw_delta + base_vel_local, uses the stored full root orientation,
and compares prediction and GT on their common temporal prefix. Only foot metrics need the
grounded-stance z anchor.

Input contract: prediction and reference files must already contain physical 39D G1 robot_repr at
30 FPS in the same coordinate system. This evaluator never de-normalizes or coordinate-aligns them.
Repr tensors are auto-detected under keys robot_repr/robot_motion/pred/robot/motion.

Usage after installing echo-g[benchmark]:
  python scripts/eval_g1_motion_cls.py \
    --pred-dir <dir>/pred --ref-dir <dir>/gt --g1-ae-ckpt <encoder>.bin \
    --wav-dir data/audio --sem-dir data/semantics --mmae-file data/mean_vel.npy \
    --fps 30 --enable-foot-metrics --out out.json
"""

# Preserve archived computational syntax for direct source comparison.
# ruff: noqa: UP004, UP031, B905

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

import numpy as np
import torch

from echo_g.evaluation import g1_kinematics as G1K
from echo_g.evaluation.robot_fgd.featurize import (
    ENCODER_INPUT_NORMALIZATION_PROTOCOL,
    normalize_encoder_input,
)
from echo_g.evaluation.robot_repr import TOTAL_ROBOT_MOTION_DIM  # 39
from echo_g.evaluation.rotation import (
    extract_yaw_from_rotation,
    rotation_6d_to_matrix,
    rotation_matrix_from_yaw,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

# ------------------------------------------------------------------ constants
ROBOT_MOTION_DIM = TOTAL_ROBOT_MOTION_DIM  # 39
NUM_G1_BODIES = G1K.NUM_G1_BODIES  # 30
ROOT_INDEX = 0  # pelvis (BFS body 0)
UP_AXIS = 2  # z
_REPR_KEYS = ("robot_repr", "robot_motion", "pred", "robot", "motion")

# BA upper-body G1 bodies (waist/torso + both arms; gesture-bearing). BFS indices.
UPPER_G1_BODIES = [3, 6, 9, 12, 13, 16, 17, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29]

# Sole proxy spheres (from Unitree G1 URDF ankle_roll_link <collision><sphere>): 4 per
# foot, r=0.005, offsets in the ankle_roll link frame. Same hardware as OMG's URDF.
_ANKLE_BFS = G1K.G1_ANKLE_BFS  # foot_id -> BFS body (left/right ankle_roll_link)
_SOLE_OFFSETS = G1K.G1_SOLE_OFFSETS
_SOLE_RADIUS = G1K.G1_SOLE_RADIUS

_DEFAULT_ENCODER = ""
_DEFAULT_SEM_DIR = ""
_SRGR_SEMANTIC_PROTOCOL = "beat2_official_first_match_30fps_v1"
_REQUIRED_EVAL_FPS = 30.0


# ============================================================================
# G1 forward kinematics (repr[T,39] -> body positions/rotations)  [stateless]
# ============================================================================
def _fk_pos_rot(
    joint_angles_canonical: torch.Tensor, base_rot: torch.Tensor, base_trans: torch.Tensor
):
    """Run the shared G1 FK and return per-body world position and rotation.
    joint_angles_canonical: (T,29) canonical; base_rot: (T,3,3); base_trans: (T,3).
    Returns body_pos (T,30,3), body_rot (T,30,3,3)."""
    positions, rotations = G1K.g1_forward_kinematics_with_rotations(
        joint_angles_canonical.unsqueeze(0),
        base_rot.unsqueeze(0),
        base_trans.unsqueeze(0),
    )
    return positions[0], rotations[0]


def _split_repr(rr: torch.Tensor):
    base_rot = rotation_6d_to_matrix(rr[:, :6])  # (T,3,3)
    yaw_delta = rr[:, 6]
    base_vel_local = rr[:, 7:10]
    joints = rr[:, 10:]  # (T,29) canonical
    return base_rot, yaw_delta, base_vel_local, joints


def _integrate_base_trans(
    yaw_delta: torch.Tensor,
    base_vel_local: torch.Tensor,
    fps: float,
    initial_trans: torch.Tensor,
    initial_yaw: torch.Tensor | float = 0.0,
) -> torch.Tensor:
    """Integrate root velocity -> base_trans (T,3), starting at the supplied world anchor."""
    T = yaw_delta.shape[0]
    dev, dt = yaw_delta.device, yaw_delta.dtype
    yaw = torch.zeros(T, device=dev, dtype=dt)
    yaw[0] = torch.as_tensor(initial_yaw, device=dev, dtype=dt)
    yaw[1:] = yaw[0] + torch.cumsum(yaw_delta[1:], dim=0)
    yaw_rot = rotation_matrix_from_yaw(yaw)  # (T,3,3)
    gvel = torch.einsum("tij,tj->ti", yaw_rot, base_vel_local)
    trans = torch.zeros(T, 3, device=dev, dtype=dt)
    trans[0] = initial_trans
    trans[1:] = initial_trans + torch.cumsum(gvel[1:] / float(fps), dim=0)
    return trans


def _sole_points(body_pos: torch.Tensor, body_rot: torch.Tensor):
    """OMG-style sole proxy points: ankle_body_pos + R(ankle) @ local_offset.
    Returns sole_points (T,P,3), sole_radii (P,), sole_foot_ids (P,)."""
    pts, radii, fids = [], [], []
    for foot_id, bfs in _ANKLE_BFS.items():
        R = body_rot[:, bfs]  # (T,3,3)
        p0 = body_pos[:, bfs]  # (T,3)
        for off in _SOLE_OFFSETS:
            o = torch.tensor(off, device=body_pos.device, dtype=body_pos.dtype)
            pts.append(p0 + torch.einsum("tij,j->ti", R, o))  # (T,3)
            radii.append(_SOLE_RADIUS)
            fids.append(foot_id)
    sole = torch.stack(pts, dim=1)  # (T,P,3)
    return sole, np.array(radii, dtype=np.float64), np.array(fids, dtype=np.int64)


def fk_positions(rr: torch.Tensor) -> np.ndarray:
    """Pose-only FK (root at origin). For the anchor-invariant metrics. -> (T,30,3)."""
    base_rot, _, _, joints = _split_repr(rr)
    base_trans = torch.zeros(rr.shape[0], 3, device=rr.device, dtype=rr.dtype)
    pos, _ = _fk_pos_rot(joints, base_rot, base_trans)
    return pos.cpu().numpy()


def fk_positions_world(rr: torch.Tensor, fps: float) -> np.ndarray:
    """Full world-space FK for OMG body Jerk, with reconstructed root rotation and translation."""
    base_rot, yaw_delta, base_vel_local, joints = _split_repr(rr)
    initial_trans = torch.zeros(3, device=rr.device, dtype=rr.dtype)
    initial_yaw = extract_yaw_from_rotation(base_rot[0])
    base_trans = _integrate_base_trans(
        yaw_delta,
        base_vel_local,
        fps,
        initial_trans,
        initial_yaw,
    )
    pos, _ = _fk_pos_rot(joints, base_rot, base_trans)
    return pos.cpu().numpy()


def fk_positions_ba(rr: torch.Tensor) -> np.ndarray:
    """EMAGE Div/BA FK with zero translation and identity global orientation."""
    _, _, _, joints = _split_repr(rr)
    base_rot = (
        torch.eye(3, device=rr.device, dtype=rr.dtype).unsqueeze(0).expand(rr.shape[0], -1, -1)
    )
    base_trans = torch.zeros(rr.shape[0], 3, device=rr.device, dtype=rr.dtype)
    pos, _ = _fk_pos_rot(joints, base_rot, base_trans)
    return pos.cpu().numpy()


def fk_grounded_world(rr: torch.Tensor, fps: float):
    """Foot-metric FK with a GROUNDED-STANCE anchor (GT-free, OMG _ground_default_root_pos style):
    FK frame-0 at origin -> drop root z so lowest sole-bottom sits at z=0 -> integrate root velocity
    -> full FK. Returns body_pos (T,30,3), sole_points (T,P,3), radii, foot_ids."""
    base_rot, yaw_delta, base_vel_local, joints = _split_repr(rr)
    dev, dt = rr.device, rr.dtype
    p0, r0 = _fk_pos_rot(joints[:1], base_rot[:1], torch.zeros(1, 3, device=dev, dtype=dt))
    sole0, radii0, _ = _sole_points(p0, r0)  # (1,P,3)
    sole_bottom0 = sole0[..., 2] - torch.tensor(radii0, device=dev, dtype=dt).view(1, -1)
    init = torch.zeros(3, device=dev, dtype=dt)
    init[2] = -sole_bottom0.min()
    initial_yaw = extract_yaw_from_rotation(base_rot[0])
    base_trans = _integrate_base_trans(
        yaw_delta,
        base_vel_local,
        fps,
        init,
        initial_yaw,
    )
    pos, rot = _fk_pos_rot(joints, base_rot, base_trans)
    sole, radii, fids = _sole_points(pos, rot)
    return pos, sole, radii, fids


# ============================================================================
# Stateless feature / distance helpers (shared by the metric classes below)
# ============================================================================
def pooled_joint_stats_feature(joints: np.ndarray) -> np.ndarray:
    """Per-clip Div feature: concat of time-mean and time-std of joint xyz.
    joints (T,J,3) -> (J*3*2,). Robot: 30*3*2 = 180.
    (== eval_emage_robot / hermes.benchmark.fid.)
    """
    j = np.asarray(joints, dtype=np.float64)
    return np.concatenate([j.mean(axis=0).reshape(-1), j.std(axis=0).reshape(-1)], axis=0)


def resolve_checkpoint_fps(
    checkpoint: dict,
    expected_fps: float,
) -> float | None:
    """Return optional checkpoint FPS metadata and reject an explicit mismatch."""
    checkpoint_fps_raw = checkpoint.get("fps")
    if checkpoint_fps_raw is None:
        return None
    checkpoint_fps = float(checkpoint_fps_raw)
    if checkpoint_fps != float(expected_fps):
        raise ValueError(
            f"SKCNN/evaluator FPS mismatch: checkpoint={checkpoint_fps} evaluator={expected_fps}"
        )
    return checkpoint_fps


def load_g1_ae(ckpt_path, device, expected_fps):
    """Load a trained G1 skeleton-conv AE and its encoder-input normalization."""
    import types as _types

    from echo_g.evaluation.g1_kinematics import G1_BFS_PARENTS
    from echo_g.evaluation.robot_fgd.motion_encoder import VAESKConv

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    checkpoint_fps = resolve_checkpoint_fps(ck, expected_fps)
    conf = _types.SimpleNamespace(
        vae_layer=ck.get("vae_layer", 4),
        vae_grow=ck.get("vae_grow", [1, 1, 2, 1]),
        variational=False,
        vae_test_dim=ck["vae_test_dim"],
        vae_length=ck["vae_length"],
        channel_base=ck["channel_base"],
    )
    net = VAESKConv(conf, topology=list(ck.get("topology", G1_BFS_PARENTS))).to(device)
    net.load_state_dict({k.replace("module.", ""): v for k, v in ck["model_state"].items()})
    net.eval()
    input_normalization = ck.get("input_normalization")
    if input_normalization is None:
        input_normalization = "zscore" if ck["input_mode"] == "rich12" else "none"
    if input_normalization not in {"none", "zscore"}:
        raise ValueError(f"unknown checkpoint input_normalization={input_normalization!r}")
    normalization_protocol = ck.get("input_normalization_protocol")
    if input_normalization == "zscore":
        if normalization_protocol not in {None, ENCODER_INPUT_NORMALIZATION_PROTOCOL}:
            raise ValueError(
                f"unsupported SKCNN input normalization protocol={normalization_protocol!r}"
            )
        normalization_protocol = ENCODER_INPUT_NORMALIZATION_PROTOCOL
    elif normalization_protocol not in {None, "none"}:
        raise ValueError(
            f"checkpoint says input_normalization=none but protocol={normalization_protocol!r}"
        )
    else:
        normalization_protocol = "none"
    cfg = {
        "input_mode": ck["input_mode"],
        "vae_length": int(ck["vae_length"]),
        "checkpoint_fps": checkpoint_fps,
        "feature_mean": None,
        "feature_std": None,
        "input_normalization": input_normalization,
        "input_normalization_protocol": normalization_protocol,
    }
    if input_normalization == "zscore":
        feature_mean = ck.get("feature_mean", ck.get("joint_mean"))
        feature_std = ck.get("feature_std", ck.get("joint_std"))
        if feature_mean is None or feature_std is None:
            raise ValueError(f"rich12 checkpoint lacks feature mean/std: {ckpt_path}")
        feature_mean = np.asarray(feature_mean, dtype=np.float32).reshape(-1)
        feature_std = np.asarray(feature_std, dtype=np.float32).reshape(-1)
        expected_dim = int(ck["channel_base"]) * 30
        if feature_mean.shape != (expected_dim,) or feature_std.shape != (expected_dim,):
            raise ValueError(
                f"rich12 checkpoint stats must be ({expected_dim},), got "
                f"{feature_mean.shape} and {feature_std.shape}"
            )
        if not np.isfinite(feature_mean).all() or not np.isfinite(feature_std).all():
            raise ValueError(f"rich12 checkpoint stats contain non-finite values: {ckpt_path}")
        if np.any(feature_std <= 0):
            raise ValueError(f"rich12 checkpoint feature_std must be positive: {ckpt_path}")
        normalize_encoder_input(
            np.zeros((1, expected_dim), dtype=np.float32),
            feature_mean,
            feature_std,
        )
        cfg["feature_mean"], cfg["feature_std"] = feature_mean, feature_std
    return net, cfg


@torch.no_grad()
def map2latent_feature(net, cfg, rr, device):
    """(T,39) physical robot_repr -> (N, vae_length) learned-FGD latent samples, or None if
    T<32. %32 tail-truncate (EMAGE get_feature), then map2latent -> reshape(-1, vae_length)."""
    from echo_g.evaluation.robot_fgd.featurize import robot_repr_to_encoder_input

    rr = np.asarray(rr, dtype=np.float64)
    T = rr.shape[0]
    if T < 32:  # encoder pools /16 over >=32-frame windows
        return None
    if T % 32 != 0:  # EMAGE get_feature %32 tail-drop (no pad)
        rr = rr[: T - (T % 32)]
    x = robot_repr_to_encoder_input(rr, cfg["input_mode"], align="none")  # (T', 30*C)
    if cfg["feature_mean"] is not None:
        x = normalize_encoder_input(x, cfg["feature_mean"], cfg["feature_std"])
    xt = torch.from_numpy(x).float().unsqueeze(0).to(device)  # (1,T',30*C)
    lat = net.map2latent(xt)  # (1, T'/16, vae_length)
    return lat[0].cpu().numpy().reshape(-1, cfg["vae_length"])  # (N, vae_length)


# ============================================================================
# Metric classes (EMAGE mertic.py style: __init__ / update / compute|avg / reset)
# ============================================================================
class FGD(object):
    """Learned-FGD: accumulate per-clip encoder latents (pred & gt), Fréchet at compute().
    update() takes physical robot_repr[T,39] for pred and gt (operates on repr, not FK)."""

    def __init__(self, net, cfg, device):
        self.net, self.cfg, self.device = net, cfg, device
        self.reset()

    def reset(self):
        self.pred_lat, self.gt_lat = [], []

    def update(self, pr, gr):
        lp = map2latent_feature(self.net, self.cfg, pr, self.device)
        lg = map2latent_feature(self.net, self.cfg, gr, self.device)
        if lp is not None and lg is not None:  # both non-None -> paired (T>=32)
            self.pred_lat.append(lp)
            self.gt_lat.append(lg)

    @staticmethod
    def frechet_distance(
        pred_mat: np.ndarray | torch.Tensor,
        gt_mat: np.ndarray | torch.Tensor,
        eps: float = 1e-6,
    ) -> float:
        """EMAGE Fréchet kernel with jitter only inside sqrtm and unmodified covariance traces."""
        pred_mat = torch.as_tensor(pred_mat) if not torch.is_tensor(pred_mat) else pred_mat
        gt_mat = torch.as_tensor(gt_mat) if not torch.is_tensor(gt_mat) else gt_mat
        if pred_mat.shape[0] < 2 or gt_mat.shape[0] < 2:
            return float("nan")
        p = pred_mat.float().numpy()
        g = gt_mat.float().numpy()
        mu_p, mu_g = p.mean(axis=0), g.mean(axis=0)
        sigma_p = np.cov(p, rowvar=False)
        sigma_g = np.cov(g, rowvar=False)
        diff = mu_p - mu_g
        from scipy.linalg import sqrtm

        offset = np.eye(sigma_p.shape[0]) * eps
        covmean, _ = sqrtm((sigma_p + offset) @ (sigma_g + offset), disp=False)
        if np.iscomplexobj(covmean):
            covmean = covmean.real
        return float(diff @ diff + np.trace(sigma_p) + np.trace(sigma_g) - 2 * np.trace(covmean))

    def compute(self):
        if len(self.pred_lat) < 2:
            raise SystemExit(
                f"FGD needs >=2 clips with latents; got {len(self.pred_lat)} "
                f"(clips may be shorter than the encoder's 32-frame minimum)."
            )
        pred_mat = np.concatenate(self.pred_lat, axis=0)  # (ΣN, vae_length)
        gt_mat = np.concatenate(self.gt_lat, axis=0)
        return self.frechet_distance(pred_mat, gt_mat), pred_mat.shape[0], gt_mat.shape[0]


class DivCrossClip(object):
    """CROSS-clip diversity (ours): accumulate one pooled_joint_stats feature per clip (pred & gt),
    at compute() report the EMAGE L1div (MAD about the batch mean) for pred and gt separately."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.pred_feats, self.gt_feats = [], []

    def update(self, pj, gj):
        self.pred_feats.append(pooled_joint_stats_feature(pj))
        self.gt_feats.append(pooled_joint_stats_feature(gj))

    @staticmethod
    def _l1div(feats):
        F = np.stack(feats).astype(np.float64)  # (N, D)
        if F.shape[0] < 2:
            return float("nan")
        return float(np.abs(F - F.mean(axis=0)).sum() / F.shape[0])  # Σ|F-mean|/N (EMAGE L1div)

    def compute(self):
        return self._l1div(self.pred_feats), self._l1div(self.gt_feats)


class Div(object):
    """EMAGE L1div in G1 position space; core methods intentionally mirror mertic.py."""

    def __init__(self):
        self.counter = 0
        self.sum = 0

    def compute(self, results):
        self.counter += results.shape[0]
        mean = np.mean(results, axis=0)
        sum_l1 = np.sum(np.abs(results - mean), axis=None)
        self.sum += float(sum_l1)

    def avg(self):
        if self.counter == 0:
            return 0.0
        return float(self.sum / self.counter)

    def reset(self):
        self.counter = 0
        self.sum = 0.0


def gt_closeness(prediction: float | None, reference: float | None) -> dict[str, float | None]:
    """Return signed/absolute/relative distance to GT; absolute distance is the primary score."""
    if prediction is None or reference is None:
        return {"signed_gap": None, "absolute_gap": None, "relative_gap": None}
    prediction_value = float(prediction)
    reference_value = float(reference)
    if not np.isfinite(prediction_value) or not np.isfinite(reference_value):
        return {"signed_gap": None, "absolute_gap": None, "relative_gap": None}
    signed_gap = prediction_value - reference_value
    absolute_gap = abs(signed_gap)
    relative_gap = absolute_gap / abs(reference_value) if reference_value != 0.0 else None
    return {
        "signed_gap": float(signed_gap),
        "absolute_gap": float(absolute_gap),
        "relative_gap": None if relative_gap is None else float(relative_gap),
    }


class Multimodality(object):
    """Audio2Gestures-style multimodality over repeated samples of the same audio.

    Each clip contributes the mean unordered-pair L1 distance per frame in pose-only G1 FK
    position space. Clips are averaged equally. ``avg_eq16_literal`` also exposes the value from
    the denominator printed in Audio2Gestures Eq. 16, which differs from the true pair count.
    """

    def __init__(self, expected_runs: int = 20) -> None:
        if expected_runs < 2:
            raise ValueError(f"Multimodality needs at least 2 runs, got {expected_runs}")
        self.expected_runs = expected_runs
        self.pair_count = expected_runs * (expected_runs - 1) // 2
        self.eq16_denominator = expected_runs * ((expected_runs + 1) // 2)
        self.reset()

    def update(self, sampled_positions: list[np.ndarray]) -> tuple[float, float]:
        if len(sampled_positions) != self.expected_runs:
            raise ValueError(
                f"Multimodality expected {self.expected_runs} runs, got {len(sampled_positions)}"
            )

        samples = [np.asarray(sample, dtype=np.float64) for sample in sampled_positions]
        reference_shape = samples[0].shape
        if len(reference_shape) != 3 or reference_shape[0] <= 0 or reference_shape[-1] != 3:
            raise ValueError(
                f"Multimodality samples must have nonempty (T, J, 3) shape, got {reference_shape}"
            )
        for run_index, sample in enumerate(samples):
            if sample.shape != reference_shape:
                raise ValueError(
                    "Multimodality requires exactly equal lengths/shapes across runs; "
                    f"run_000={reference_shape}, run_{run_index:03d}={sample.shape}"
                )
            if not np.isfinite(sample).all():
                raise ValueError(
                    f"Multimodality run_{run_index:03d} contains non-finite FK positions"
                )

        # For each scalar coordinate, sum_{a<b}|x_a-x_b| exactly after sorting:
        # sum_i (2*i-N+1)*x_(i). This avoids materializing all N*(N-1)/2 pair differences.
        flattened = np.stack(samples, axis=0).reshape(self.expected_runs, -1)
        ordered = np.sort(flattened, axis=0)
        coefficients = (
            2.0 * np.arange(self.expected_runs, dtype=np.float64) - self.expected_runs + 1.0
        ).reshape(-1, 1)
        pair_l1_sum = float(np.sum(ordered * coefficients, dtype=np.float64))
        roundoff = (
            np.finfo(np.float64).eps
            * max(1.0, float(np.sum(np.abs(ordered * coefficients), dtype=np.float64)))
            * 32.0
        )
        if pair_l1_sum < -roundoff:
            raise RuntimeError(f"Multimodality pairwise L1 sum is negative: {pair_l1_sum}")
        pair_l1_sum = max(0.0, pair_l1_sum)

        frames = reference_shape[0]
        pair_mean = pair_l1_sum / (self.pair_count * frames)
        eq16_literal = pair_l1_sum / (self.eq16_denominator * frames)
        self.sum += pair_mean
        self.eq16_sum += eq16_literal
        self.counter += 1
        self.frames += frames
        return float(pair_mean), float(eq16_literal)

    def avg(self) -> float:
        return float(self.sum / self.counter) if self.counter else float("nan")

    def avg_eq16_literal(self) -> float:
        return float(self.eq16_sum / self.counter) if self.counter else float("nan")

    def reset(self) -> None:
        self.counter = 0
        self.frames = 0
        self.sum = 0.0
        self.eq16_sum = 0.0


class SRGRMass(object):
    """Semantic Relevant Gesture Recall normalized by actual evaluated semantic mass.

    This preserves EMAGE's per-frame/per-body Euclidean hit definition while replacing its
    split-specific ``1 / 0.165`` scale with the exact maximum attainable weighted hit mass.
    Consequently a perfect prediction is 1 on every nonzero-semantic evaluation set.
    """

    def __init__(
        self, threshold: float = 0.1, joints: int = NUM_G1_BODIES, joint_dim: int = 3
    ) -> None:
        if threshold <= 0:
            raise ValueError(f"SRGR threshold must be positive, got {threshold}")
        if joints <= 0 or joint_dim <= 0:
            raise ValueError(f"SRGR joints/joint_dim must be positive, got {joints}/{joint_dim}")
        self.threshold = threshold
        self.pose_dimes = joints
        self.joint_dim = joint_dim
        self.reset()

    def run(
        self, results: np.ndarray, targets: np.ndarray, semantic: np.ndarray, verbose: bool = False
    ) -> float:
        results = np.asarray(results).reshape(-1, self.pose_dimes, self.joint_dim)
        targets = np.asarray(targets).reshape(-1, self.pose_dimes, self.joint_dim)
        semantic = np.asarray(semantic, dtype=np.float64).reshape(-1)
        if results.shape != targets.shape:
            raise ValueError(f"SRGR pred/target shape mismatch: {results.shape} vs {targets.shape}")
        if semantic.shape[0] != results.shape[0]:
            raise ValueError(
                f"SRGR semantic length {semantic.shape[0]} != motion length {results.shape[0]}"
            )
        if not np.isfinite(semantic).all():
            raise ValueError("SRGR semantic weights must be finite")
        if np.any(semantic < 0):
            raise ValueError("SRGR semantic weights must be non-negative")

        diff = np.linalg.norm(results - targets, axis=2)  # T, J
        if not np.isfinite(diff).all():
            raise ValueError("SRGR FK distances must be finite")
        if verbose:
            print(diff)
        success = (diff < self.threshold).astype(np.float64)
        semantic_mass = float(np.sum(semantic, dtype=np.float64))
        numerator = float(np.sum(success * semantic[:, None], dtype=np.float64))
        denominator = float(self.pose_dimes * semantic_mass)
        if numerator > denominator:
            roundoff = np.finfo(np.float64).eps * max(1.0, denominator) * 32
            if numerator - denominator > roundoff:
                raise RuntimeError(f"SRGR numerator {numerator} exceeds maximum mass {denominator}")
            numerator = denominator

        self.counter += success.shape[0]
        self.semantic_mass += semantic_mass
        self.numerator += numerator
        self.denominator += denominator
        return float(numerator / denominator) if denominator > 0 else float("nan")

    def avg(self) -> float:
        if self.denominator <= 0:
            return float("nan")
        return float(self.numerator / self.denominator)

    def reset(self) -> None:
        self.counter = 0
        self.semantic_mass = 0.0
        self.numerator = 0.0
        self.denominator = 0.0


class BeatAlign(object):
    """BeatAlign (EMAGE-official BC): per pred clip, per-body motion beats (speed local-minima ∩
    frames where mmae-normalized speed>threshold) -> GAHR to nearest audio onset (Gaussian σ),
    averaged over upper bodies (no-beat body scores 0).

    Cross-clip aggregation is FRAME-WEIGHTED (EMAGE BC.compute / GestureLSM align: sum += BA*len,
    counter += len -> Σ(BA_i·len_i)/Σlen_i), so long clips weigh more, matching both official
    implementations. NOTE: we do NOT do their head/tail 2s trim, so weight = full frame count T
    (their weight is the trimmed length n-2*mask). summary() reports the frame-weighted mean plus
    the plain per-clip stats for reference."""

    def __init__(
        self,
        fps,
        sigma=0.3,
        order=7,
        bc_threshold=0.10,
        upper_bodies=None,
        direction="audio_to_motion",
    ):
        if direction not in {"motion_to_audio", "audio_to_motion"}:
            raise ValueError(f"Unknown BA direction: {direction}")
        self.fps, self.sigma, self.order, self.bc_threshold = fps, sigma, order, bc_threshold
        self.direction = direction
        self.upper_bodies = UPPER_G1_BODIES if upper_bodies is None else upper_bodies
        self._mmae = _g1_mean_vel()  # (30,) or None
        self.reset()

    def reset(self):
        self.scores = []  # per clip: BA scalar (may be nan)
        self.weights = []  # per clip: frame count T (weight)

    def update(self, pj, wav_path):
        self.scores.append(self._beat_consistency(pj, wav_path))
        self.weights.append(int(np.asarray(pj).shape[0]))  # frame count = weight

    def _beat_consistency(self, joints, wav_path):
        if not os.path.exists(wav_path):
            raise FileNotFoundError(f"BA audio is missing: {wav_path}")
        try:
            import librosa
            import scipy.signal
        except ImportError as exc:
            raise RuntimeError(
                "BA requires librosa and scipy; refusing a degraded evaluation"
            ) from exc
        J = np.asarray(joints, dtype=np.float64)  # (T,30,3)
        if J.shape[0] < 3:
            return float("nan")
        spd = body_speed_central(J, self.fps)  # (T,30) m/s, EMAGE central diff
        # EMAGE BC.load_audio convention: librosa.load(sr=16000) (force-resample + mono) then
        # onset_detect(sr=16000, hop_length=512). Matches PantoMatrix exactly
        # (not sf.read native sr).
        y, _sr = librosa.load(wav_path, sr=16000)
        # Keep [0, T/fps) only, matching the common pred/GT motion prefix. Unlike the EMAGE paper
        # protocol, Hermes intentionally does not remove two seconds from either boundary.
        y = y[: int(J.shape[0] / float(self.fps) * 16000)]
        a_beats = librosa.onset.onset_detect(y=y, sr=16000, hop_length=512, units="time")
        if len(a_beats) == 0:
            return float("nan")
        a = np.asarray(a_beats, dtype=np.float64)  # audio onset times (s)

        def gahr(motion_beats_t):
            pairwise = np.abs(motion_beats_t[:, None] - a[None, :])
            if self.direction == "audio_to_motion":
                d = pairwise.min(axis=0)  # EMAGE: one term per audio onset
            else:
                d = pairwise.min(axis=1)  # historical Hermes: one term per motion beat
            return float(np.mean(np.exp(-(d**2) / (2.0 * self.sigma**2))))

        per_body = []
        for b in self.upper_bodies:
            v = spd[:, b]
            norm = (
                v / (self._mmae[b] + 1e-8)
                if (self._mmae is not None and self._mmae[b] > 1e-8)
                else v
            )
            minima = scipy.signal.argrelextrema(v, np.less, order=self.order)[
                0
            ]  # speed local minima
            moving = np.where(norm > self.bc_threshold)[0]  # EMAGE vel_mask
            beats = np.intersect1d(minima, moving).astype(np.float64)  # minima ∩ moving
            per_body.append(gahr(beats / self.fps) if beats.size else 0.0)  # no beats -> 0
        if len(per_body) == 0:
            return float("nan")
        return float(np.sum(per_body) / len(self.upper_bodies))  # EMAGE: /len(upper_body)

    def summary(self):
        d = summ(self.scores)  # plain per-clip stats (nan dropped)
        # Frame-weighted mean = Σ(BA_i·len_i)/Σlen_i over non-nan clips
        # (EMAGE/GestureLSM convention).
        sw = ns = 0.0
        for v, w in zip(self.scores, self.weights):
            if v == v:  # drop nan (audio missing / clip too short)
                sw += v * w
                ns += w
        d["mean_unweighted"] = d["mean"]  # keep the old clip-equal mean for reference
        d["mean"] = float(sw / ns) if ns else None  # headline BA is now frame-weighted
        return d


class Jerk(object):
    """Mean 3rd finite-difference magnitude * fps^3 (m/s^3), per clip, for pred and GT."""

    def __init__(self, fps: float) -> None:
        self.fps = float(fps)
        self.reset()

    def reset(self) -> None:
        self.gen, self.ref = [], []
        self.gen_length_weights, self.ref_length_weights = [], []

    @staticmethod
    def _jerk(body_pos_w: np.ndarray, fps: float) -> float:
        p = np.asarray(body_pos_w, dtype=np.float64)
        if p.ndim != 3 or p.shape[-1] != 3:
            raise ValueError(f"body_pos_w must be (T,J,3), got {p.shape}")
        if p.shape[0] < 4:
            return float("nan")
        jerk = p[3:] - 3.0 * p[2:-1] + 3.0 * p[1:-2] - p[:-3]  # (T-3,J,3)
        return float((np.linalg.norm(jerk, axis=-1) * (float(fps) ** 3)).mean())

    def update(self, pj: np.ndarray, gj: np.ndarray) -> None:
        self.gen.append(self._jerk(pj, self.fps))
        self.ref.append(self._jerk(gj, self.fps))
        # A T-frame sequence has T-3 valid third-difference samples. All evaluated
        # G1 sequences contain the same 30 bodies, so the body-count factor cancels.
        self.gen_length_weights.append(max(int(np.asarray(pj).shape[0]) - 3, 0))
        self.ref_length_weights.append(max(int(np.asarray(gj).shape[0]) - 3, 0))

    def summary(self) -> tuple[dict, dict]:
        return summ(self.gen), summ(self.ref)

    def summary_length_weighted(self) -> tuple[dict, dict]:
        """Summarize with each clip weighted by its number of valid jerk samples (T-3)."""
        return (
            weighted_summ(self.gen, self.gen_length_weights),
            weighted_summ(self.ref, self.ref_length_weights),
        )


class FootMetrics(object):
    """foot_ground_error + contact_sliding_speed (OPTIONAL), from grounded-stance FK. update()
    takes the pred robot_repr[T,39] (needs the absolute-z grounded anchor, unlike pose-only FK)."""

    def __init__(
        self,
        fps: float,
        contact_height_threshold: float = 0.12,
        contact_penetration_tolerance: float = 0.02,
    ) -> None:
        self.fps = float(fps)
        self.contact_height_threshold = float(contact_height_threshold)
        self.contact_penetration_tolerance = float(contact_penetration_tolerance)
        self.reset()

    def reset(self) -> None:
        self.fge, self.csl = [], []

    def update(self, pr: torch.Tensor) -> None:
        _, sole_points, sole_radii, sole_foot_ids = fk_grounded_world(pr, self.fps)
        sole_points_np = sole_points.cpu().numpy()
        self.fge.append(self._foot_ground_error(sole_points_np, sole_radii))
        if sole_points_np.shape[0] > 1:
            self.csl.append(
                self._contact_sliding_speed(
                    sole_points_np,
                    sole_radii,
                    sole_foot_ids,
                )
            )

    @staticmethod
    def _masked_mean(value: np.ndarray, mask: np.ndarray) -> float:
        mask_float = mask.astype(value.dtype, copy=False)
        return float((value * mask_float).sum() / max(mask_float.sum(), 1.0))

    @staticmethod
    def _foot_ground_error(sole_points: np.ndarray, sole_radii: np.ndarray) -> float:
        sole_bottom = sole_points[..., 2] - sole_radii.reshape(1, -1)
        lowest_sole_bottom = sole_bottom.min(axis=-1)
        return float(np.abs(lowest_sole_bottom).mean())

    def _contact_sliding_speed(
        self,
        sole_points: np.ndarray,
        sole_radii: np.ndarray,
        sole_foot_ids: np.ndarray,
    ) -> float:
        sole_bottom = sole_points[..., 2] - sole_radii.reshape(1, -1)
        point_speed = np.linalg.norm(np.diff(sole_points[..., :2], axis=0), axis=-1) * self.fps
        point_contact = (sole_bottom >= -self.contact_penetration_tolerance) & (
            sole_bottom <= self.contact_height_threshold
        )

        speeds, masks = [], []
        for foot_id in np.unique(sole_foot_ids):
            foot_mask = sole_foot_ids == foot_id
            foot_contact = point_contact[..., foot_mask].any(axis=-1)
            foot_interval = foot_contact[1:] & foot_contact[:-1]
            foot_speed = point_speed[..., foot_mask].max(axis=-1)
            speeds.append(foot_speed)
            masks.append(foot_interval)

        return self._masked_mean(
            np.stack(speeds, axis=-1),
            np.stack(masks, axis=-1),
        )

    def summary(self) -> tuple[dict, dict]:
        return summ(self.fge), summ(self.csl)


# G1 per-body mean velocity (mmae, EMAGE's mean_vel role), computed from z_robot_beat2_g1 GT.
_G1_MEAN_VEL_PATH = ""
_G1_MEAN_VEL = None


def _g1_mean_vel() -> np.ndarray:
    global _G1_MEAN_VEL
    if _G1_MEAN_VEL is None:
        try:
            _G1_MEAN_VEL = np.load(_G1_MEAN_VEL_PATH).astype(np.float64)  # (30,)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to load --mmae-file {_G1_MEAN_VEL_PATH!r}; "
                "EMAGE-aligned BA requires per-body velocity normalization."
            ) from exc
        if _G1_MEAN_VEL.shape != (NUM_G1_BODIES,) or not np.isfinite(_G1_MEAN_VEL).all():
            raise ValueError(
                f"Invalid --mmae-file {_G1_MEAN_VEL_PATH!r}: expected finite shape "
                f"({NUM_G1_BODIES},), got {_G1_MEAN_VEL.shape}."
            )
    return _G1_MEAN_VEL


def body_speed_central(J, fps):
    """Per-body speed via EMAGE BC.load_motion central difference (NOT np.diff forward diff).
    J:(T,B,3) -> (T,B) m/s, same length T. Endpoints one-sided, interior (x[i+1]-x[i-1])/(2dt).
    Byte-matches PantoMatrix mertic.py::BC.load_motion. BA + its mmae MUST both use this."""
    J = np.asarray(J, dtype=np.float64)
    dt = 1.0 / float(fps)
    if J.shape[0] < 3:
        return (
            np.linalg.norm(np.diff(J, axis=0), axis=0) * float(fps)
            if J.shape[0] == 2
            else np.zeros((J.shape[0], J.shape[1]), dtype=np.float64)
        )
    init = (J[1:2] - J[0:1]) / dt  # forward diff  @ frame 0
    mid = (J[2:] - J[0:-2]) / (2.0 * dt)  # central diff  @ interior
    fin = (J[-1:] - J[-2:-1]) / dt  # backward diff @ last frame
    vel = np.concatenate([init, mid, fin], axis=0)  # (T,B,3)
    return np.linalg.norm(vel, axis=-1)  # (T,B) m/s


# ============================================================================
# IO helpers  [stateless]
# ============================================================================
def load_repr(pt_path: str, expected_fps: float) -> torch.Tensor | None:
    """Load a physical robot_repr[T,39] payload and enforce any available metadata."""
    d = torch.load(pt_path, map_location="cpu", weights_only=False)
    t = None
    if isinstance(d, dict):
        units = d.get("representation_units")
        if units is not None and str(units).strip().lower() != "physical":
            raise ValueError(
                f"{pt_path}: evaluator accepts physical robot_repr only, got "
                f"representation_units={units!r}"
            )
        stored_fps = d.get("fps")
        if stored_fps is not None and float(stored_fps) != float(expected_fps):
            raise ValueError(
                f"{pt_path}: stored fps={float(stored_fps)} != evaluator fps={expected_fps}"
            )
        for k in _REPR_KEYS:
            if k in d and torch.is_tensor(d[k]):
                t = d[k]
                break
        if t is None:
            for v in d.values():
                if torch.is_tensor(v) and v.ndim == 2 and v.shape[-1] == ROBOT_MOTION_DIM:
                    t = v
                    break
    elif torch.is_tensor(d):
        t = d
    if t is None:
        return None
    t = t.float()
    if t.ndim == 3 and t.shape[0] == 1:
        t = t[0]
    if t.ndim != 2 or t.shape[-1] != ROBOT_MOTION_DIM:
        logger.warning("%s: unexpected repr shape %s, skip", pt_path, tuple(t.shape))
        return None
    if not torch.isfinite(t).all():
        raise ValueError(f"{pt_path}: robot_repr contains non-finite values")
    # A normalized tensor cannot be identified reliably from numeric ranges alone. Producers must
    # de-normalize before saving and should set representation_units='physical' in the payload.
    return t


def wav_of(wav_dir: str, stem: str) -> str:
    c1 = f"{wav_dir}/{stem}/audio.wav"
    return c1 if os.path.exists(c1) else f"{wav_dir}/{stem}.wav"


def load_semantic(
    sem_dir: str,
    stem: str,
    target_length: int,
    target_fps: float,
    semantic_fps: float,
    semantic_protocol: str = _SRGR_SEMANTIC_PROTOCOL,
) -> np.ndarray | None:
    """Load a validated first-match semantic prefix without padding or resampling."""
    path = os.path.join(sem_dir, f"{stem}.pt")
    if not os.path.isfile(path):
        return None
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: semantic payload must be a metadata dictionary")
    if payload.get("protocol") != semantic_protocol:
        raise ValueError(
            f"{path}: semantic protocol {payload.get('protocol')!r} != {semantic_protocol!r}"
        )
    if payload.get("stem") != stem:
        raise ValueError(f"{path}: payload stem {payload.get('stem')!r} != {stem!r}")
    payload_fps = float(payload.get("fps", float("nan")))
    if not np.isclose(payload_fps, semantic_fps, rtol=0.0, atol=1e-9):
        raise ValueError(f"{path}: payload fps {payload_fps} != --semantic-fps {semantic_fps}")
    if not np.isclose(payload_fps, target_fps, rtol=0.0, atol=1e-9):
        raise ValueError(
            f"{path}: semantic fps {payload_fps} != evaluated motion fps {target_fps}; "
            "resampling is forbidden for the canonical SRGR cache"
        )

    semantic = payload.get("sem")
    if semantic is None:
        raise ValueError(f"{path}: semantic payload has no 'sem' tensor")
    semantic_np = np.asarray(torch.as_tensor(semantic).float().reshape(-1), dtype=np.float64)
    if semantic_np.size == 0 or not np.isfinite(semantic_np).all():
        raise ValueError(f"{path}: semantic scores are empty or non-finite")
    if np.any(semantic_np < 0) or np.any(semantic_np > 1):
        raise ValueError(f"{path}: semantic scores must lie in [0, 1]")
    payload_frames = int(payload.get("num_frames", -1))
    if payload_frames != semantic_np.size:
        raise ValueError(
            f"{path}: payload num_frames {payload_frames} != semantic length {semantic_np.size}"
        )
    if target_length > semantic_np.size:
        raise ValueError(
            f"{path}: evaluated motion length {target_length} exceeds semantic length "
            f"{semantic_np.size}; final-value repetition is forbidden"
        )
    return semantic_np[:target_length]


def summ(vals: list) -> dict:
    """Summary stats over a list, dropping NaNs."""
    a = np.asarray([v for v in vals if v == v], dtype=np.float64)  # drop nan
    if a.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def weighted_summ(vals: list, weights: list[int]) -> dict:
    """Weighted summary, dropping NaNs and non-positive weights."""
    if len(vals) != len(weights):
        raise ValueError(f"values/weights length mismatch: {len(vals)} != {len(weights)}")
    valid = [(float(v), int(w)) for v, w in zip(vals, weights) if v == v and int(w) > 0]
    if not valid:
        return {
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "n": 0,
            "weight_sum": 0,
        }
    values = np.asarray([value for value, _ in valid], dtype=np.float64)
    sample_weights = np.asarray([weight for _, weight in valid], dtype=np.float64)
    mean = float(np.average(values, weights=sample_weights))
    variance = float(np.average((values - mean) ** 2, weights=sample_weights))
    return {
        "mean": mean,
        "std": float(np.sqrt(variance)),
        "min": float(values.min()),
        "max": float(values.max()),
        "n": int(values.size),
        "weight_sum": int(sample_weights.sum()),
    }


# ============================================================================
# main: argparse -> stats -> stems -> per-clip fan-out to evaluators -> gather
# ============================================================================
def build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pred-dir", required=True, help="dir of predicted robot_repr[T,39] .pt")
    ap.add_argument("--ref-dir", required=True, help="dir of reference (GT) robot_repr[T,39] .pt")
    ap.add_argument(
        "--multimodality-root",
        default="",
        help="optional root containing independent samples as run_000/<stem>.pt ...; "
        "empty explicitly skips Multimodality",
    )
    ap.add_argument(
        "--multimodality-runs",
        type=int,
        default=20,
        help="number of independent samples per audio (Audio2Gestures protocol: 20)",
    )
    ap.add_argument(
        "--require-all-multimodality",
        action="store_true",
        help="fail unless every evaluated stem exists in every multimodality run directory",
    )
    ap.add_argument(
        "--wav-dir",
        required=True,
        help="audio root: <wav-dir>/<stem>/audio.wav or <wav-dir>/<stem>.wav",
    )
    ap.add_argument(
        "--sem-dir",
        required=True,
        help="per-clip semantic relevance .pt root: <sem-dir>/<stem>.pt with key 'sem'",
    )
    ap.add_argument(
        "--semantic-fps",
        type=float,
        default=30.0,
        help="expected frame rate stored in the canonical semantic cache",
    )
    ap.add_argument(
        "--semantic-protocol",
        default=_SRGR_SEMANTIC_PROTOCOL,
        help="expected protocol string stored in every canonical semantic payload",
    )
    ap.add_argument(
        "--srgr-threshold",
        type=float,
        default=0.1,
        help="per-G1-body Euclidean FK-position success threshold in meters",
    )
    ap.add_argument(
        "--expected-srgr-frames",
        type=int,
        default=0,
        help="optional formal-run audit: require exactly this many evaluated frames",
    )
    ap.add_argument(
        "--val-split", default="", help="stem list; empty -> all stems present in BOTH dirs"
    )
    ap.add_argument(
        "--exclude-stems-file",
        default="",
        help="optional stem list to exclude from --val-split for every method",
    )
    ap.add_argument(
        "--require-all-stems",
        action="store_true",
        help="fail if any requested stem lacks pred/ref or contains an invalid short clip",
    )
    ap.add_argument(
        "--require-equal-lengths",
        action="store_true",
        help="fail instead of common-prefix cropping when a pred/GT pair differs in length",
    )
    ap.add_argument("--fps", type=float, default=30.0)
    ap.add_argument(
        "--limit",
        type=int,
        default=0,
        help="cap #clips (<=0 -> ALL stems, full test set). Default 0 = full.",
    )
    ap.add_argument("--bc-sigma", type=float, default=0.3, help="BeatAlign gaussian sigma (s)")
    ap.add_argument("--bc-order", type=int, default=7, help="BeatAlign local-extrema window")
    ap.add_argument(
        "--mmae-file",
        required=True,
        help="required G1 per-body mean velocity .npy for EMAGE-aligned BA; "
        "missing or invalid files are a hard error",
    )
    ap.add_argument(
        "--ba-direction",
        choices=["motion_to_audio", "audio_to_motion"],
        default="audio_to_motion",
        help="directed BA matching. audio_to_motion is the EMAGE-compatible default; "
        "motion_to_audio preserves historical Hermes.",
    )
    ap.add_argument(
        "--enable-foot-metrics",
        action="store_true",
        help="also compute foot_ground_error + contact_sliding_speed (grounded-stance anchor)",
    )
    ap.add_argument("--contact-height-threshold", type=float, default=0.12)
    ap.add_argument("--contact-penetration-tolerance", type=float, default=0.02)
    ap.add_argument(
        "--g1-ae-ckpt",
        required=True,
        help="REQUIRED trained G1 skeleton-conv AE (.bin). Use the release w192 sincos encoder "
        "(docs/BENCHMARK.md). FGD = Frechet on its map2latent latent; comparable "
        "ONLY within one ckpt. Empty/missing -> hard error (no pooled fallback FGD).",
    )
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", default="outputs/omg_robot_bench.json")
    ap.add_argument(
        "--dump-fk-dir",
        default="",
        help="debug: save each clip's pose-only FK body positions (T,30,3) as "
        "<dir>/<stem>_{pred,gt}.npy (does NOT affect any metric)",
    )
    ap.add_argument("--dump-fk-limit", type=int, default=5, help="max clips to dump FK for")
    return ap


def validate_physical_evaluation_contract(args: argparse.Namespace) -> None:
    if float(args.fps) != _REQUIRED_EVAL_FPS:
        raise ValueError(
            f"eval_g1_motion_cls.py is fixed to {_REQUIRED_EVAL_FPS:g} FPS; got {args.fps}"
        )


def read_stem_file(path: str) -> list[str]:
    with open(path, encoding="utf-8") as handle:
        return [
            line.strip() for line in handle if line.strip() and not line.lstrip().startswith("#")
        ]


def resolve_multimodality_run_dirs(root: str, expected_runs: int) -> list[Path]:
    """Resolve the deterministic ``run_000`` ... ``run_{N-1}`` sampling layout."""
    if expected_runs < 2:
        raise ValueError(f"--multimodality-runs must be at least 2, got {expected_runs}")
    if not root:
        return []
    root_path = Path(root)
    if not root_path.is_dir():
        raise ValueError(f"--multimodality-root is not a directory: {root}")
    run_dirs = [root_path / f"run_{index:03d}" for index in range(expected_runs)]
    missing = [str(path) for path in run_dirs if not path.is_dir()]
    if missing:
        raise ValueError(
            f"multimodality layout needs {expected_runs} run directories; "
            f"missing={len(missing)} {missing[:5]}"
        )
    return run_dirs


def crop_to_common_length(
    pred: torch.Tensor,
    gt: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply EMAGE's paired temporal alignment before any metric-specific processing."""
    common_length = min(pred.shape[0], gt.shape[0])
    return pred[:common_length], gt[:common_length]


def select_stems(args: argparse.Namespace) -> list[str]:
    """Apply one shared inclusion/exclusion set, validate coverage, then apply --limit."""
    if args.val_split:
        if not os.path.isfile(args.val_split):
            raise SystemExit(f"--val-split does not exist: {args.val_split}")
        want = read_stem_file(args.val_split)
    else:
        want = sorted(p.stem for p in Path(args.ref_dir).glob("*.pt"))

    if args.exclude_stems_file:
        if not os.path.isfile(args.exclude_stems_file):
            raise SystemExit(f"--exclude-stems-file does not exist: {args.exclude_stems_file}")
        excluded = set(read_stem_file(args.exclude_stems_file))
        want = [stem for stem in want if stem not in excluded]
        logger.info(
            "excluded %d configured stems; %d requested stems remain", len(excluded), len(want)
        )

    duplicate_count = len(want) - len(set(want))
    if duplicate_count and args.require_all_stems:
        raise SystemExit(f"stem selection contains {duplicate_count} duplicate entries")
    want = list(dict.fromkeys(want))
    missing_pred = [stem for stem in want if not os.path.isfile(f"{args.pred_dir}/{stem}.pt")]
    missing_ref = [stem for stem in want if not os.path.isfile(f"{args.ref_dir}/{stem}.pt")]
    if args.require_all_stems and (missing_pred or missing_ref):
        raise SystemExit(
            "strict stem coverage failed: "
            f"requested={len(want)} missing_pred={len(missing_pred)} {missing_pred[:5]} "
            f"missing_ref={len(missing_ref)} {missing_ref[:5]}"
        )
    missing_pred_set = set(missing_pred)
    missing_ref_set = set(missing_ref)
    stems = [stem for stem in want if stem not in missing_pred_set and stem not in missing_ref_set]
    if args.limit and args.limit > 0:  # limit<=0 -> evaluate ALL stems (full test set)
        stems = stems[: args.limit]
    return stems


def main() -> None:
    global _G1_MEAN_VEL_PATH, _G1_MEAN_VEL
    args = build_argparser().parse_args()
    validate_physical_evaluation_contract(args)
    _G1_MEAN_VEL_PATH = args.mmae_file
    _G1_MEAN_VEL = None
    _g1_mean_vel()
    if not os.path.isdir(args.sem_dir):
        raise SystemExit(f"--sem-dir required for SRGR and does not exist: {args.sem_dir}")
    if args.semantic_fps <= 0:
        raise ValueError(f"--semantic-fps must be positive, got {args.semantic_fps}")
    if not args.semantic_protocol.strip():
        raise ValueError("--semantic-protocol must be non-empty")
    if args.srgr_threshold <= 0:
        raise ValueError(f"--srgr-threshold must be positive, got {args.srgr_threshold}")
    if args.expected_srgr_frames < 0:
        raise ValueError(
            f"--expected-srgr-frames must be non-negative, got {args.expected_srgr_frames}"
        )
    multimodality_run_dirs = resolve_multimodality_run_dirs(
        args.multimodality_root, args.multimodality_runs
    )
    if args.require_all_multimodality and not multimodality_run_dirs:
        raise ValueError("--require-all-multimodality requires a nonempty --multimodality-root")

    # learned-FGD encoder is REQUIRED (no pooled-stats fallback FGD).
    if not (args.g1_ae_ckpt and os.path.exists(args.g1_ae_ckpt)):
        raise SystemExit(
            f"--g1-ae-ckpt required (FGD needs a learned G1 encoder); "
            f"got {args.g1_ae_ckpt!r} (missing). See docs/BENCHMARK.md."
        )
    ae_dev = "cuda" if torch.cuda.is_available() else "cpu"
    net, cfg = load_g1_ae(args.g1_ae_ckpt, ae_dev, args.fps)
    checkpoint_fps_label = (
        "not recorded" if cfg["checkpoint_fps"] is None else f"{cfg['checkpoint_fps']:g}"
    )
    logger.info(
        "learned FGD via %s (mode=%s vae_length=%d checkpoint_fps=%s)",
        args.g1_ae_ckpt,
        cfg["input_mode"],
        cfg["vae_length"],
        checkpoint_fps_label,
    )

    stems = select_stems(args)
    if len(stems) < 2:
        print(f"Need >=2 stems with both pred and ref; found {len(stems)}.")
        return
    missing_semantic = [s for s in stems if not os.path.isfile(f"{args.sem_dir}/{s}.pt")]
    if missing_semantic and args.require_all_stems:
        raise SystemExit(
            f"strict semantic coverage failed: requested={len(stems)} "
            f"missing_semantic={len(missing_semantic)} {missing_semantic[:5]}"
        )
    if missing_semantic:
        logger.warning(
            "SRGR semantic coverage misses %d/%d clips", len(missing_semantic), len(stems)
        )

    multimodality_missing: dict[str, list[str]] = {}
    if multimodality_run_dirs:
        for stem in stems:
            missing_runs = [
                run_dir.name
                for run_dir in multimodality_run_dirs
                if not (run_dir / f"{stem}.pt").is_file()
            ]
            if missing_runs:
                multimodality_missing[stem] = missing_runs
        if multimodality_missing and args.require_all_multimodality:
            examples = [
                f"{stem}:{runs[:3]}" for stem, runs in list(multimodality_missing.items())[:5]
            ]
            raise SystemExit(
                "strict Multimodality coverage failed: "
                f"incomplete_stems={len(multimodality_missing)}/{len(stems)} "
                f"examples={examples}"
            )
        if multimodality_missing:
            logger.warning(
                "Multimodality coverage misses one or more runs for %d/%d stems; "
                "those stems will be skipped",
                len(multimodality_missing),
                len(stems),
            )
        logger.info(
            "Multimodality enabled: root=%s runs=%d complete_stems=%d/%d",
            args.multimodality_root,
            args.multimodality_runs,
            len(stems) - len(multimodality_missing),
            len(stems),
        )
    else:
        logger.info(
            "Multimodality skipped: no --multimodality-root (a single prediction is insufficient)"
        )

    # ---- instantiate the metric evaluators (EMAGE-style) ----
    fgd = FGD(net, cfg, ae_dev)
    # Disabled: cross-clip pooled-feature Div is a research diagnostic, not EMAGE L1div.
    # div_cross = DivCrossClip()
    div = Div()
    div_gt = Div()
    multimodality = (
        Multimodality(expected_runs=args.multimodality_runs) if multimodality_run_dirs else None
    )
    srgr = SRGRMass(threshold=args.srgr_threshold)
    beat = BeatAlign(
        args.fps, sigma=args.bc_sigma, order=args.bc_order, direction=args.ba_direction
    )
    beat_gt = BeatAlign(
        args.fps, sigma=args.bc_sigma, order=args.bc_order, direction=args.ba_direction
    )
    jerk = Jerk(args.fps)
    foot = (
        FootMetrics(args.fps, args.contact_height_threshold, args.contact_penetration_tolerance)
        if args.enable_foot_metrics
        else None
    )

    # ---- per-clip loop: load -> FK -> fan out to every evaluator's .update() ----
    n_done = 0
    n_srgr = 0
    n_multimodality = 0
    for s in stems:
        pr = load_repr(f"{args.pred_dir}/{s}.pt", args.fps)
        gr = load_repr(f"{args.ref_dir}/{s}.pt", args.fps)
        if pr is None or gr is None or pr.shape[0] < 4 or gr.shape[0] < 4:
            if args.require_all_stems:
                raise ValueError(
                    f"strict stem validation failed for {s}: "
                    f"pred_shape={None if pr is None else tuple(pr.shape)} "
                    f"ref_shape={None if gr is None else tuple(gr.shape)}"
                )
            continue
        if args.require_equal_lengths and pr.shape[0] != gr.shape[0]:
            raise ValueError(
                f"strict temporal coverage failed for {s}: "
                f"pred_frames={pr.shape[0]} gt_frames={gr.shape[0]}"
            )
        # Every paired metric, including Jerk-to-GT, uses exactly the same temporal support.
        pr, gr = crop_to_common_length(pr, gr)
        jerk_pj = fk_positions_world(pr, args.fps)
        jerk_gj = fk_positions_world(gr, args.fps)

        if args.dump_fk_dir and n_done < args.dump_fk_limit:  # debug FK dump (no metric effect)
            pj = fk_positions(pr)  # (Tp,30,3) pose-only visualization dump
            gj = fk_positions(gr)
            os.makedirs(args.dump_fk_dir, exist_ok=True)
            np.save(f"{args.dump_fk_dir}/{s}_pred.npy", pj)
            np.save(f"{args.dump_fk_dir}/{s}_gt.npy", gj)
        fgd.update(pr, gr)  # FGD operates on repr directly
        # Disabled: div_cross.update(pj, gj) used pooled_joint_stats, not EMAGE L1div.
        emage_pj = fk_positions_ba(pr)  # identity root rotation + zero translation
        emage_gj = fk_positions_ba(gr)
        motion_position_pred = emage_pj.reshape(emage_pj.shape[0], -1)  # EMAGE: (T, J*3)
        motion_position_gt = emage_gj.reshape(emage_gj.shape[0], -1)
        div.compute(motion_position_pred)
        div_gt.compute(motion_position_gt)
        if multimodality is not None and s not in multimodality_missing:
            sampled_positions = []
            for run_dir in multimodality_run_dirs:
                sampled_repr = load_repr(str(run_dir / f"{s}.pt"), args.fps)
                if sampled_repr is None:
                    raise ValueError(f"invalid Multimodality representation: {run_dir / f'{s}.pt'}")
                sampled_positions.append(fk_positions_ba(sampled_repr))
            # No GT crop, temporal resampling, padding, or tail-drop is permitted here.
            multimodality.update(sampled_positions)
            n_multimodality += 1
        semantic = load_semantic(
            args.sem_dir,
            s,
            emage_pj.shape[0],
            args.fps,
            args.semantic_fps,
            args.semantic_protocol,
        )
        if semantic is not None:
            srgr.run(emage_pj, emage_gj, semantic)
            n_srgr += 1
        elif args.require_all_stems:
            raise ValueError(f"strict semantic validation failed for {s}")
        wav_path = wav_of(args.wav_dir, s)
        beat.update(emage_pj, wav_path)
        beat_gt.update(emage_gj, wav_path)
        jerk.update(jerk_pj, jerk_gj)
        if foot is not None:
            foot.update(pr)
        n_done += 1

    if n_done < 2:
        print(f"Only {n_done} valid clips — need >=2 for FGD/Div.")
        return

    # ---- gather ----
    fgd_val, fgd_np, fgd_ng = fgd.compute()
    # Disabled with the cross-clip Div calls above:
    # div_pred, div_gt = div_cross.compute()
    jerk_gen, jerk_ref = jerk.summary()
    jerk_gen_length_weighted, jerk_ref_length_weighted = jerk.summary_length_weighted()
    ba_result = beat.summary()
    ba_gt_result = beat_gt.summary()
    if args.require_all_stems and (ba_result["n"] != n_done or ba_gt_result["n"] != n_done):
        raise ValueError(
            "strict BA coverage failed: "
            f"pred_valid={ba_result['n']} gt_valid={ba_gt_result['n']} "
            f"evaluated_clips={n_done}"
        )
    if args.require_all_stems and (jerk_gen["n"] != n_done or jerk_ref["n"] != n_done):
        raise ValueError(
            "strict Jerk coverage failed: "
            f"pred_valid={jerk_gen['n']} gt_valid={jerk_ref['n']} "
            f"evaluated_clips={n_done}"
        )
    div_value = div.avg()
    div_gt_value = div_gt.avg()
    div_closeness = gt_closeness(div_value, div_gt_value)
    ba_closeness = gt_closeness(ba_result["mean"], ba_gt_result["mean"])
    jerk_closeness = gt_closeness(jerk_gen["mean"], jerk_ref["mean"])
    jerk_ratio = (
        float(jerk_gen["mean"] / jerk_ref["mean"])
        if jerk_gen["mean"] is not None and jerk_ref["mean"] is not None and jerk_ref["mean"] != 0.0
        else None
    )
    jerk_length_weighted_closeness = gt_closeness(
        jerk_gen_length_weighted["mean"],
        jerk_ref_length_weighted["mean"],
    )
    jerk_length_weighted_ratio = (
        float(jerk_gen_length_weighted["mean"] / jerk_ref_length_weighted["mean"])
        if jerk_gen_length_weighted["mean"] is not None
        and jerk_ref_length_weighted["mean"] is not None
        and jerk_ref_length_weighted["mean"] != 0.0
        else None
    )
    if srgr.counter == 0 or srgr.denominator <= 0:
        raise SystemExit("SRGR has zero evaluated frames or zero semantic mass")
    if args.expected_srgr_frames and srgr.counter != args.expected_srgr_frames:
        raise ValueError(
            f"formal SRGR frame audit failed: actual={srgr.counter} "
            f"expected={args.expected_srgr_frames}"
        )
    srgr_val = srgr.avg()
    if multimodality is None:
        multimodality_status = "SKIPPED_NO_MULTIRUN"
        multimodality_val = None
        multimodality_eq16_literal = None
    elif multimodality.counter == 0:
        multimodality_status = "SKIPPED_NO_COMPLETE_CLIPS"
        multimodality_val = None
        multimodality_eq16_literal = None
    else:
        multimodality_status = "COMPLETE" if multimodality.counter == n_done else "PARTIAL_COVERAGE"
        multimodality_val = multimodality.avg()
        multimodality_eq16_literal = multimodality.avg_eq16_literal()
    if args.require_all_multimodality and n_multimodality != n_done:
        raise ValueError(
            "strict Multimodality evaluated-clip coverage failed: "
            f"valid={n_multimodality} evaluated_clips={n_done}"
        )
    result = {
        "tag": args.tag or args.pred_dir,
        "n_clips": n_done,
        "fps": args.fps,
        "params": {
            "root_index": ROOT_INDEX,
            "up_axis": UP_AXIS,
            "num_bodies": NUM_G1_BODIES,
            "fgd_feature": (
                f"map2latent(ckpt={os.path.basename(args.g1_ae_ckpt)}, "
                f"mode={cfg['input_mode']}, vae_length={cfg['vae_length']}, "
                f"input_norm={cfg['input_normalization']}, "
                f"input_norm_protocol={cfg['input_normalization_protocol']}, "
                f"checkpoint_fps={checkpoint_fps_label}); learned FGD — "
                f"COMPARABLE ONLY across runs "
                f"with the SAME encoder ckpt; NOT EMAGE-paper-comparable"
            ),
            "fgd_samples": f"{fgd_np}/{fgd_ng} latent samples",
            "fgd_distance": "EMAGE FGD.frechet_distance: eps=1e-6 offset only inside sqrtm; "
            "unmodified covariance traces; no clamp",
            "div_metric": "Div = EMAGE L1div logic in G1 space: within-clip per-frame position "
            "MAD, pred and corresponding GT, frame-weighted; identity root "
            "orientation and zero translation; pred/GT common-prefix aligned before "
            "FK; primary comparison is abs(Div-Div_GT), lower is better.",
            "multimodality_metric": "Audio2Gestures repeated-sampling protocol in G1 space: for "
            "each audio/stem, average per-frame L1 FK-position distance "
            "over every unordered pair of independent samples; all 30 G1 "
            "BFS bodies; identity root orientation and zero translation; "
            "then equal-weight average over complete stems.",
            "multimodality_paper": "https://arxiv.org/abs/2108.06720",
            "multimodality_root": args.multimodality_root or None,
            "multimodality_expected_runs": args.multimodality_runs,
            "multimodality_pair_normalization": "C(N,2) for Multimodality; paper Eq.16 printed "
            "N*ceil(N/2) is also reported as a diagnostic",
            "multimodality_temporal_policy": "all runs for a stem must have exactly equal T; "
            "no GT crop, resampling, padding, or tail-drop",
            "multimodality_coverage": f"{n_multimodality}/{n_done} evaluated clips",
            "multimodality_status": multimodality_status,
            "srgr_metric": "SRGR mass on G1 FK positions: numerator=sum(hit*semantic), "
            "denominator=30*sum(semantic) over all evaluated clips; per-frame, "
            "per-body Euclidean-distance hit; all 30 G1 BFS bodies; identity root "
            "orientation and zero translation; dynamic normalization guarantees a "
            "perfect-prediction score of 1 on every nonzero-semantic test set.",
            "srgr_variant": "semantic_mass",
            "srgr_threshold_m": args.srgr_threshold,
            "srgr_semantic_dir": args.sem_dir,
            "srgr_semantic_fps": args.semantic_fps,
            "srgr_semantic_protocol": args.semantic_protocol,
            "srgr_coverage": f"{n_srgr}/{n_done} clips, {srgr.counter} frames",
            "srgr_semantic_mass": srgr.semantic_mass,
            "srgr_semantic_mass_source": ("dynamic_sum_after_pred_gt_common_prefix_crop"),
            "srgr_semantic_mean_diagnostic": srgr.semantic_mass / srgr.counter,
            "srgr_expected_frames": args.expected_srgr_frames or None,
            "temporal_alignment": "All paired metrics, including Jerk-to-GT, use the common "
            "prefix: "
            "t=min(T_pred,T_gt), pred=pred[:t], gt=gt[:t]; no temporal "
            "resampling/padding and no fixed-duration crop.",
            "jerk_metric": "OMG body_jerk_mean adapted to Hermes G1: full world-space FK with "
            "stored root rotation and root translation reconstructed from "
            "yaw_delta + base_vel_local; all 30 bodies; m/s^3; both the historical "
            "clip-equal mean and a length-weighted mean with per-clip weight T-3 "
            "are reported; "
            "prediction and GT use the same common temporal prefix; primary "
            "comparison is abs(Jerk-Jerk_GT), lower is better; no valid mask.",
            "jerk_coverage": (
                f"pred={jerk_gen['n']}/{n_done} clips, gt={jerk_ref['n']}/{n_done} clips"
            ),
            "require_equal_lengths": bool(args.require_equal_lengths),
            "ba_convention": (
                "%s (sigma=%.2fs, order=%d, upper-body, identity root orientation); "
                "common-prefix audio/motion duration; cross-clip FRAME-WEIGHTED mean "
                "(EMAGE/GestureLSM); NO head/tail trim; NOT OMG music->motion"
            )
            % (args.ba_direction, args.bc_sigma, args.bc_order),
            "ba_gt_comparison": "prediction and corresponding GT use the same audio, common "
            "prefix, MMAE, FK, and frame weighting; primary comparison is "
            "abs(BA-BA_GT), lower is better.",
            "ba_coverage": (
                f"pred={ba_result['n']}/{n_done} clips, gt={ba_gt_result['n']}/{n_done} clips"
            ),
            "mmae_file": args.mmae_file,
            "stem_selection": {
                "val_split": args.val_split or None,
                "exclude_stems_file": args.exclude_stems_file or None,
                "require_all_stems": bool(args.require_all_stems),
            },
            "foot_metrics_enabled": bool(args.enable_foot_metrics),
            "foot_anchor": "canonical-grounded-stance" if args.enable_foot_metrics else None,
        },
        "metrics": {
            "FGD": fgd_val,
            # Disabled: cross-clip pooled-feature Div_pred/Div_gt is not EMAGE L1div.
            # "Div_pred": div_pred,
            # "Div_gt": div_gt,
            "Div": div_value,
            "Div_GT": div_gt_value,
            "Div_signed_gap": div_closeness["signed_gap"],
            "Div_gap": div_closeness["absolute_gap"],
            "Div_relative_gap": div_closeness["relative_gap"],
            "Multimodality": multimodality_val,
            "Multimodality_eq16_literal": multimodality_eq16_literal,
            "Multimodality_status": multimodality_status,
            "Multimodality_clips": n_multimodality,
            "Multimodality_runs_per_clip": (
                args.multimodality_runs if multimodality is not None else 0
            ),
            "Multimodality_pairs_per_clip": (
                multimodality.pair_count if multimodality is not None else 0
            ),
            "Multimodality_frames": (multimodality.frames if multimodality is not None else 0),
            "SRGR": srgr_val,
            "SRGR_numerator": srgr.numerator,
            "SRGR_denominator": srgr.denominator,
            "SRGR_semantic_mass": srgr.semantic_mass,
            "SRGR_frames": srgr.counter,
            "BA": ba_result,
            "BA_GT": ba_gt_result,
            "BA_signed_gap": ba_closeness["signed_gap"],
            "BA_gap": ba_closeness["absolute_gap"],
            "BA_relative_gap": ba_closeness["relative_gap"],
            "Jerk": jerk_gen,
            "Jerk_GT": jerk_ref,
            "Jerk_signed_gap": jerk_closeness["signed_gap"],
            "Jerk_gap": jerk_closeness["absolute_gap"],
            "Jerk_relative_gap": jerk_closeness["relative_gap"],
            "Jerk_ratio": jerk_ratio,
            "Jerk_length_weighted": jerk_gen_length_weighted,
            "Jerk_GT_length_weighted": jerk_ref_length_weighted,
            "Jerk_length_weighted_signed_gap": jerk_length_weighted_closeness["signed_gap"],
            "Jerk_length_weighted_gap": jerk_length_weighted_closeness["absolute_gap"],
            "Jerk_length_weighted_relative_gap": jerk_length_weighted_closeness["relative_gap"],
            "Jerk_length_weighted_ratio": jerk_length_weighted_ratio,
            # Backward-compatible aliases for existing benchmark consumers.
            "jerk_generated": jerk_gen,
            "jerk_reference": jerk_ref,
            "jerk_generated_length_weighted": jerk_gen_length_weighted,
            "jerk_reference_length_weighted": jerk_ref_length_weighted,
        },
    }
    if foot is not None:
        fge, csl = foot.summary()
        result["metrics"]["foot_ground_error"] = fge
        result["metrics"]["contact_sliding_speed"] = csl

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(result, f, indent=2)

    m = result["metrics"]
    ba = m["BA"]["mean"]
    ba_str = f"{ba:.4f}" if ba is not None else "nan"
    ba_gt = m["BA_GT"]["mean"]
    ba_gt_str = f"{ba_gt:.4f}" if ba_gt is not None else "nan"
    ba_gap = m["BA_gap"]
    ba_gap_str = f"{ba_gap:.4f}" if ba_gap is not None else "nan"
    div = m["Div"]
    div_str = f"{div:.4f}" if div is not None else "nan"
    div_gt_value = m["Div_GT"]
    div_gt_str = f"{div_gt_value:.4f}" if div_gt_value is not None else "nan"
    div_gap = m["Div_gap"]
    div_gap_str = f"{div_gap:.4f}" if div_gap is not None else "nan"
    mm = m["Multimodality"]
    mm_str = f"{mm:.4f}" if mm is not None else m["Multimodality_status"]
    line = (
        f"[{result['tag']}] G1-emageFGD N={n_done} | FGD={m['FGD']:.3f}(lower=better) | "
        f"Div={div_str}(GT {div_gt_str}, gap {div_gap_str} lower=better) | "
        f"Multimodality={mm_str} | SRGR={m['SRGR']:.4f} | "
        f"BA={ba_str}(GT {ba_gt_str}, gap {ba_gap_str} lower=better) "
    )
    jg = m["Jerk"]["mean"]
    jr = m["Jerk_GT"]["mean"]
    jerk_gap = m["Jerk_gap"]
    if jg is not None and jr is not None:
        jerk_gap_str = f"{jerk_gap:.1f}" if jerk_gap is not None else "nan"
        ratio_str = f"{m['Jerk_ratio']:.2f}" if m["Jerk_ratio"] is not None else "nan"
        line += f"| Jerk={jg:.1f}(GT {jr:.1f}, gap {jerk_gap_str} lower=better, ratio {ratio_str}) "
    jg_weighted = m["Jerk_length_weighted"]["mean"]
    jr_weighted = m["Jerk_GT_length_weighted"]["mean"]
    if jg_weighted is not None and jr_weighted is not None:
        weighted_gap = m["Jerk_length_weighted_gap"]
        weighted_ratio = m["Jerk_length_weighted_ratio"]
        weighted_gap_str = f"{weighted_gap:.1f}" if weighted_gap is not None else "nan"
        weighted_ratio_str = f"{weighted_ratio:.2f}" if weighted_ratio is not None else "nan"
        line += (
            f"| JerkW={jg_weighted:.1f}(GT {jr_weighted:.1f}, "
            f"gap {weighted_gap_str} lower=better, ratio {weighted_ratio_str}, weight=T-3) "
        )
    if foot is not None:
        line += (
            f"| foot_ground_err={m['foot_ground_error']['mean']} "
            f"contact_slide={m['contact_sliding_speed']['mean']}"
        )
    print(line, flush=True)
    print(f"wrote {args.out}", flush=True)


if __name__ == "__main__":
    main()
