# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

"""Featurize G1 robot_repr[T,39] -> the G1 skeleton-conv AE encoder input (SINGLE SOURCE OF
TRUTH, shared by scripts/train_g1_aeskconv.py and scripts/eval_omg_robot_emage_FGD.py).

The skeleton-conv encoder (hermes/robot_fgd/motion_encoder.VAESKConv, topology=G1_BFS_PARENTS)
sees 30 "bodies" (edge0 = virtual root, bodies 1..29 = the 29 URDF joints in BFS order). Each
G1 joint is a single revolute DoF (scalar radian), so we feed [sin θ, cos θ] per body (avoids
the ±π wrap discontinuity a raw-angle temporal conv would hit).

  robot_repr[10:39] = 29 joint angles in CANONICAL order (G1_JOINT_NAMES).
  encoder wants BFS body order -> body b (1..29) uses canonical index _BFS_TO_CANONICAL[b-1];
  body 0 (pelvis root) has no joint -> zeros.

Encoder input modes (all keep the SkeletonConv in_channels % 30 == 0 constraint):
  * "padded6": 6 channels/body = [sinθ, cosθ, 0, 0, 0, 0]  -> (T, 30*6 = 180).
               Matches EMAGE's hard-coded channel_base[0]=6, so LocalEncoder is UNEDITED.
  * "pure2"  : 2 channels/body = [sinθ, cosθ]              -> (T, 30*2 =  60).
               Requires the encoder built with args.channel_base=2 (LocalEncoder honours it).
  * "rot6d"  : 6 channels/body = root-global + joint-local rotation-6D
                                                               -> (T, 30*6 = 180).
  * "rich12" : 12 channels/body = rot6d + position/dynamics -> (T, 30*12 = 360).

The base/root block robot_repr[0:10] (base_orient_6d + yaw_delta + base_vel_local) is
DELIBERATELY NOT fed to the skeleton conv (FGD must not cover root motion, per project decision).
It is only a recon target for the plain decoder; see the trainer's --base-weight.
"""

from __future__ import annotations

import numpy as np

from echo_g.evaluation.g1_kinematics import NUM_G1_BODIES, NUM_G1_JOINTS, canonical_to_bfs

ROBOT_MOTION_DIM = 39
BASE_DIM = 10  # robot_repr[0:10] = base_orient_6d(6) + yaw_delta(1) + base_vel(3)
JOINT_SLICE = slice(10, 39)  # 29 canonical joint angles
ENCODER_INPUT_NORMALIZATION_PROTOCOL = "per_channel_train_frame_zscore_std_plus_1e-4_v1"

_CH_PER_BODY = {"padded6": 6, "pure2": 2, "rot6d": 6, "rich12": 12}
ENCODER_INPUT_DIM = {
    m: NUM_G1_BODIES * c for m, c in _CH_PER_BODY.items()
}  # padded6:180,pure2:60,rich12:360


def channels_per_body(input_mode: str) -> int:
    if input_mode not in _CH_PER_BODY:
        raise ValueError(f"input_mode must be one of {list(_CH_PER_BODY)}, got {input_mode!r}")
    return _CH_PER_BODY[input_mode]


def normalize_encoder_input(
    values: np.ndarray,
    feature_mean: np.ndarray,
    feature_std: np.ndarray,
) -> np.ndarray:
    """Apply the checkpoint's train-split per-channel z-score to SKCNN input features."""
    # The network always consumes float32. Fix the arithmetic dtype here so training inputs
    # (cached float32 features) and evaluation inputs (some featurizers return float64 arrays)
    # cannot differ merely because NumPy promoted one side before the final torch.float() cast.
    array = np.asarray(values, dtype=np.float32)
    mean = np.asarray(feature_mean, dtype=np.float32)
    std = np.asarray(feature_std, dtype=np.float32)
    if array.ndim < 2:
        raise ValueError(f"SKCNN input must be at least 2D, got {array.shape}")
    expected_shape = (array.shape[-1],)
    if mean.shape != expected_shape or std.shape != expected_shape:
        raise ValueError(
            f"SKCNN normalization stats must be {expected_shape}, got {mean.shape}/{std.shape}"
        )
    if not np.isfinite(array).all() or not np.isfinite(mean).all() or not np.isfinite(std).all():
        raise ValueError("SKCNN input normalization received non-finite values")
    if np.any(std <= 0):
        raise ValueError("SKCNN feature_std must be strictly positive")
    normalized = (array - mean) / std
    if not np.isfinite(normalized).all():
        raise ValueError("SKCNN input normalization produced non-finite values")
    return normalized


def bfs_joint_angles_to_encoder_input(
    joints_bfs: np.ndarray, input_mode: str = "padded6"
) -> np.ndarray:
    """(T, 29) joint angles in BFS-joint order -> (T, 30*C) skeleton-conv encoder input.

    This is the CORE featurizer both entry points funnel into, so training (which reads
    beat2_robot_g1 robot_g1.npz `joint_pos`, already BFS order) and eval (robot_repr[10:39],
    canonical order → remapped) produce byte-identical encoder inputs.

    C = 6 (padded6) or 2 (pure2). Body 0 (pelvis root) is zeros; body b (1..29) = joint b-1.
    """
    joints_bfs = np.asarray(joints_bfs, dtype=np.float64)
    if joints_bfs.ndim != 2 or joints_bfs.shape[-1] != NUM_G1_JOINTS:
        raise ValueError(f"joints_bfs must be (T, {NUM_G1_JOINTS}), got {joints_bfs.shape}")
    C = channels_per_body(input_mode)
    T = joints_bfs.shape[0]
    out = np.zeros((T, NUM_G1_BODIES, C), dtype=np.float64)  # body 0 stays zeros (root, no DoF)
    out[:, 1:, 0] = np.sin(joints_bfs)
    out[:, 1:, 1] = np.cos(joints_bfs)  # channels 2..C-1 stay 0 (padded6 pad)
    return out.reshape(T, NUM_G1_BODIES * C)  # (T, 180) or (T, 60)


def robot_repr_to_encoder_input(
    rr: np.ndarray, input_mode: str = "padded6", align: str = "none"
) -> np.ndarray:
    """(T, 39) DE-NORMALIZED robot_repr (canonical joint order) -> (T, 30*C) encoder input.
    Eval-side entry: slices joints[10:39] (canonical) → BFS reorder → core featurizer.
    input_mode='rich12' dispatches to the PHUMA-style rich featurizer (rot6d+pos+vel+root).
    input_mode='rot6d' takes its per-body rotation block. `align='rz90'` is meaningful for both
    root-covering modes; sincos modes exclude the root entirely."""
    if input_mode == "rich12":
        return robot_repr_to_rich12(rr, align=align)
    if input_mode == "rot6d":
        rich = robot_repr_to_rich12(rr, align=align).reshape(-1, NUM_G1_BODIES, RICH12_CH)
        return rich[:, :, :6].reshape(rich.shape[0], NUM_G1_BODIES * 6)
    rr = np.asarray(rr, dtype=np.float64)
    if rr.ndim != 2 or rr.shape[-1] != ROBOT_MOTION_DIM:
        raise ValueError(f"robot_repr must be (T, {ROBOT_MOTION_DIM}), got {rr.shape}")
    joints_canon = rr[:, JOINT_SLICE]  # (T, 29) canonical order
    joints_bfs = canonical_to_bfs(joints_canon)  # (T, 29) canonical -> BFS
    return bfs_joint_angles_to_encoder_input(joints_bfs, input_mode)


# ------------------------------------------------------------------ rich12 (PHUMA-style)
# Per body b (1..29): [rot6d(R_local) (6) | FK local pos (3) | pos velocity (3)]  -> 12 ch
# Root body 0:        [base_orient_6d (6) | base_vel_local (3) | yaw_delta (1) | 0,0] -> 12 ch
# FK is run with base_rot=I, base_trans=0 so joint pos/rot are ROOT-RELATIVE (translation- and
# heading-invariant, cross-retarget-consistent). The root block keeps base orientation (PHUMA:
# root global rotation retained, no de-heading) and the frame-wise root LINEAR VELOCITY (not
# absolute translation), matching robot_repr[0:10]. FGD DOES cover root motion in this mode.
RICH12_CH = 12


# canonv6 -> GMR coordinate alignment: LEFT-multiply base orientation by Rz(+90deg). VERIFIED
# (canonv6-vs-GMR mean-base angle 89.7deg is a pure Rz(90); after this the residual is 0.04deg).
# Joint angles / FK positions / root velocity are already cross-source consistent, so ONLY the
# root global rotation (base_orient_6d) needs this when mixing canonv6 (method A) with GMR methods.
_RZ90 = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def robot_repr_to_rich12(rr, align: str = "none") -> np.ndarray:
    """(T, 39) DE-NORMALIZED robot_repr -> (T, 30*12=360) rich encoder input. Torch internally
    (reuses g1_kinematics Rodrigues constants + FK), returns numpy float64.

    align='rz90' LEFT-multiplies base_orient_6d by Rz(+90deg) to bring a canonv6-source clip into
    the GMR coordinate system (root-covering FGD across mixed sources). 'none' = leave as-is."""
    import torch

    from echo_g.evaluation import g1_kinematics as G1K
    from echo_g.evaluation.rotation import matrix_to_rotation_6d, rotation_6d_to_matrix

    rr_t = torch.as_tensor(np.asarray(rr, dtype=np.float32))
    if rr_t.ndim != 2 or rr_t.shape[-1] != ROBOT_MOTION_DIM:
        raise ValueError(f"robot_repr must be (T, {ROBOT_MOTION_DIM}), got {tuple(rr_t.shape)}")
    T = rr_t.shape[0]
    dt = rr_t.dtype
    if align == "rz90":
        A = torch.as_tensor(_RZ90, dtype=dt)
        R0 = rotation_6d_to_matrix(rr_t[:, 0:6])  # (T,3,3)
        rr_t = rr_t.clone()
        rr_t[:, 0:6] = matrix_to_rotation_6d(torch.einsum("ij,tjk->tik", A, R0))
    elif align != "none":
        raise ValueError(f"align must be 'none' or 'rz90', got {align!r}")
    base_orient_6d = rr_t[:, 0:6]  # (T,6)  PHUMA: root global rotation, kept
    yaw_delta = rr_t[:, 6:7]  # (T,1)
    base_vel_local = rr_t[:, 7:10]  # (T,3)  PHUMA: root linear velocity (not abs trans)
    joints_canon = rr_t[:, 10:39]  # (T,29) canonical order

    # --- per-joint LOCAL rotation R_local (BFS order), same Rodrigues as g1_forward_kinematics ---
    K = G1K._K.to(dt)
    K2 = G1K._K2.to(dt)
    rpy = G1K._JOINT_RPY_MAT_T.to(dt)
    ang_bfs = G1K.canonical_to_bfs(joints_canon)  # (T,29) canonical -> BFS
    sin_a = torch.sin(ang_bfs).unsqueeze(-1).unsqueeze(-1)
    cos_a = torch.cos(ang_bfs).unsqueeze(-1).unsqueeze(-1)
    eye3 = torch.eye(3, dtype=dt)
    R_axis = eye3 + sin_a * K + (1 - cos_a) * K2  # (T,29,3,3)
    R_local = torch.einsum("jik,tjkl->tjil", rpy, R_axis)  # (T,29,3,3)
    rot6d = matrix_to_rotation_6d(R_local)  # (T,29,6)

    # --- ROOT-RELATIVE FK positions (base_rot=I, base_trans=0) -> (T,30,3) BFS ---
    base_rot_I = torch.eye(3, dtype=dt).view(1, 3, 3).expand(T, 3, 3)
    base_trans_0 = torch.zeros(T, 3, dtype=dt)
    pos = G1K.g1_forward_kinematics(
        joints_canon.unsqueeze(0), base_rot_I.unsqueeze(0), base_trans_0.unsqueeze(0)
    )[0]  # (T,30,3) BFS
    vel = torch.zeros_like(pos)
    vel[1:] = pos[1:] - pos[:-1]  # frame-wise pos velocity

    out = torch.zeros(T, NUM_G1_BODIES, RICH12_CH, dtype=dt)
    # root body 0: [base_orient_6d(6) | base_vel_local(3) | yaw_delta(1) | 0,0]
    out[:, 0, 0:6] = base_orient_6d
    out[:, 0, 6:9] = base_vel_local
    out[:, 0, 9:10] = yaw_delta
    # bodies 1..29: [rot6d(6) | fk_pos(3) | pos_vel(3)]  (body b uses BFS joint b-1, BFS pos b)
    out[:, 1:, 0:6] = rot6d
    out[:, 1:, 6:9] = pos[:, 1:]
    out[:, 1:, 9:12] = vel[:, 1:]
    return out.reshape(T, NUM_G1_BODIES * RICH12_CH).double().numpy()  # (T,360)


def recon_target(rr: np.ndarray) -> np.ndarray:
    """The 39-d recon target = the raw robot_repr itself (decoder reconstructs native units)."""
    return np.asarray(rr, dtype=np.float64)
