# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

"""Accurate 30-body forward kinematics for the Unitree G1 29-DOF robot.

All kinematic data is extracted from g1_beyondmimic_29dof.urdf.
Body and joint ordering follows the IsaacLab BFS convention, matching
the ``body_pos_w`` arrays in GMR-exported NPZ files.

This module is the single source of truth for G1 FK — both visualization
and training loss should use functions from here.
"""

from __future__ import annotations

import torch

# ---------------------------------------------------------------------------
# Constants: IsaacLab BFS ordering (30 bodies, 29 joints)
# Source: md_retarget/src/md_retarget/export/isaaclab_remap.py
# ---------------------------------------------------------------------------

NUM_G1_BODIES: int = 30
NUM_G1_JOINTS: int = 29

G1_BFS_BODY_NAMES: list[str] = [
    "pelvis",  # 0
    "left_hip_pitch_link",  # 1
    "right_hip_pitch_link",  # 2
    "waist_yaw_link",  # 3
    "left_hip_roll_link",  # 4
    "right_hip_roll_link",  # 5
    "waist_roll_link",  # 6
    "left_hip_yaw_link",  # 7
    "right_hip_yaw_link",  # 8
    "torso_link",  # 9
    "left_knee_link",  # 10
    "right_knee_link",  # 11
    "left_shoulder_pitch_link",  # 12
    "right_shoulder_pitch_link",  # 13
    "left_ankle_pitch_link",  # 14
    "right_ankle_pitch_link",  # 15
    "left_shoulder_roll_link",  # 16
    "right_shoulder_roll_link",  # 17
    "left_ankle_roll_link",  # 18
    "right_ankle_roll_link",  # 19
    "left_shoulder_yaw_link",  # 20
    "right_shoulder_yaw_link",  # 21
    "left_elbow_link",  # 22
    "right_elbow_link",  # 23
    "left_wrist_roll_link",  # 24
    "right_wrist_roll_link",  # 25
    "left_wrist_pitch_link",  # 26
    "right_wrist_pitch_link",  # 27
    "left_wrist_yaw_link",  # 28
    "right_wrist_yaw_link",  # 29
]

# Parent body index for each body (-1 = root).
# BFS joint i connects parent body to child body i+1.
G1_BFS_PARENTS: list[int] = [
    -1,  # 0: pelvis (root)
    0,  # 1: left_hip_pitch ← pelvis
    0,  # 2: right_hip_pitch ← pelvis
    0,  # 3: waist_yaw ← pelvis
    1,  # 4: left_hip_roll ← left_hip_pitch
    2,  # 5: right_hip_roll ← right_hip_pitch
    3,  # 6: waist_roll ← waist_yaw
    4,  # 7: left_hip_yaw ← left_hip_roll
    5,  # 8: right_hip_yaw ← right_hip_roll
    6,  # 9: torso ← waist_roll
    7,  # 10: left_knee ← left_hip_yaw
    8,  # 11: right_knee ← right_hip_yaw
    9,  # 12: left_shoulder_pitch ← torso
    9,  # 13: right_shoulder_pitch ← torso
    10,  # 14: left_ankle_pitch ← left_knee
    11,  # 15: right_ankle_pitch ← right_knee
    12,  # 16: left_shoulder_roll ← left_shoulder_pitch
    13,  # 17: right_shoulder_roll ← right_shoulder_pitch
    14,  # 18: left_ankle_roll ← left_ankle_pitch
    15,  # 19: right_ankle_roll ← right_ankle_pitch
    16,  # 20: left_shoulder_yaw ← left_shoulder_roll
    17,  # 21: right_shoulder_yaw ← right_shoulder_roll
    20,  # 22: left_elbow ← left_shoulder_yaw
    21,  # 23: right_elbow ← right_shoulder_yaw
    22,  # 24: left_wrist_roll ← left_elbow
    23,  # 25: right_wrist_roll ← right_elbow
    24,  # 26: left_wrist_pitch ← left_wrist_roll
    25,  # 27: right_wrist_pitch ← right_wrist_roll
    26,  # 28: left_wrist_yaw ← left_wrist_pitch
    27,  # 29: right_wrist_yaw ← right_wrist_pitch
]

# Skeleton connections for rendering: list of (parent_body, child_body).
G1_SKELETON_CONNECTIONS_30: list[tuple[int, int]] = [
    (parent, child) for child, parent in enumerate(G1_BFS_PARENTS) if parent >= 0
]

# OMG/eval_g1_motion_cls sole-contact proxy geometry.  Keep these definitions
# next to the canonical G1 FK so evaluation and MuJoCo-export grounding cannot
# silently drift apart.
G1_ANKLE_BFS: dict[int, int] = {0: 18, 1: 19}
G1_SOLE_OFFSETS: tuple[tuple[float, float, float], ...] = (
    (-0.05, 0.025, -0.03),
    (-0.05, -0.025, -0.03),
    (0.12, 0.03, -0.03),
    (0.12, -0.03, -0.03),
)
G1_SOLE_RADIUS: float = 0.005

# ---------------------------------------------------------------------------
# URDF joint data in BFS order (29 joints, BFS joint i → body i+1)
# Extracted from g1_beyondmimic_29dof.urdf
# ---------------------------------------------------------------------------

# Joint origin xyz offsets — child relative to parent, in parent frame.
# Shape: (29, 3)
_BFS_JOINT_XYZ: list[list[float]] = [
    # BFS 0: left_hip_pitch
    [0.0, 0.064452, -0.1027],
    # BFS 1: right_hip_pitch
    [0.0, -0.064452, -0.1027],
    # BFS 2: waist_yaw
    [0.0, 0.0, 0.0],
    # BFS 3: left_hip_roll
    [0.0, 0.052, -0.030465],
    # BFS 4: right_hip_roll
    [0.0, -0.052, -0.030465],
    # BFS 5: waist_roll
    [-0.0039635, 0.0, 0.044],
    # BFS 6: left_hip_yaw
    [0.025001, 0.0, -0.12412],
    # BFS 7: right_hip_yaw
    [0.025001, 0.0, -0.12412],
    # BFS 8: waist_pitch
    [0.0, 0.0, 0.0],
    # BFS 9: left_knee
    [-0.078273, 0.0021489, -0.17734],
    # BFS 10: right_knee
    [-0.078273, -0.0021489, -0.17734],
    # BFS 11: left_shoulder_pitch
    [0.0039563, 0.10022, 0.24778],
    # BFS 12: right_shoulder_pitch
    [0.0039563, -0.10021, 0.24778],
    # BFS 13: left_ankle_pitch
    [0.0, -9.4445e-05, -0.30001],
    # BFS 14: right_ankle_pitch
    [0.0, 9.4445e-05, -0.30001],
    # BFS 15: left_shoulder_roll
    [0.0, 0.038, -0.013831],
    # BFS 16: right_shoulder_roll
    [0.0, -0.038, -0.013831],
    # BFS 17: left_ankle_roll
    [0.0, 0.0, -0.017558],
    # BFS 18: right_ankle_roll
    [0.0, 0.0, -0.017558],
    # BFS 19: left_shoulder_yaw
    [0.0, 0.00624, -0.1032],
    # BFS 20: right_shoulder_yaw
    [0.0, -0.00624, -0.1032],
    # BFS 21: left_elbow
    [0.015783, 0.0, -0.080518],
    # BFS 22: right_elbow
    [0.015783, 0.0, -0.080518],
    # BFS 23: left_wrist_roll
    [0.100, 0.00188791, -0.010],
    # BFS 24: right_wrist_roll
    [0.100, -0.00188791, -0.010],
    # BFS 25: left_wrist_pitch
    [0.038, 0.0, 0.0],
    # BFS 26: right_wrist_pitch
    [0.038, 0.0, 0.0],
    # BFS 27: left_wrist_yaw
    [0.046, 0.0, 0.0],
    # BFS 28: right_wrist_yaw
    [0.046, 0.0, 0.0],
]

# Joint origin rpy rotation offsets (radians). Most are zero.
# Non-zero: hip_roll (±10° pitch), knee (±10° pitch), shoulder_pitch (±16° roll),
#           shoulder_roll (±16° roll).
# Shape: (29, 3) as [roll, pitch, yaw]
_BFS_JOINT_RPY: list[list[float]] = [
    # BFS 0: left_hip_pitch
    [0.0, 0.0, 0.0],
    # BFS 1: right_hip_pitch
    [0.0, 0.0, 0.0],
    # BFS 2: waist_yaw
    [0.0, 0.0, 0.0],
    # BFS 3: left_hip_roll
    [0.0, -0.1749, 0.0],
    # BFS 4: right_hip_roll
    [0.0, -0.1749, 0.0],
    # BFS 5: waist_roll
    [0.0, 0.0, 0.0],
    # BFS 6: left_hip_yaw
    [0.0, 0.0, 0.0],
    # BFS 7: right_hip_yaw
    [0.0, 0.0, 0.0],
    # BFS 8: waist_pitch
    [0.0, 0.0, 0.0],
    # BFS 9: left_knee
    [0.0, 0.1749, 0.0],
    # BFS 10: right_knee
    [0.0, 0.1749, 0.0],
    # BFS 11: left_shoulder_pitch
    [0.27931, 5.4949e-05, -0.00019159],
    # BFS 12: right_shoulder_pitch
    [-0.27931, 5.4949e-05, 0.00019159],
    # BFS 13: left_ankle_pitch
    [0.0, 0.0, 0.0],
    # BFS 14: right_ankle_pitch
    [0.0, 0.0, 0.0],
    # BFS 15: left_shoulder_roll
    [-0.27925, 0.0, 0.0],
    # BFS 16: right_shoulder_roll
    [0.27925, 0.0, 0.0],
    # BFS 17: left_ankle_roll
    [0.0, 0.0, 0.0],
    # BFS 18: right_ankle_roll
    [0.0, 0.0, 0.0],
    # BFS 19: left_shoulder_yaw
    [0.0, 0.0, 0.0],
    # BFS 20: right_shoulder_yaw
    [0.0, 0.0, 0.0],
    # BFS 21: left_elbow
    [0.0, 0.0, 0.0],
    # BFS 22: right_elbow
    [0.0, 0.0, 0.0],
    # BFS 23: left_wrist_roll
    [0.0, 0.0, 0.0],
    # BFS 24: right_wrist_roll
    [0.0, 0.0, 0.0],
    # BFS 25: left_wrist_pitch
    [0.0, 0.0, 0.0],
    # BFS 26: right_wrist_pitch
    [0.0, 0.0, 0.0],
    # BFS 27: left_wrist_yaw
    [0.0, 0.0, 0.0],
    # BFS 28: right_wrist_yaw
    [0.0, 0.0, 0.0],
]

# Joint rotation axis (unit vector in child frame).
# Shape: (29, 3)
_BFS_JOINT_AXIS: list[list[float]] = [
    # BFS 0: left_hip_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 1: right_hip_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 2: waist_yaw — Z
    [0.0, 0.0, 1.0],
    # BFS 3: left_hip_roll — X
    [1.0, 0.0, 0.0],
    # BFS 4: right_hip_roll — X
    [1.0, 0.0, 0.0],
    # BFS 5: waist_roll — X
    [1.0, 0.0, 0.0],
    # BFS 6: left_hip_yaw — Z
    [0.0, 0.0, 1.0],
    # BFS 7: right_hip_yaw — Z
    [0.0, 0.0, 1.0],
    # BFS 8: waist_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 9: left_knee — Y
    [0.0, 1.0, 0.0],
    # BFS 10: right_knee — Y
    [0.0, 1.0, 0.0],
    # BFS 11: left_shoulder_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 12: right_shoulder_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 13: left_ankle_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 14: right_ankle_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 15: left_shoulder_roll — X
    [1.0, 0.0, 0.0],
    # BFS 16: right_shoulder_roll — X
    [1.0, 0.0, 0.0],
    # BFS 17: left_ankle_roll — X
    [1.0, 0.0, 0.0],
    # BFS 18: right_ankle_roll — X
    [1.0, 0.0, 0.0],
    # BFS 19: left_shoulder_yaw — Z
    [0.0, 0.0, 1.0],
    # BFS 20: right_shoulder_yaw — Z
    [0.0, 0.0, 1.0],
    # BFS 21: left_elbow — Y
    [0.0, 1.0, 0.0],
    # BFS 22: right_elbow — Y
    [0.0, 1.0, 0.0],
    # BFS 23: left_wrist_roll — X
    [1.0, 0.0, 0.0],
    # BFS 24: right_wrist_roll — X
    [1.0, 0.0, 0.0],
    # BFS 25: left_wrist_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 26: right_wrist_pitch — Y
    [0.0, 1.0, 0.0],
    # BFS 27: left_wrist_yaw — Z
    [0.0, 0.0, 1.0],
    # BFS 28: right_wrist_yaw — Z
    [0.0, 0.0, 1.0],
]

# Mapping from canonical joint order (robot_repr dims 10:39) to BFS joint index.
# canonical[i] → BFS joint G1_CANONICAL_TO_BFS[i]
G1_CANONICAL_TO_BFS: list[int] = [
    0,  # canonical 0  (left_hip_pitch)     → BFS 0
    3,  # canonical 1  (left_hip_roll)      → BFS 3
    6,  # canonical 2  (left_hip_yaw)       → BFS 6
    9,  # canonical 3  (left_knee)          → BFS 9
    13,  # canonical 4  (left_ankle_pitch)   → BFS 13
    17,  # canonical 5  (left_ankle_roll)    → BFS 17
    1,  # canonical 6  (right_hip_pitch)    → BFS 1
    4,  # canonical 7  (right_hip_roll)     → BFS 4
    7,  # canonical 8  (right_hip_yaw)      → BFS 7
    10,  # canonical 9  (right_knee)         → BFS 10
    14,  # canonical 10 (right_ankle_pitch)  → BFS 14
    18,  # canonical 11 (right_ankle_roll)   → BFS 18
    2,  # canonical 12 (waist_yaw)          → BFS 2
    5,  # canonical 13 (waist_roll)         → BFS 5
    8,  # canonical 14 (waist_pitch)        → BFS 8
    11,  # canonical 15 (left_shoulder_pitch)  → BFS 11
    15,  # canonical 16 (left_shoulder_roll)   → BFS 15
    19,  # canonical 17 (left_shoulder_yaw)    → BFS 19
    21,  # canonical 18 (left_elbow)           → BFS 21
    23,  # canonical 19 (left_wrist_roll)      → BFS 23
    25,  # canonical 20 (left_wrist_pitch)     → BFS 25
    27,  # canonical 21 (left_wrist_yaw)       → BFS 27
    12,  # canonical 22 (right_shoulder_pitch) → BFS 12
    16,  # canonical 23 (right_shoulder_roll)  → BFS 16
    20,  # canonical 24 (right_shoulder_yaw)   → BFS 20
    22,  # canonical 25 (right_elbow)          → BFS 22
    24,  # canonical 26 (right_wrist_roll)     → BFS 24
    26,  # canonical 27 (right_wrist_pitch)    → BFS 26
    28,  # canonical 28 (right_wrist_yaw)      → BFS 28
]

# Inverse mapping: BFS joint index → canonical index.
_BFS_TO_CANONICAL: list[int] = [0] * NUM_G1_JOINTS
for _c, _b in enumerate(G1_CANONICAL_TO_BFS):
    _BFS_TO_CANONICAL[_b] = _c

# Self-consistency guard (module-load): the two maps MUST be exact inverses, or every reorder
# below silently scrambles limbs. Cheap, runs once at import.
assert all(_BFS_TO_CANONICAL[G1_CANONICAL_TO_BFS[_c]] == _c for _c in range(NUM_G1_JOINTS)), (
    "G1_CANONICAL_TO_BFS / _BFS_TO_CANONICAL are not inverse permutations"
)


# =============================================================================
# THE joint-order conversion bottleneck. TWO conventions exist for the 29 joints:
#   * CANONICAL — human-readable URDF-doc grouping (left leg, right leg, waist, left arm, right
#     arm). This is how robot_repr[10:39] stores joint angles.
#   * BFS       — IsaacLab breadth-first tree order (parent index < child index), REQUIRED by the
#     FK recursion and the skeleton-conv encoder. This is how body_pos_w / joint_pos npz store.
# ALL reordering between the two MUST go through these two functions (do NOT inline `x[..., idx]`
# elsewhere — that scatters the mapping and invites getting the direction backwards). Direction is
# in the name. Verified against GMR body_pos_w truth (FK err=0) — see scripts test_g1_order.
# Reorder acts on the LAST axis (works for torch (...,29) and numpy (...,29)).
# =============================================================================
def canonical_to_bfs(x):
    """(..., 29) joint values in CANONICAL order -> BFS order. Inverse of bfs_to_canonical()."""
    return x[..., _BFS_TO_CANONICAL]  # out[..., b] = x[..., canonical idx of BFS joint b]


def bfs_to_canonical(x):
    """(..., 29) joint values in BFS order -> CANONICAL order. Inverse of canonical_to_bfs()."""
    return x[..., G1_CANONICAL_TO_BFS]  # out[..., c] = x[..., BFS idx of canonical joint c]


# ---------------------------------------------------------------------------
# Pre-computed constant tensors (created once at module load)
# ---------------------------------------------------------------------------


def _rpy_to_matrix(rpy: torch.Tensor) -> torch.Tensor:
    """Convert (roll, pitch, yaw) to 3x3 rotation matrix.

    URDF convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll).

    Args:
        rpy: shape (..., 3) as [roll, pitch, yaw].

    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    roll = rpy[..., 0]
    pitch = rpy[..., 1]
    yaw = rpy[..., 2]

    cr, sr = torch.cos(roll), torch.sin(roll)
    cp, sp = torch.cos(pitch), torch.sin(pitch)
    cy, sy = torch.cos(yaw), torch.sin(yaw)

    # Rz @ Ry @ Rx
    r00 = cy * cp
    r01 = cy * sp * sr - sy * cr
    r02 = cy * sp * cr + sy * sr
    r10 = sy * cp
    r11 = sy * sp * sr + cy * cr
    r12 = sy * sp * cr - cy * sr
    r20 = -sp
    r21 = cp * sr
    r22 = cp * cr

    return torch.stack(
        [
            torch.stack([r00, r01, r02], dim=-1),
            torch.stack([r10, r11, r12], dim=-1),
            torch.stack([r20, r21, r22], dim=-1),
        ],
        dim=-2,
    )


# Convert raw lists to tensors (float32, CPU). These are registered as
# buffers or moved to device inside FK functions.
_JOINT_XYZ_T = torch.tensor(_BFS_JOINT_XYZ, dtype=torch.float32)  # (29, 3)
_JOINT_AXIS_T = torch.tensor(_BFS_JOINT_AXIS, dtype=torch.float32)  # (29, 3)
_JOINT_RPY_T = torch.tensor(_BFS_JOINT_RPY, dtype=torch.float32)  # (29, 3)
_JOINT_RPY_MAT_T = _rpy_to_matrix(_JOINT_RPY_T)  # (29, 3, 3)
_CANONICAL_TO_BFS_T = torch.tensor(G1_CANONICAL_TO_BFS, dtype=torch.long)  # (29,)

# Pre-compute skew-symmetric matrices for Rodrigues formula.
# K[j] is the skew-symmetric matrix of axis[j].  Shape: (29, 3, 3)
_ax = _JOINT_AXIS_T[:, 0]
_ay = _JOINT_AXIS_T[:, 1]
_az = _JOINT_AXIS_T[:, 2]
_zeros = torch.zeros(NUM_G1_JOINTS)
_K = torch.stack(
    [
        torch.stack([_zeros, -_az, _ay], dim=-1),
        torch.stack([_az, _zeros, -_ax], dim=-1),
        torch.stack([-_ay, _ax, _zeros], dim=-1),
    ],
    dim=-2,
)  # (29, 3, 3)
_K2 = _K @ _K  # (29, 3, 3)


# ---------------------------------------------------------------------------
# Forward Kinematics (differentiable, batched)
# ---------------------------------------------------------------------------


def g1_forward_kinematics_with_rotations(
    joint_angles_canonical: torch.Tensor,
    base_rot: torch.Tensor,
    base_trans: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute G1 body positions and rotations via differentiable FK.

    Uses exact URDF link offsets, RPY rotation offsets, and joint axes.

    Args:
        joint_angles_canonical: Joint angles in canonical order (as stored in
            robot_repr dims 10:39), shape (B, T, 29).
        base_rot: Base orientation as 3x3 rotation matrix, shape (B, T, 3, 3).
        base_trans: Base translation, shape (B, T, 3).

    Returns:
        Body positions and rotations in BFS order with shapes
        ``(B, T, 30, 3)`` and ``(B, T, 30, 3, 3)``.
    """
    B, T, _ = joint_angles_canonical.shape
    device = joint_angles_canonical.device
    dtype = joint_angles_canonical.dtype

    # Move constants to the right device/dtype (cached after first call)
    xyz = _JOINT_XYZ_T.to(device=device, dtype=dtype)  # (29, 3)
    rpy_mat = _JOINT_RPY_MAT_T.to(device=device, dtype=dtype)  # (29, 3, 3)
    K = _K.to(device=device, dtype=dtype)  # (29, 3, 3)
    K2 = _K2.to(device=device, dtype=dtype)  # (29, 3, 3)
    joint_angles_bfs = canonical_to_bfs(joint_angles_canonical)  # (B,T,29) canonical -> BFS

    # Build local rotation matrices for all 29 joints via Rodrigues:
    # R_joint = I + sin(a) * K + (1 - cos(a)) * K^2
    angles = joint_angles_bfs  # (B, T, 29)
    sin_a = torch.sin(angles).unsqueeze(-1).unsqueeze(-1)  # (B, T, 29, 1, 1)
    cos_a = torch.cos(angles).unsqueeze(-1).unsqueeze(-1)  # (B, T, 29, 1, 1)
    eye3 = torch.eye(3, device=device, dtype=dtype)

    # R_axis_angle: (B, T, 29, 3, 3)
    R_axis = eye3 + sin_a * K + (1 - cos_a) * K2

    # Apply RPY rotation offset: R_local = R_rpy @ R_axis
    # rpy_mat: (29, 3, 3) → broadcast to (1, 1, 29, 3, 3)
    R_local = torch.einsum("jik,btjkl->btjil", rpy_mat, R_axis)  # (B, T, 29, 3, 3)

    # FK tree traversal: compute global_rot and position for each body
    # Use a list to avoid in-place operations (autograd-safe)
    global_rot = [torch.empty(0)] * NUM_G1_BODIES
    positions = [torch.empty(0)] * NUM_G1_BODIES

    # Body 0 (pelvis): root
    global_rot[0] = base_rot  # (B, T, 3, 3)
    positions[0] = base_trans  # (B, T, 3)

    for body_idx in range(1, NUM_G1_BODIES):
        parent = G1_BFS_PARENTS[body_idx]
        joint_idx = body_idx - 1  # BFS joint i connects to child body i+1

        # Global rotation: parent_rot @ R_local[joint]
        global_rot[body_idx] = torch.einsum(
            "btij,btjk->btik",
            global_rot[parent],
            R_local[:, :, joint_idx],
        )

        # Position: parent_pos + parent_rot @ xyz_offset
        offset = xyz[joint_idx]  # (3,)
        rotated_offset = torch.einsum(
            "btij,j->bti",
            global_rot[parent],
            offset,
        )
        positions[body_idx] = positions[parent] + rotated_offset

    return torch.stack(positions, dim=2), torch.stack(global_rot, dim=2)


def g1_forward_kinematics(
    joint_angles_canonical: torch.Tensor,
    base_rot: torch.Tensor,
    base_trans: torch.Tensor,
) -> torch.Tensor:
    """Compute G1 body positions via forward kinematics (differentiable).

    Uses exact URDF link offsets, RPY rotation offsets, and joint axes.

    Args:
        joint_angles_canonical: Joint angles in canonical order (as stored in
            robot_repr dims 10:39), shape (B, T, 29).
        base_rot: Base orientation as 3x3 rotation matrix, shape (B, T, 3, 3).
        base_trans: Base translation, shape (B, T, 3).

    Returns:
        Body positions in BFS order, shape (B, T, 30, 3).
    """
    positions, _ = g1_forward_kinematics_with_rotations(
        joint_angles_canonical,
        base_rot,
        base_trans,
    )
    return positions


def robot_repr_to_body_positions(
    robot_repr: torch.Tensor,
    mean: torch.Tensor,
    std: torch.Tensor,
    fps: float = 30.0,
) -> torch.Tensor:
    """Convert normalized 39-dim robot repr to 30 body positions (differentiable).

    Parses the robot representation, reconstructs base rotation and translation
    via velocity integration, then runs FK.

    Args:
        robot_repr: Normalized robot motion, shape (B, T, 39).
        mean: Dataset mean, shape (39,).
        std: Dataset std, shape (39,).
        fps: Frame rate for velocity integration.

    Returns:
        Body positions in BFS order, shape (B, T, 30, 3).
    """
    from echo_g.evaluation.rotation import (
        extract_yaw_from_rotation,
        rotation_6d_to_matrix,
        rotation_matrix_from_yaw,
    )

    B, T, D = robot_repr.shape
    device = robot_repr.device
    dtype = robot_repr.dtype

    # Denormalize
    denorm = robot_repr * (std.to(device) + 1e-8) + mean.to(device)

    # Parse representation
    base_orient_6d = denorm[:, :, :6]  # (B, T, 6)
    yaw_delta = denorm[:, :, 6]  # (B, T)
    base_vel_local = denorm[:, :, 7:10]  # (B, T, 3)
    joint_angles = denorm[:, :, 10:]  # (B, T, 29) canonical order

    # Base orientation → rotation matrix
    base_rot = rotation_6d_to_matrix(base_orient_6d)  # (B, T, 3, 3)

    # Integrate yaw + velocity → translation.
    # Seed the cumulative yaw with the TRUE frame-0 heading from base_orient (not 0). The encoder
    # (compute_root_velocity_local) projects global velocity into the local frame using the
    # ABSOLUTE yaw, so the decoder must re-seed with that same absolute frame-0 yaw or the world
    # trajectory is rotated by frame-0 yaw. For yaw-canonicalized data (frame-0 yaw≈0, e.g. GMR
    # bench GT) this is a no-op; for non-canonicalized data (e.g. canonv6, frame-0 yaw≈-88°) it
    # fixes a ~0.32 m trajectory error. See memory project_render_yaw0_fix.
    yaw = torch.zeros(B, T, device=device, dtype=dtype)
    yaw[:, 0] = extract_yaw_from_rotation(base_rot[:, 0])  # true frame-0 heading
    yaw[:, 1:] = yaw[:, 0:1] + torch.cumsum(yaw_delta[:, 1:], dim=1)
    yaw_rot = rotation_matrix_from_yaw(yaw)  # (B, T, 3, 3)
    global_vel = torch.einsum("btij,btj->bti", yaw_rot, base_vel_local)

    trans = torch.zeros(B, T, 3, device=device, dtype=dtype)
    trans[:, 1:] = torch.cumsum(global_vel[:, 1:] / fps, dim=1)

    return g1_forward_kinematics(joint_angles, base_rot, trans)
