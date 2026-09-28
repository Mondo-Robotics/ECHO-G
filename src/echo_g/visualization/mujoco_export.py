# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

import numpy as np
import torch

from echo_g.evaluation.g1_kinematics import (
    G1_ANKLE_BFS,
    G1_SOLE_OFFSETS,
    G1_SOLE_RADIUS,
    g1_forward_kinematics_with_rotations,
)
from echo_g.evaluation.robot_repr import TOTAL_ROBOT_MOTION_DIM
from echo_g.evaluation.rotation import (
    extract_yaw_from_rotation,
    matrix_to_quaternion,
    rotation_6d_to_matrix,
    rotation_matrix_from_yaw,
)


class RootTranslationMode(str, Enum):
    GROUNDED_INTEGRATED = "grounded_integrated"
    GROUNDED_FIXED = "grounded_fixed"


class RootOrientationMode(str, Enum):
    PER_FRAME_6D = "per_frame_6d"
    FIRST_FRAME = "first_frame"
    IDENTITY = "identity"


@dataclass(frozen=True)
class MujocoMotionArrays:
    fps: float
    joint_pos: np.ndarray
    body_pos_w: np.ndarray
    body_quat_w: np.ndarray
    body_names: np.ndarray
    initial_pelvis_height: float
    root_translation_mode: RootTranslationMode
    root_orientation_mode: RootOrientationMode
    ground_each_frame: bool


def _validate_physical_repr(robot_repr: torch.Tensor | np.ndarray, fps: float) -> torch.Tensor:
    representation = torch.as_tensor(robot_repr, dtype=torch.float32, device="cpu")
    if representation.ndim == 3 and representation.shape[0] == 1:
        representation = representation[0]
    if representation.ndim != 2 or representation.shape[1] != TOTAL_ROBOT_MOTION_DIM:
        raise ValueError(
            f"expected physical robot_repr with shape (T,{TOTAL_ROBOT_MOTION_DIM}), "
            f"got {tuple(representation.shape)}"
        )
    if representation.shape[0] < 1:
        raise ValueError("robot_repr must contain at least one frame")
    if not torch.isfinite(representation).all():
        raise ValueError("robot_repr contains non-finite values")
    if not np.isfinite(fps) or fps <= 0.0:
        raise ValueError(f"fps must be positive and finite, got {fps}")
    return representation.contiguous()


def _grounded_translation_z(
    joints: torch.Tensor,
    base_rotation: torch.Tensor,
) -> torch.Tensor:
    frame_count = int(joints.shape[0])
    zero_translation = torch.zeros(1, frame_count, 3, dtype=joints.dtype)
    body_positions, body_rotations = g1_forward_kinematics_with_rotations(
        joints.unsqueeze(0),
        base_rotation.unsqueeze(0),
        zero_translation,
    )
    offsets = torch.tensor(G1_SOLE_OFFSETS, dtype=joints.dtype)
    sole_points: list[torch.Tensor] = []
    for ankle_index in G1_ANKLE_BFS.values():
        ankle_position = body_positions[0, :, ankle_index]
        ankle_rotation = body_rotations[0, :, ankle_index]
        sole_points.append(
            ankle_position.unsqueeze(1) + torch.einsum("tij,pj->tpi", ankle_rotation, offsets)
        )
    sole_bottom = torch.cat(sole_points, dim=1)[..., 2] - G1_SOLE_RADIUS
    return -sole_bottom.min(dim=1).values


def _grounded_initial_translation(
    joints: torch.Tensor,
    base_rotation: torch.Tensor,
) -> torch.Tensor:
    initial_translation = torch.zeros(3, dtype=joints.dtype)
    initial_translation[2] = _grounded_translation_z(
        joints[:1],
        base_rotation[:1],
    )[0]
    return initial_translation


def _resolve_root_orientation(
    stored_rotation: torch.Tensor,
    mode: RootOrientationMode,
) -> torch.Tensor:
    if mode == RootOrientationMode.PER_FRAME_6D:
        return stored_rotation
    if mode == RootOrientationMode.FIRST_FRAME:
        return stored_rotation[:1].expand_as(stored_rotation)
    if mode == RootOrientationMode.IDENTITY:
        return (
            torch.eye(
                3,
                device=stored_rotation.device,
                dtype=stored_rotation.dtype,
            )
            .view(1, 3, 3)
            .expand_as(stored_rotation)
        )
    raise ValueError(f"unsupported root orientation mode: {mode}")


def _resolve_base_translation(
    yaw_delta: torch.Tensor,
    base_velocity_local: torch.Tensor,
    base_rotation: torch.Tensor,
    initial_translation: torch.Tensor,
    fps: float,
    mode: RootTranslationMode,
) -> torch.Tensor:
    frame_count = int(yaw_delta.shape[0])
    translation = initial_translation.unsqueeze(0).repeat(frame_count, 1)
    if mode == RootTranslationMode.GROUNDED_FIXED or frame_count == 1:
        return translation
    if mode != RootTranslationMode.GROUNDED_INTEGRATED:
        raise ValueError(f"unsupported root translation mode: {mode}")

    yaw = torch.zeros(frame_count, dtype=yaw_delta.dtype)
    yaw[0] = extract_yaw_from_rotation(base_rotation[0])
    yaw[1:] = yaw[0] + torch.cumsum(yaw_delta[1:], dim=0)
    yaw_rotation = rotation_matrix_from_yaw(yaw)
    global_velocity = torch.einsum("tij,tj->ti", yaw_rotation, base_velocity_local)
    translation[1:] = initial_translation + torch.cumsum(
        global_velocity[1:] / float(fps),
        dim=0,
    )
    return translation


def _parse_root_modes(
    root_translation_mode: RootTranslationMode | str | None,
    root_orientation_mode: RootOrientationMode | str,
    integrate_translation: bool | None,
) -> tuple[RootTranslationMode, RootOrientationMode]:
    orientation_mode = RootOrientationMode(root_orientation_mode)
    explicit_translation_mode = (
        RootTranslationMode(root_translation_mode) if root_translation_mode is not None else None
    )
    legacy_translation_mode = None
    if integrate_translation is not None:
        legacy_translation_mode = (
            RootTranslationMode.GROUNDED_INTEGRATED
            if integrate_translation
            else RootTranslationMode.GROUNDED_FIXED
        )
    if (
        explicit_translation_mode is not None
        and legacy_translation_mode is not None
        and explicit_translation_mode != legacy_translation_mode
    ):
        raise ValueError("root_translation_mode conflicts with legacy integrate_translation")
    translation_mode = (
        explicit_translation_mode
        or legacy_translation_mode
        or RootTranslationMode.GROUNDED_INTEGRATED
    )
    return translation_mode, orientation_mode


@torch.no_grad()
def robot_repr_to_mujoco_arrays(
    robot_repr: torch.Tensor | np.ndarray,
    fps: float = 30.0,
    integrate_translation: bool | None = None,
    *,
    root_translation_mode: RootTranslationMode | str | None = None,
    root_orientation_mode: RootOrientationMode | str = RootOrientationMode.PER_FRAME_6D,
    ground_each_frame: bool = False,
) -> MujocoMotionArrays:
    """Decode physical 39D G1 representation for ``render_g1_motion.py``.

    The default modes match ``scripts/eval_g1_motion_cls.py::fk_grounded_world``:
    first-frame sole grounding, integrated local root velocity, and the stored
    per-frame 6D root rotation. Root orientation can instead be held at the
    stored first frame or replaced by identity for every frame. When
    ``ground_each_frame`` is enabled, the resolved translation Z is overwritten
    each frame so the lowest sole proxy bottom is exactly at ground level.
    ``integrate_translation`` is retained as a compatibility alias for the two
    explicit translation modes.
    """
    representation = _validate_physical_repr(robot_repr, fps)
    translation_mode, orientation_mode = _parse_root_modes(
        root_translation_mode,
        root_orientation_mode,
        integrate_translation,
    )
    stored_base_rotation = rotation_6d_to_matrix(representation[:, :6])
    base_rotation = _resolve_root_orientation(stored_base_rotation, orientation_mode)
    yaw_delta = representation[:, 6]
    base_velocity_local = representation[:, 7:10]
    joints = representation[:, 10:39]

    initial_translation = _grounded_initial_translation(joints, base_rotation)
    base_translation = _resolve_base_translation(
        yaw_delta,
        base_velocity_local,
        base_rotation,
        initial_translation,
        fps,
        translation_mode,
    )
    if ground_each_frame:
        base_translation[:, 2] = _grounded_translation_z(joints, base_rotation)
    base_quaternion_wxyz = matrix_to_quaternion(base_rotation)
    base_quaternion_wxyz = base_quaternion_wxyz / torch.linalg.vector_norm(
        base_quaternion_wxyz,
        dim=-1,
        keepdim=True,
    ).clamp_min(1e-8)

    frame_count = int(representation.shape[0])
    body_pos_w = np.zeros((frame_count, 2, 3), dtype=np.float32)
    body_pos_w[:, 1] = base_translation.numpy()
    body_quat_w = np.zeros((frame_count, 2, 4), dtype=np.float32)
    body_quat_w[:, 0, 0] = 1.0
    body_quat_w[:, 1] = base_quaternion_wxyz.numpy()

    return MujocoMotionArrays(
        fps=float(fps),
        joint_pos=joints.numpy().astype(np.float32, copy=False),
        body_pos_w=body_pos_w,
        body_quat_w=body_quat_w,
        body_names=np.asarray(("world", "pelvis")),
        initial_pelvis_height=float(initial_translation[2]),
        root_translation_mode=translation_mode,
        root_orientation_mode=orientation_mode,
        ground_each_frame=bool(ground_each_frame),
    )


def save_robot_repr_mujoco_npz(
    robot_repr: torch.Tensor | np.ndarray,
    output_path: str | Path,
    fps: float = 30.0,
    integrate_translation: bool | None = None,
    *,
    root_translation_mode: RootTranslationMode | str | None = None,
    root_orientation_mode: RootOrientationMode | str = RootOrientationMode.PER_FRAME_6D,
    ground_each_frame: bool = False,
) -> MujocoMotionArrays:
    arrays = robot_repr_to_mujoco_arrays(
        robot_repr,
        fps=fps,
        integrate_translation=integrate_translation,
        root_translation_mode=root_translation_mode,
        root_orientation_mode=root_orientation_mode,
        ground_each_frame=ground_each_frame,
    )
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        destination,
        fps=np.asarray([arrays.fps], dtype=np.float32),
        joint_pos=arrays.joint_pos,
        body_pos_w=arrays.body_pos_w,
        body_quat_w=arrays.body_quat_w,
        body_names=arrays.body_names,
        root_translation_mode=np.asarray(arrays.root_translation_mode.value),
        root_orientation_mode=np.asarray(arrays.root_orientation_mode.value),
        ground_each_frame=np.asarray(arrays.ground_each_frame),
    )
    return arrays
