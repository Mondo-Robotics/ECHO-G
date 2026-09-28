# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

"""Rotation conversion utilities for motion representation.

Based on PyTorch3D rotation_conversions.py, adapted for Hermes.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def axis_angle_to_matrix(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to rotation matrix.

    Args:
        axis_angle: Rotations as axis-angle vectors, shape (..., 3).

    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    return quaternion_to_matrix(axis_angle_to_quaternion(axis_angle))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to quaternion.

    Args:
        axis_angle: Rotations as axis-angle vectors, shape (..., 3).

    Returns:
        Quaternions with real part first, shape (..., 4).
    """
    angles = torch.norm(axis_angle, p=2, dim=-1, keepdim=True)
    half_angles = 0.5 * angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    # For small angles: sin(x/2)/x ≈ 1/2 - x²/48
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    quaternions = torch.cat(
        [torch.cos(half_angles), axis_angle * sin_half_angles_over_angles], dim=-1
    )
    return quaternions


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to rotation matrices.

    Args:
        quaternions: Quaternions with real part first, shape (..., 4).

    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))


def matrix_to_rotation_6d(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to 6D representation.

    Uses Zhou et al.'s continuous 6D representation by taking first two columns.

    Args:
        matrix: Rotation matrices, shape (..., 3, 3).

    Returns:
        6D rotation representation, shape (..., 6).
    """
    return matrix[..., :2, :].clone().reshape(*matrix.size()[:-2], 6)


def rotation_6d_to_matrix(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to rotation matrix.

    Uses Gram-Schmidt orthogonalization.

    Args:
        d6: 6D rotation representation, shape (..., 6).

    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    a1, a2 = d6[..., :3], d6[..., 3:]
    b1 = F.normalize(a1, dim=-1)
    b2 = a2 - (b1 * a2).sum(-1, keepdim=True) * b1
    b2 = F.normalize(b2, dim=-1)
    b3 = torch.cross(b1, b2, dim=-1)
    return torch.stack((b1, b2, b3), dim=-2)


def axis_angle_to_rotation_6d(axis_angle: torch.Tensor) -> torch.Tensor:
    """Convert axis-angle to 6D rotation representation.

    Args:
        axis_angle: Rotations as axis-angle vectors, shape (..., 3).

    Returns:
        6D rotation representation, shape (..., 6).
    """
    return matrix_to_rotation_6d(axis_angle_to_matrix(axis_angle))


def rotation_6d_to_axis_angle(d6: torch.Tensor) -> torch.Tensor:
    """Convert 6D rotation representation to axis-angle.

    Args:
        d6: 6D rotation representation, shape (..., 6).

    Returns:
        Axis-angle rotations, shape (..., 3).
    """
    return matrix_to_axis_angle(rotation_6d_to_matrix(d6))


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to quaternions.

    Args:
        matrix: Rotation matrices, shape (..., 3, 3).

    Returns:
        Quaternions with real part first, shape (..., 4).
    """
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")

    def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
        ret = torch.zeros_like(x)
        positive_mask = x > 0
        ret[positive_mask] = torch.sqrt(x[positive_mask])
        return ret

    batch_shape = matrix.shape[:-2]
    m00 = matrix[..., 0, 0]
    m01 = matrix[..., 0, 1]
    m02 = matrix[..., 0, 2]
    m10 = matrix[..., 1, 0]
    m11 = matrix[..., 1, 1]
    m12 = matrix[..., 1, 2]
    m20 = matrix[..., 2, 0]
    m21 = matrix[..., 2, 1]
    m22 = matrix[..., 2, 2]
    quaternion_component_magnitudes = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    quaternion_candidates = torch.stack(
        (
            torch.stack(
                (
                    quaternion_component_magnitudes[..., 0] ** 2,
                    m21 - m12,
                    m02 - m20,
                    m10 - m01,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m21 - m12,
                    quaternion_component_magnitudes[..., 1] ** 2,
                    m10 + m01,
                    m02 + m20,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m02 - m20,
                    m10 + m01,
                    quaternion_component_magnitudes[..., 2] ** 2,
                    m12 + m21,
                ),
                dim=-1,
            ),
            torch.stack(
                (
                    m10 - m01,
                    m20 + m02,
                    m21 + m12,
                    quaternion_component_magnitudes[..., 3] ** 2,
                ),
                dim=-1,
            ),
        ),
        dim=-2,
    )
    denominator_floor = torch.tensor(
        0.1,
        dtype=matrix.dtype,
        device=matrix.device,
    )
    quaternion_candidates = quaternion_candidates / (
        2.0 * quaternion_component_magnitudes[..., None].clamp_min(denominator_floor)
    )
    best_conditioned = F.one_hot(
        quaternion_component_magnitudes.argmax(dim=-1),
        num_classes=4,
    ).to(dtype=torch.bool)
    return quaternion_candidates[best_conditioned].reshape(batch_shape + (4,))


def quaternion_to_axis_angle(quaternions: torch.Tensor) -> torch.Tensor:
    """Convert quaternions to axis-angle.

    Args:
        quaternions: Quaternions with real part first, shape (..., 4).

    Returns:
        Axis-angle rotations, shape (..., 3).
    """
    norms = torch.norm(quaternions[..., 1:], p=2, dim=-1, keepdim=True)
    half_angles = torch.atan2(norms, quaternions[..., :1])
    angles = 2 * half_angles
    eps = 1e-6
    small_angles = angles.abs() < eps
    sin_half_angles_over_angles = torch.empty_like(angles)
    sin_half_angles_over_angles[~small_angles] = (
        torch.sin(half_angles[~small_angles]) / angles[~small_angles]
    )
    sin_half_angles_over_angles[small_angles] = (
        0.5 - (angles[small_angles] * angles[small_angles]) / 48
    )
    return quaternions[..., 1:] / sin_half_angles_over_angles


def matrix_to_axis_angle(matrix: torch.Tensor) -> torch.Tensor:
    """Convert rotation matrices to axis-angle.

    Args:
        matrix: Rotation matrices, shape (..., 3, 3).

    Returns:
        Axis-angle rotations, shape (..., 3).
    """
    return quaternion_to_axis_angle(matrix_to_quaternion(matrix))


def extract_yaw_from_rotation(rot_matrix: torch.Tensor) -> torch.Tensor:
    """Extract yaw angle from rotation matrix.

    Assumes Z-up coordinate system where yaw is rotation around Z axis.
    AMASS and SMPL-X use Z-up convention.

    Args:
        rot_matrix: Rotation matrices, shape (..., 3, 3).

    Returns:
        Yaw angles in radians, shape (...).
    """
    # In Z-up, the forward direction projects onto XY plane.
    # First column of R is the rotated X-axis direction.
    # yaw = atan2(R[1,0], R[0,0])  (standard ZYX Euler extraction for heading)
    yaw = torch.atan2(rot_matrix[..., 1, 0], rot_matrix[..., 0, 0])
    return yaw


def rotation_matrix_from_yaw(yaw: torch.Tensor) -> torch.Tensor:
    """Create rotation matrix from yaw angle (Z-up, rotation around Z axis).

    Args:
        yaw: Yaw angles in radians, shape (...).

    Returns:
        Rotation matrices, shape (..., 3, 3).
    """
    cos_yaw = torch.cos(yaw)
    sin_yaw = torch.sin(yaw)
    zeros = torch.zeros_like(yaw)
    ones = torch.ones_like(yaw)

    # Rotation around Z axis
    rot = torch.stack(
        [
            torch.stack([cos_yaw, -sin_yaw, zeros], dim=-1),
            torch.stack([sin_yaw, cos_yaw, zeros], dim=-1),
            torch.stack([zeros, zeros, ones], dim=-1),
        ],
        dim=-2,
    )
    return rot
