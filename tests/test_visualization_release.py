# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from echo_g.evaluation.g1_motion_cls import fk_grounded_world
from echo_g.evaluation.robot_repr import G1_JOINT_NAMES
from echo_g.evaluation.rotation import matrix_to_rotation_6d
from echo_g.visualization.export import _extract_representation
from echo_g.visualization.mujoco_export import save_robot_repr_mujoco_npz
from echo_g.visualization.render import _joint_addresses, _load_motion, output_frame_indices


def test_export_root_trajectory_matches_benchmark_grounding(tmp_path: Path) -> None:
    motion = torch.zeros(45, 39)
    motion[:, :6] = matrix_to_rotation_6d(torch.eye(3))
    motion[:, 7] = 0.9
    path = tmp_path / "motion.npz"
    exported = save_robot_repr_mujoco_npz(motion, path)
    bodies, _, _, _ = fk_grounded_world(motion, 30)
    np.testing.assert_allclose(exported.body_pos_w[:, 1], bodies[:, 0].numpy(), atol=1e-6)
    joints, positions, quaternions, fps = _load_motion(path)
    assert joints.shape == (45, 29)
    assert fps == 30
    np.testing.assert_allclose(positions[:, 0], np.arange(45) * 0.03, atol=1e-6)
    np.testing.assert_allclose(quaternions, np.tile([1, 0, 0, 0], (45, 1)), atol=1e-6)
    assert exported.initial_pelvis_height > 0


def test_video_sampling_preserves_duration_and_zero_based_frames() -> None:
    np.testing.assert_array_equal(output_frame_indices(452, 30, 30), np.arange(452))
    indices = output_frame_indices(500, 50, 30)
    assert len(indices) == 300
    assert indices[0] == 0
    assert indices[-1] < 500
    with pytest.raises(ValueError):
        output_frame_indices(0, 30, 30)


def test_export_rejects_normalized_predictions_and_frame_metadata_mismatch() -> None:
    payload = {"robot_repr": torch.zeros(8, 39), "representation_units": "normalized"}
    with pytest.raises(ValueError, match="physical"):
        _extract_representation(payload, Path("sample.pt"), 30)
    payload = {"robot_repr": torch.zeros(8, 39), "real_num_frames": 9}
    with pytest.raises(ValueError, match="real_num_frames"):
        _extract_representation(payload, Path("sample.pt"), 30)


def test_mujoco_joint_mapping_is_named_not_document_order() -> None:
    mujoco = pytest.importorskip("mujoco")
    bodies = "".join(
        f'<body name="link_{index}" pos="0 0 0.02"><joint name="{name}_joint" '
        f'type="hinge"/><geom type="sphere" size="0.01" mass="0.01"/></body>'
        for index, name in enumerate(reversed(G1_JOINT_NAMES))
    )
    xml = (
        '<mujoco><worldbody><body name="pelvis"><freejoint/>'
        '<geom type="sphere" size="0.02" mass="0.1"/>' + bodies + "</body></worldbody></mujoco>"
    )
    model = mujoco.MjModel.from_xml_string(xml)
    addresses, pelvis = _joint_addresses(model, mujoco)
    assert pelvis == 1
    np.testing.assert_array_equal(addresses, np.arange(35, 6, -1))
