# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

G1_JOINT_NAMES: list[str] = [
    # Left leg (6)
    "left_hip_pitch",
    "left_hip_roll",
    "left_hip_yaw",
    "left_knee",
    "left_ankle_pitch",
    "left_ankle_roll",
    # Right leg (6)
    "right_hip_pitch",
    "right_hip_roll",
    "right_hip_yaw",
    "right_knee",
    "right_ankle_pitch",
    "right_ankle_roll",
    # Waist (3)
    "waist_yaw",
    "waist_roll",
    "waist_pitch",
    # Left arm (7)
    "left_shoulder_pitch",
    "left_shoulder_roll",
    "left_shoulder_yaw",
    "left_elbow",
    "left_wrist_roll",
    "left_wrist_pitch",
    "left_wrist_yaw",
    # Right arm (7)
    "right_shoulder_pitch",
    "right_shoulder_roll",
    "right_shoulder_yaw",
    "right_elbow",
    "right_wrist_roll",
    "right_wrist_pitch",
    "right_wrist_yaw",
]

NUM_ROBOT_JOINTS = 29
BASE_ORIENT_6D_DIM = 6
BASE_YAW_DELTA_DIM = 1
BASE_VEL_LOCAL_DIM = 3
JOINT_ANGLES_DIM = NUM_ROBOT_JOINTS
TOTAL_ROBOT_MOTION_DIM = (
    BASE_ORIENT_6D_DIM + BASE_YAW_DELTA_DIM + BASE_VEL_LOCAL_DIM + JOINT_ANGLES_DIM
)  # 39
