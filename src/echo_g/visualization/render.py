# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

"""Previous G1 MuJoCo rendering pipeline, with portable asset paths and schema checks."""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from echo_g.evaluation.robot_repr import G1_JOINT_NAMES

logger = logging.getLogger(__name__)


def output_frame_indices(frame_count: int, source_fps: float, output_fps: float) -> np.ndarray:
    if frame_count < 1 or not np.isfinite(source_fps) or source_fps <= 0:
        raise ValueError("motion must contain frames and have a positive finite FPS")
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("output FPS must be positive and finite")
    count = max(1, int(round(frame_count / source_fps * output_fps)))
    return np.minimum((np.arange(count) * (frame_count / count)).astype(np.int64), frame_count - 1)


def _load_motion(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as payload:
        joints = np.asarray(payload["joint_pos"], dtype=np.float64)
        body_positions = np.asarray(payload["body_pos_w"], dtype=np.float64)
        body_quaternions = np.asarray(payload["body_quat_w"], dtype=np.float64)
        source_fps = float(np.asarray(payload["fps"]).reshape(-1)[0])
        names = [str(name) for name in payload["body_names"]]
    if joints.ndim != 2 or joints.shape[1] != 29 or joints.shape[0] < 1:
        raise ValueError(f"expected nonempty joint_pos[T,29], got {joints.shape}")
    if body_positions.shape != (joints.shape[0], len(names), 3):
        raise ValueError(
            f"body_pos_w shape does not match joint frames/body_names: {body_positions.shape}"
        )
    if body_quaternions.shape != (joints.shape[0], len(names), 4):
        raise ValueError("body_quat_w must contain one wxyz quaternion per frame and body")
    if names.count("pelvis") != 1:
        raise ValueError("body_names must contain exactly one pelvis")
    pelvis_index = names.index("pelvis")
    root_position = body_positions[:, pelvis_index]
    root_quaternion = body_quaternions[:, pelvis_index]
    if not all(np.isfinite(values).all() for values in (joints, root_position, root_quaternion)):
        raise ValueError("motion contains non-finite values")
    if not np.allclose(np.linalg.norm(root_quaternion, axis=1), 1.0, atol=1e-4):
        raise ValueError("pelvis quaternions must be normalized wxyz rotations")
    output_frame_indices(len(joints), source_fps, source_fps)
    return joints, root_position, root_quaternion, source_fps


def _joint_addresses(model: Any, mujoco: Any) -> tuple[np.ndarray, int]:
    if model.nq != 36 or model.nv != 35:
        raise ValueError(
            f"expected a floating-base G1 with 29 joints, got nq={model.nq}, nv={model.nv}"
        )
    if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE) or int(model.jnt_qposadr[0]) != 0:
        raise ValueError("the G1 root must be the first free joint")
    pelvis_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    if pelvis_id < 0:
        raise ValueError("MJCF is missing the pelvis body")
    addresses: list[int] = []
    for name in G1_JOINT_NAMES:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name + "_joint")
        if joint_id < 0:
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0 or int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_HINGE):
            raise ValueError(f"MJCF is missing expected G1 hinge joint {name!r}")
        addresses.append(int(model.jnt_qposadr[joint_id]))
    if len(set(addresses)) != 29:
        raise ValueError("MJCF joint mapping is not one-to-one")
    return np.asarray(addresses), pelvis_id


def render_g1_npz(
    npz_path: str,
    out_mp4: str,
    mjcf_path: str,
    width: int = 640,
    height: int = 480,
    fps_out: int = 30,
    camera_distance: float = 3.0,
    camera_azimuth: float = 90.0,
    camera_elevation: float = -15.0,
    follow_pelvis: bool = True,
) -> str:
    if width < 1 or height < 1 or camera_distance <= 0:
        raise ValueError("image size and camera distance must be positive")
    if not Path(mjcf_path).is_file():
        raise FileNotFoundError(mjcf_path)
    joints, root_position, root_quaternion, source_fps = _load_motion(npz_path)
    frame_ids = output_frame_indices(len(joints), source_fps, fps_out)
    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import imageio.v2 as imageio
    import mujoco

    model = mujoco.MjModel.from_xml_path(mjcf_path)
    joint_addresses, pelvis_id = _joint_addresses(model, mujoco)
    model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
    model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, width=width, height=height)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = camera_distance
    camera.azimuth = camera_azimuth
    camera.elevation = camera_elevation
    if not follow_pelvis:
        camera.lookat[:] = (root_position.min(axis=0) + root_position.max(axis=0)) / 2.0
    destination = Path(out_mp4)
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        with imageio.get_writer(
            str(destination), fps=fps_out, codec="libx264", quality=8
        ) as writer:
            for frame in frame_ids:
                data.qpos[:3] = root_position[frame]
                data.qpos[3:7] = root_quaternion[frame]
                data.qpos[joint_addresses] = joints[frame]
                mujoco.mj_forward(model, data)
                if follow_pelvis:
                    camera.lookat[:] = data.xpos[pelvis_id]
                renderer.update_scene(data, camera=camera)
                writer.append_data(renderer.render())
    finally:
        renderer.close()
    logger.info(
        "Wrote %s: %d frames @ %d FPS, %d x %d", destination, len(frame_ids), fps_out, width, height
    )
    return str(destination)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render physical G1 MuJoCo NPZ motion to MP4.")
    parser.add_argument("--npz", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument(
        "--mjcf", required=True, help="External G1 29-DoF robot MJCF with its meshes"
    )
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument(
        "--azimuth",
        type=float,
        default=180.0,
        help="0=behind, 90=right shoulder, 180=face-on, 270=left shoulder",
    )
    parser.add_argument("--elevation", type=float, default=-15.0)
    parser.add_argument("--distance", type=float, default=3.0)
    parser.add_argument("--fixed-camera", action="store_true")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    render_g1_npz(
        args.npz,
        args.out,
        args.mjcf,
        width=args.width,
        height=args.height,
        fps_out=args.fps,
        camera_azimuth=args.azimuth,
        camera_elevation=args.elevation,
        camera_distance=args.distance,
        follow_pelvis=not args.fixed_camera,
    )


if __name__ == "__main__":
    main()
