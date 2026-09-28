# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import torch

from echo_g.evaluation.robot_repr import TOTAL_ROBOT_MOTION_DIM
from echo_g.visualization.mujoco_export import (
    RootOrientationMode,
    RootTranslationMode,
    save_robot_repr_mujoco_npz,
)

logger = logging.getLogger(__name__)

REPR_KEYS = ("robot_repr", "robot_motion", "pred", "robot", "motion")
REQUIRED_FPS = 30.0


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export physical G1 robot_repr[T,39] .pt files to the minimal NPZ "
            "schema consumed by scripts/render_g1_motion.py."
        )
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="One physical robot_repr .pt file")
    source.add_argument("--input-dir", type=Path, help="Directory recursively containing .pt files")
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path, help="Output .npz for --input")
    destination.add_argument("--output-dir", type=Path, help="Output root for --input-dir")
    parser.add_argument("--fps", type=float, default=REQUIRED_FPS)
    parser.add_argument(
        "--root-translation-mode",
        choices=[mode.value for mode in RootTranslationMode],
        default=RootTranslationMode.GROUNDED_INTEGRATED.value,
        help="How to reconstruct the root translation after first-frame sole grounding.",
    )
    parser.add_argument(
        "--root-orientation-mode",
        choices=[mode.value for mode in RootOrientationMode],
        default=RootOrientationMode.PER_FRAME_6D.value,
        help=(
            "Use each stored 6D root orientation, hold the stored first frame "
            "fixed, or force identity orientation for every frame."
        ),
    )
    parser.add_argument(
        "--ground-each-frame",
        action="store_true",
        help=(
            "Override root translation Z every frame so the lowest sole proxy "
            "bottom remains at z=0."
        ),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _extract_representation(payload: Any, path: Path, expected_fps: float) -> torch.Tensor:
    representation: torch.Tensor | None = None
    if isinstance(payload, dict):
        units = payload.get("representation_units")
        if units is not None and str(units).strip().lower() != "physical":
            raise ValueError(f"{path}: expected physical robot_repr, got units={units!r}")
        stored_fps = payload.get("fps")
        if stored_fps is not None and float(stored_fps) != expected_fps:
            raise ValueError(
                f"{path}: stored fps={float(stored_fps)} != requested fps={expected_fps}"
            )
        for key in REPR_KEYS:
            value = payload.get(key)
            if torch.is_tensor(value):
                representation = value
                break
        if representation is None:
            for value in payload.values():
                if (
                    torch.is_tensor(value)
                    and value.ndim in {2, 3}
                    and value.shape[-1] == TOTAL_ROBOT_MOTION_DIM
                ):
                    representation = value
                    break
    elif torch.is_tensor(payload):
        representation = payload

    if representation is None:
        raise ValueError(f"{path}: no robot_repr tensor found")
    representation = representation.detach().cpu().float()
    if representation.ndim == 3 and representation.shape[0] == 1:
        representation = representation[0]
    if isinstance(payload, dict) and payload.get("real_num_frames") is not None:
        real_num_frames = int(payload["real_num_frames"])
        if real_num_frames != int(representation.shape[0]):
            raise ValueError(
                f"{path}: real_num_frames={real_num_frames} but tensor has "
                f"{representation.shape[0]} frames"
            )
    return representation


def _convert(
    source: Path,
    destination: Path,
    fps: float,
    overwrite: bool,
    root_translation_mode: str,
    root_orientation_mode: str,
    ground_each_frame: bool,
) -> None:
    if not source.is_file():
        raise FileNotFoundError(source)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"output exists; pass --overwrite to replace it: {destination}")
    payload = torch.load(source, map_location="cpu", weights_only=False)
    representation = _extract_representation(payload, source, fps)
    arrays = save_robot_repr_mujoco_npz(
        representation,
        destination,
        fps=fps,
        root_translation_mode=root_translation_mode,
        root_orientation_mode=root_orientation_mode,
        ground_each_frame=ground_each_frame,
    )
    logger.info(
        "Exported %s: %d frames @ %.1f FPS, root_translation=%s, "
        "root_orientation=%s, ground_each_frame=%s, grounded pelvis z0=%.6f m -> %s",
        source,
        representation.shape[0],
        fps,
        arrays.root_translation_mode.value,
        arrays.root_orientation_mode.value,
        arrays.ground_each_frame,
        arrays.initial_pelvis_height,
        destination,
    )


def main() -> None:
    args = _parse_args()
    if args.fps != REQUIRED_FPS:
        raise ValueError(f"this evaluation/rendering protocol requires --fps {REQUIRED_FPS:g}")
    if (args.input is None) != (args.output is None):
        raise ValueError("--input must be paired with --output")
    if (args.input_dir is None) != (args.output_dir is None):
        raise ValueError("--input-dir must be paired with --output-dir")

    if args.input is not None:
        _convert(
            args.input.resolve(),
            args.output.resolve(),
            args.fps,
            args.overwrite,
            args.root_translation_mode,
            args.root_orientation_mode,
            args.ground_each_frame,
        )
        return

    input_root = args.input_dir.resolve()
    output_root = args.output_dir.resolve()
    sources = sorted(input_root.rglob("*.pt"))
    if not sources:
        raise ValueError(f"no .pt files found under {input_root}")
    for source in sources:
        destination = output_root / source.relative_to(input_root).with_suffix(".npz")
        _convert(
            source,
            destination,
            args.fps,
            args.overwrite,
            args.root_translation_mode,
            args.root_orientation_mode,
            args.ground_each_frame,
        )
    logger.info("Exported %d files from %s to %s", len(sources), input_root, output_root)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    main()
