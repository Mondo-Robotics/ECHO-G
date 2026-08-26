# Copyright (c) 2026 SOAR-LAB,
# School of Intelligence Science and Technology, Nanjing University.
# Project Hermes is developed by Dr. Hao Xu's team with sponsorship from
# Mondo Robotics (妙动科技).
# Licensed under PolyForm Noncommercial License 1.0.0.
# Commercial use requires written permission from Dr. Hao Xu
# (xuhao3e8@gmail.com) and Dr. Shuo Yang (shuo.yang@mondorobotics.com).

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Any

from echo_g.config import ExperimentConfig
from echo_g.data import RobotSpeechDataset, load_motion_stats

LOGGER = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate an ECHO-G dataset release")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--limit", type=int, default=0)
    return parser.parse_args()


def validate_release(
    config: ExperimentConfig,
    data_root: Path,
    limit: int = 0,
) -> dict[str, Any]:
    mean, std = load_motion_stats(
        data_root / config.data.stats_file, config.model.motion_dim
    )
    summary: dict[str, Any] = {
        "status": "valid",
        "fps": config.data.fps,
        "motion_dim": config.model.motion_dim,
        "splits": {},
    }
    for split in ("train", "val"):
        dataset = RobotSpeechDataset(
            data_root,
            config.data,
            config.model,
            split,
            mean,
            std,
            max_items=limit,
        )
        timed = 0
        minimum_frames = None
        maximum_frames = 0
        for index in range(len(dataset)):
            sample = dataset[index]
            frames = int(sample["frames"])
            minimum_frames = frames if minimum_frames is None else min(minimum_frames, frames)
            maximum_frames = max(maximum_frames, frames)
            timed += int(sample["has_timing"])
        summary["splits"][split] = {
            "utterances": len(dataset),
            "timed_utterances": timed,
            "minimum_frames": minimum_frames,
            "maximum_frames": maximum_frames,
        }
    return summary


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    result = validate_release(config, args.data_root, args.limit)
    LOGGER.info("dataset validation complete\n%s", json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
