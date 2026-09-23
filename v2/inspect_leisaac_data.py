#!/usr/bin/env python3
"""Inspect a LeRobot LeIsaac dataset before starting fine-tuning."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


LOCAL_LEROBOT_SRC = Path(__file__).resolve().parents[1] / "v1" / "lerobot" / "src"
if LOCAL_LEROBOT_SRC.is_dir():
    sys.path.insert(0, str(LOCAL_LEROBOT_SRC))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="LightwheelAI/leisaac-pick-orange")
    parser.add_argument("--root", type=Path, default=None, help="Optional local LeRobot dataset root.")
    parser.add_argument("--no-videos", action="store_true", help="Skip video downloads when metadata is local.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    from lerobot.datasets.utils import BackwardCompatibilityError
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    try:
        dataset = LeRobotDataset(
            repo_id=args.dataset,
            root=args.root,
            download_videos=not args.no_videos,
        )
    except BackwardCompatibilityError:
        print(f"Dataset {args.dataset} is in LeRobot v2.1 format.")
        print("Run ./convert_leisaac_dataset.sh, then rerun this inspector.")
        return 2
    metadata = dataset.meta

    print(f"Dataset: {args.dataset}")
    print(f"Root: {metadata.root}")
    print(f"Episodes: {metadata.total_episodes}")
    print(f"Frames: {metadata.total_frames}")
    print(f"Tasks: {metadata.total_tasks}")
    print(f"FPS: {metadata.fps}")
    print("Task instructions:")
    for task in metadata.tasks.index.tolist():
        print(f"  - {task}")

    print("Features:")
    for key, feature in metadata.features.items():
        print(f"  - {key}: dtype={feature.get('dtype')}, shape={feature.get('shape')}, names={feature.get('names')}")

    print(f"Camera inputs: {metadata.camera_keys}")
    state_keys = [key for key in metadata.features if key == "observation.state"]
    action_keys = [key for key in metadata.features if key == "action"]
    print(f"State shape: {metadata.shapes.get(state_keys[0]) if state_keys else 'not found'}")
    print(f"Action shape: {metadata.shapes.get(action_keys[0]) if action_keys else 'not found'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
