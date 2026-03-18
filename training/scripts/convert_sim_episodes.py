"""Convert raw Isaac Sim episode recordings to LeRobot v3.0 format.

If sim_data_collector.py was not used (e.g., data collected via ROS bags
or custom scripts), this tool converts raw recordings into the standard
LeRobot v3.0 directory layout expected by OpenPi training.

Usage:
    python convert_sim_episodes.py \
        --input-dir /data/raw_episodes \
        --output-dir /data/episodes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None


def convert(input_dir: str, output_dir: str, fps: int = 30):
    in_path = Path(input_dir)
    out_path = Path(output_dir)

    (out_path / "meta" / "episodes").mkdir(parents=True, exist_ok=True)
    (out_path / "data").mkdir(exist_ok=True)
    (out_path / "videos").mkdir(exist_ok=True)

    all_frames = []
    episodes = []
    global_idx = 0

    # Look for episode directories or numpy files
    episode_dirs = sorted(in_path.glob("episode_*"))
    if not episode_dirs:
        # Try flat numpy files
        episode_dirs = [in_path]

    for ep_idx, ep_dir in enumerate(episode_dirs):
        state_file = ep_dir / "states.npy"
        action_file = ep_dir / "actions.npy"

        if not state_file.exists() or not action_file.exists():
            print(f"Skipping {ep_dir}: missing states.npy or actions.npy")
            continue

        states = np.load(state_file)
        actions = np.load(action_file)
        ep_len = min(len(states), len(actions))

        for frame_idx in range(ep_len):
            frame = {
                "episode_index": ep_idx,
                "frame_index": frame_idx,
                "index": global_idx,
                "timestamp": frame_idx / fps,
                "observation.state": states[frame_idx].tolist(),
                "action": actions[frame_idx].tolist(),
            }
            all_frames.append(frame)
            global_idx += 1

        episodes.append({
            "episode_index": ep_idx,
            "tasks": ["sim_task"],
            "length": ep_len,
        })

        # Save episode metadata
        ep_meta_path = out_path / "meta" / "episodes" / f"{ep_idx:06d}.json"
        with open(ep_meta_path, "w") as f:
            json.dump(episodes[-1], f, indent=2)

    if not all_frames:
        print("No frames found to convert.")
        return

    # Write Parquet
    if pa is not None and pq is not None:
        table = pa.Table.from_pylist(all_frames)
        pq.write_table(table, out_path / "data" / "train-00000-of-00001.parquet")

    # Write info.json
    info = {
        "codebase_version": "v3.0",
        "robot_type": "so_arm101",
        "fps": fps,
        "total_episodes": len(episodes),
        "total_frames": global_idx,
        "features": {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
        },
    }
    with open(out_path / "meta" / "info.json", "w") as f:
        json.dump(info, f, indent=2)

    print(f"Converted {len(episodes)} episodes ({global_idx} frames) to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", required=True)
    parser.add_argument("--output-dir", default="/data/episodes")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()
    convert(args.input_dir, args.output_dir, args.fps)


if __name__ == "__main__":
    main()
