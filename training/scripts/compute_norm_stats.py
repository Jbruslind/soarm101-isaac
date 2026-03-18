"""Compute normalization statistics for a LeRobot v3.0 dataset.

Reads all episode Parquet files and computes per-feature mean, std, min, max.
Saves to meta/stats.json, which OpenPi uses during training and inference.

Usage:
    python compute_norm_stats.py --data-dir /data/episodes
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

try:
    import pyarrow.parquet as pq
except ImportError:
    pq = None


def compute_stats(data_dir: str):
    data_path = Path(data_dir)
    parquet_dir = data_path / "data"

    if pq is None:
        print("pyarrow not installed. Cannot read Parquet files.")
        return

    parquet_files = sorted(parquet_dir.glob("*.parquet"))
    if not parquet_files:
        print(f"No Parquet files found in {parquet_dir}")
        return

    all_states = []
    all_actions = []

    for pf in parquet_files:
        table = pq.read_table(pf)
        df = table.to_pandas()

        if "observation.state" in df.columns:
            states = np.array(df["observation.state"].tolist(), dtype=np.float32)
            all_states.append(states)

        if "action" in df.columns:
            actions = np.array(df["action"].tolist(), dtype=np.float32)
            all_actions.append(actions)

    stats = {}

    if all_states:
        states = np.concatenate(all_states, axis=0)
        stats["observation.state"] = {
            "mean": states.mean(axis=0).tolist(),
            "std": np.maximum(states.std(axis=0), 1e-6).tolist(),
            "min": states.min(axis=0).tolist(),
            "max": states.max(axis=0).tolist(),
        }

    if all_actions:
        actions = np.concatenate(all_actions, axis=0)
        stats["action"] = {
            "mean": actions.mean(axis=0).tolist(),
            "std": np.maximum(actions.std(axis=0), 1e-6).tolist(),
            "min": actions.min(axis=0).tolist(),
            "max": actions.max(axis=0).tolist(),
        }

    stats_path = data_path / "meta" / "stats.json"
    stats_path.parent.mkdir(parents=True, exist_ok=True)
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    print(f"Saved normalization stats to {stats_path}")
    for key, vals in stats.items():
        print(f"  {key}: mean={vals['mean'][:3]}..., std={vals['std'][:3]}...")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="/data/episodes")
    args = parser.parse_args()
    compute_stats(args.data_dir)


if __name__ == "__main__":
    main()
