"""Convert fastener demo HDF5 files (Isaac Lab ``record_demos`` format) to a LeRobot v3.0 dataset.

Each ``demo_<k>`` becomes one LeRobot episode, written with the LeRobot API
(``LeRobotDataset.create`` / ``add_frame`` / ``save_episode`` / ``finalize``), so the output is
whatever the pinned lerobot (0.6.1) itself writes: chunked parquet, one MP4 per camera per chunk,
``meta/info.json``, ``meta/stats.json``, ``meta/tasks.parquet``, ``meta/episodes/``. The overall
flow (one pass over the demos, keep successful ones) follows Isaac Lab's
``scripts/imitation_learning/locomanipulation_sdg/gr00t/convert_dataset.py``, which writes the
files by hand instead.

Mapping (``soarm_common.contract``; all angles in radians, contract joint order)::

    HDF5 (soarm_tasks.demo_schema)        LeRobot feature
    joint_pos_targets (T, 6) float32      action                         absolute joint targets
    obs/joint_pos     (T, 6) float32      observation.state              joint positions before the action
    obs/wrist_rgb     (T, 224, 224, 3)    observation.images.wrist       video
    obs/overhead_rgb  (T, 224, 224, 3)    observation.images.overhead    video
    demo attr "prompt"                    task (one per episode)
    fps                                   contract DATASET_FPS (30, one frame per control step)

Which episode came from which file / demo is written to ``meta/soarm_sources.json``; the
validator (``validate_lerobot_dataset.py``) uses it to compare frames with the HDF5.

Run it in the LeRobot image (it has h5py and mounts isaac_envs/ and training/ read-only)::

    cd docker
    docker compose --profile train-lerobot run --rm -e NVIDIA_VISIBLE_DEVICES=void lerobot \\
        python /training/scripts/hdf5_to_lerobot.py \\
            /workspace/data/demos/fastener_m6/scripted.hdf5 \\
            /workspace/data/demos/fastener_m6/teleop_selftest2.hdf5 \\
            --output-root /workspace/data/lerobot/fastener_m6 --repo-id local/fastener_m6
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

import numpy as np


def _import_schema():
    """``soarm_tasks.demo_schema`` and ``soarm_common.contract``: from PYTHONPATH, /isaac_envs (the
    docker compose mount) or the repo's isaac_envs/ in a source checkout."""
    here = Path(os.path.realpath(__file__)).parent
    for cand in ("/isaac_envs", str(here.parents[1] / "isaac_envs")):
        if (Path(cand) / "soarm_tasks" / "demo_schema.py").is_file() and cand not in sys.path:
            sys.path.insert(0, cand)
            break
    from soarm_common import contract
    from soarm_tasks import demo_schema

    return demo_schema, contract


S, C = _import_schema()

# Provenance file next to LeRobot's own metadata (LeRobot ignores it).
SOURCES_FILE = "meta/soarm_sources.json"


# [start:features]
def lerobot_features() -> dict:
    """LeRobot feature spec: 6-D state and action (rad, contract joint order), one video per
    contract camera. LeRobot adds timestamp, frame_index, episode_index, index and task_index."""
    joints = {"names": list(C.JOINT_NAMES)}
    features = {
        C.DATASET_STATE_KEY: {"dtype": "float32", "shape": (C.NUM_JOINTS,), **joints},
        C.DATASET_ACTION_KEY: {"dtype": "float32", "shape": (C.ACTION_DIM,), **joints},
    }
    for cam in C.CAMERAS:  # observation.images.wrist, observation.images.overhead
        features[cam.dataset_key] = {
            "dtype": "video",
            "shape": (cam.height, cam.width, 3),
            "names": ["height", "width", "channels"],
        }
    return features


# HDF5 dataset (demo_schema key) -> LeRobot feature
HDF5_TO_LEROBOT = {
    S.JOINT_POS: C.DATASET_STATE_KEY,
    S.JOINT_POS_TARGETS: C.DATASET_ACTION_KEY,
    **{f"obs/{cam.obs_key}": cam.dataset_key for cam in C.CAMERAS},
}
# [end:features]


def check_file(path: str) -> list[str]:
    """What the conversion needs of a demo file: the contract joint order and, per demo, the
    datasets in HDF5_TO_LEROBOT with matching lengths, 6-D state/action, uint8 224x224 images and
    a prompt. (``demo_schema.validate`` checks the full schema, e.g. for replay; files from the
    legacy ``sim_data_collector.py`` have no ``initial_state`` and pass only this check.)"""
    import h5py

    problems = []
    with h5py.File(path, "r") as f:
        data = f.get("data")
        if data is None:
            return ["no /data group"]
        if json.loads(data.attrs.get("soarm_joint_names", "[]")) != list(C.JOINT_NAMES):
            problems.append(f"soarm_joint_names differ from the contract {list(C.JOINT_NAMES)}")
        for name in data:
            demo = data[name]
            missing = [k for k in HDF5_TO_LEROBOT if k not in demo]
            if missing or "prompt" not in demo.attrs:
                problems.append(f"{name}: missing {missing + ([] if 'prompt' in demo.attrs else ['attr prompt'])}")
                continue
            lengths = {demo[k].shape[0] for k in HDF5_TO_LEROBOT}
            if len(lengths) != 1:
                problems.append(f"{name}: datasets have different lengths {sorted(lengths)}")
            for k in (S.JOINT_POS, S.JOINT_POS_TARGETS):
                if demo[k].shape[1:] != (C.NUM_JOINTS,):
                    problems.append(f"{name}: {k} shape {demo[k].shape}")
            for cam in C.CAMERAS:
                img = demo[f"obs/{cam.obs_key}"]
                if img.dtype != np.uint8 or img.shape[1:] != (cam.height, cam.width, 3):
                    problems.append(f"{name}: obs/{cam.obs_key} is {img.dtype} {img.shape[1:]}")
    return problems


def select_demos(paths: list[str], sources: set[str] | None, include_failed: bool) -> list[dict]:
    """(file, demo, attrs) of every demo to convert, in file order then demo index order."""
    import h5py

    picked = []
    for path in paths:
        with h5py.File(path, "r") as f:
            for name in S.episode_names(path):
                attrs = dict(f["data"][name].attrs)
                source = str(attrs.get("source", ""))
                if sources and source not in sources:
                    continue
                if not include_failed and not bool(attrs.get("success", False)):
                    continue
                picked.append({"file": path, "demo": name, "source": source,
                               "prompt": str(attrs["prompt"]), "num_samples": int(attrs["num_samples"]),
                               "seed": int(attrs.get("seed", -1)), "fastener": str(attrs.get("fastener", "")),
                               "success": bool(attrs.get("success", False))})
    return picked


def convert(paths: list[str], output_root: str, repo_id: str, sources: set[str] | None = None,
            include_failed: bool = False, vcodec: str | None = None, overwrite: bool = False,
            max_idle: int = -1) -> dict:
    """Write the selected demos of ``paths`` as a LeRobot v3.0 dataset at ``output_root``."""
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    demos = select_demos(paths, sources, include_failed)
    if not demos:
        raise SystemExit("no demos selected (check --source / --include-failed)")
    out = Path(output_root)
    if out.exists():
        if not overwrite:
            raise SystemExit(f"{out} exists; pass --overwrite to replace it")
        shutil.rmtree(out)

    encoder = {}
    if vcodec:
        from lerobot.configs.video import RGBEncoderConfig

        encoder = {"rgb_encoder": RGBEncoderConfig(vcodec=vcodec)}

    # [start:create]
    dataset = LeRobotDataset.create(
        repo_id=repo_id,
        root=out,
        fps=C.DATASET_FPS,             # 30: one frame per control step
        robot_type=C.ROBOT_TYPE,       # "so_arm101"
        features=lerobot_features(),
        use_videos=True,
        **encoder,
    )
    # [end:create]

    t0 = time.time()
    for ep_index, demo in enumerate(demos):
        arrays, _ = S.load_episode(demo["file"], demo["demo"], keys=tuple(HDF5_TO_LEROBOT))
        # [start:frames]
        keep = keep_frames(arrays[S.JOINT_POS_TARGETS], arrays[S.JOINT_POS], max_idle)
        num_frames = int(keep.sum())
        for t in np.flatnonzero(keep):
            frame = {feature: arrays[key][t] for key, feature in HDF5_TO_LEROBOT.items()}
            frame[C.DATASET_STATE_KEY] = frame[C.DATASET_STATE_KEY].astype(np.float32)
            frame[C.DATASET_ACTION_KEY] = frame[C.DATASET_ACTION_KEY].astype(np.float32)
            frame["task"] = demo["prompt"]  # one task string per episode
            dataset.add_frame(frame)
        dataset.save_episode()  # encodes the videos, writes parquet + episode metadata
        # [end:frames]
        demo["episode_index"] = ep_index
        if max_idle >= 0:
            demo["kept_frames"] = np.flatnonzero(keep).tolist()  # for the validator
        print(f"episode {ep_index:3d}  {Path(demo['file']).name}:{demo['demo']:<8s} {demo['source']:<9s} "
              f"{num_frames:4d} frames  ({time.time() - t0:6.1f} s)", flush=True)
    dataset.finalize()  # flushes parquet footers; the dataset is invalid without it
    shutil.rmtree(out / "images", ignore_errors=True)  # LeRobot's (now empty) frame staging dirs

    (out / SOURCES_FILE).write_text(json.dumps(
        [{**d, "file": Path(d["file"]).name} for d in demos], indent=1))
    summary = {
        "root": str(out), "repo_id": repo_id, "episodes": dataset.meta.total_episodes,
        "frames": dataset.meta.total_frames, "fps": dataset.meta.fps,
        "tasks": sorted({d["prompt"] for d in demos}),
        "sources": {s: sum(d["source"] == s for d in demos) for s in sorted({d["source"] for d in demos})},
        "max_idle_frames": max_idle,
        "seconds": round(time.time() - t0, 1),
    }
    return summary


# [start:idle]
def keep_frames(actions: np.ndarray, states: np.ndarray, max_idle: int, eps: float = 0.005) -> np.ndarray:
    """Which frames to keep when idle stretches are trimmed (``--max-idle-frames``).

    A frame is idle when neither the joint targets nor the joint angles moved more than ``eps``
    rad since the previous frame, e.g. while the expert hovers and waits to line up. In such a
    stretch the camera images hardly change, yet the label switches from "stay" to "go down" at
    its end, so a policy cannot tell when to go and, closed-loop, hovers for ever. Keep only the
    first ``max_idle`` frames of every idle stretch (all frames when ``max_idle`` < 0).
    """
    keep = np.ones(len(actions), dtype=bool)
    if max_idle < 0:
        return keep
    run = 0
    for t in range(1, len(actions)):
        idle = np.abs(actions[t] - actions[t - 1]).max() < eps and np.abs(states[t] - states[t - 1]).max() < eps
        run = run + 1 if idle else 0
        keep[t] = run <= max_idle
    return keep
# [end:idle]


def main():
    parser = argparse.ArgumentParser(description="Convert fastener demo HDF5 files to a LeRobot v3.0 dataset.")
    parser.add_argument("inputs", nargs="+", help="demo HDF5 file(s), e.g. data/demos/fastener_m6/scripted.hdf5")
    parser.add_argument("--output-root", required=True, help="dataset directory to create, e.g. data/lerobot/fastener_m6")
    parser.add_argument("--repo-id", default=None, help="LeRobot repo_id (default local/<output dir name>)")
    parser.add_argument("--source", action="append", default=None,
                        help="only demos with this 'source' attr (scripted, teleop, ...); repeatable; default all")
    parser.add_argument("--include-failed", action="store_true", help="also convert demos with success=False")
    parser.add_argument("--vcodec", default=None,
                        help="video codec (default: lerobot's, libsvtav1); e.g. h264 for faster decoding elsewhere")
    parser.add_argument("--overwrite", action="store_true", help="replace an existing output directory")
    parser.add_argument("--max-idle-frames", type=int, default=-1,
                        help="keep only the first N frames of every idle stretch (arm and targets still); "
                             "-1 keeps all. 2 is a good choice for closed-loop policies (see F7)")
    args = parser.parse_args()

    for path in args.inputs:
        problems = check_file(path)
        if problems:
            raise SystemExit(f"{path} cannot be converted:\n  " + "\n  ".join(problems[:10]))
    repo_id = args.repo_id or f"local/{Path(args.output_root).name}"
    summary = convert(args.inputs, args.output_root, repo_id, set(args.source) if args.source else None,
                      args.include_failed, args.vcodec, args.overwrite, args.max_idle_frames)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
