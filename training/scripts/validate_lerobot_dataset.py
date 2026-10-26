"""Validate a LeRobot v3.0 dataset made by ``hdf5_to_lerobot.py`` against the contract and its HDF5.

Checks, each printed as PASS/FAIL:

* ``LeRobotDataset(repo_id, root=...)`` loads it (the pinned lerobot, 0.6.1)
* codebase version v3.0, fps = contract DATASET_FPS, robot_type
* camera keys = the contract's ``observation.images.*``, 224x224x3 video features
* ``observation.state`` / ``action``: float32, shape (6,), contract joint names
* stats (mean/std/min/max/count) present for state, action and every camera
* episode count (and, with ``meta/soarm_sources.json``, frames and task string per episode)
* state and action columns equal the HDF5 ``obs/joint_pos`` / ``joint_pos_targets`` exactly
* N sampled frames decoded from the videos match the HDF5 images: mean absolute pixel error
  (0-255 scale) below ``--max-mae``, and, where the scene moved between steps, each decoded frame
  is closer to its own HDF5 frame than to the one before or after (no off-by-one in time)

Video is lossy (AV1, CRF 30 by default), so frames are not bit-exact; see ``--max-mae``.

Run it in the LeRobot image, after the conversion::

    cd docker
    docker compose --profile train-lerobot run --rm -e NVIDIA_VISIBLE_DEVICES=void lerobot \\
        python /training/scripts/validate_lerobot_dataset.py /workspace/data/lerobot/fastener_m6 \\
            --hdf5-dir /workspace/data/demos/fastener_m6 --expect-episodes 51
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hdf5_to_lerobot import C, HDF5_TO_LEROBOT, S, SOURCES_FILE  # noqa: E402

# Mean absolute error (0-255) allowed per decoded frame. At LeRobot's default AV1 CRF 30 the
# 224x224 fastener frames come back with a per-frame MAE of 1.1-2.5 (51-episode M6 set), so 4
# leaves ~1.6x headroom. Gross mistakes land far above it: the other camera 30-89, swapped colour
# channels 11-47. A wrong episode or a time shift can stay below it while the scene is still (the
# scripted episodes start alike), which is why the alignment check below exists.
DEFAULT_MAX_MAE = 4.0


class Report:
    def __init__(self):
        self.failures = 0

    def check(self, ok: bool, what: str, detail: str = ""):
        self.failures += not ok
        print(f"[{'PASS' if ok else 'FAIL'}] {what}{(': ' + detail) if detail else ''}", flush=True)
        return ok


def to_uint8_hwc(img) -> np.ndarray:
    """LeRobot returns video frames as float CHW in [0, 1]; the HDF5 stores uint8 HWC."""
    arr = img.numpy() if hasattr(img, "numpy") else np.asarray(img)
    if arr.dtype != np.uint8:
        arr = np.clip(np.rint(arr * 255.0), 0, 255).astype(np.uint8)
    return arr.transpose(1, 2, 0) if arr.shape[0] == 3 else arr


def mae(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.abs(a.astype(np.int16) - b.astype(np.int16)).mean())


def validate(root: str, repo_id: str, hdf5_dir: str | None, expect_episodes: int | None,
             num_samples: int, max_mae: float, seed: int) -> int:
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    r = Report()
    root_p = Path(root)
    # [start:load]
    ds = LeRobotDataset(repo_id, root=root_p)
    meta = ds.meta
    # [end:load]
    r.check(True, "LeRobotDataset loads", f"{meta.total_episodes} episodes, {meta.total_frames} frames, "
                                          f"{len(ds)} samples")
    r.check(meta.info.get("codebase_version") == "v3.0", "codebase_version", str(meta.info.get("codebase_version")))
    r.check(meta.fps == C.DATASET_FPS, "fps", f"{meta.fps} (contract {C.DATASET_FPS})")
    r.check(meta.robot_type == C.ROBOT_TYPE, "robot_type", str(meta.robot_type))

    # [start:keys]
    want_cams = sorted(cam.dataset_key for cam in C.CAMERAS)
    r.check(sorted(meta.video_keys) == want_cams, "camera keys", f"{sorted(meta.video_keys)}")
    for cam in C.CAMERAS:
        feat = meta.features.get(cam.dataset_key, {})
        r.check(feat.get("dtype") == "video" and tuple(feat.get("shape", ())) == (cam.height, cam.width, 3),
                f"{cam.dataset_key} feature", f"{feat.get('dtype')} {feat.get('shape')} "
                                              f"{feat.get('info', {}).get('video.codec')}")
    for key, dim in ((C.DATASET_STATE_KEY, C.NUM_JOINTS), (C.DATASET_ACTION_KEY, C.ACTION_DIM)):
        feat = meta.features.get(key, {})
        r.check(feat.get("dtype") == "float32" and tuple(feat.get("shape", ())) == (dim,)
                and list(feat.get("names", [])) == list(C.JOINT_NAMES),
                f"{key} feature", f"{feat.get('dtype')} {feat.get('shape')} {feat.get('names')}")
    # [end:keys]

    stats = meta.stats or {}
    for key in (C.DATASET_STATE_KEY, C.DATASET_ACTION_KEY, *want_cams):
        missing = [s for s in ("mean", "std", "min", "max", "count") if s not in stats.get(key, {})]
        r.check(not missing, f"stats {key}", f"missing {missing}" if missing else
                f"mean {np.round(np.asarray(stats[key]['mean']).ravel()[:6], 3).tolist()}")

    if expect_episodes is not None:
        r.check(meta.total_episodes == expect_episodes, "episode count",
                f"{meta.total_episodes} (expected {expect_episodes})")

    sources_path = root_p / SOURCES_FILE
    if not sources_path.is_file():
        r.check(False, SOURCES_FILE, "missing; cannot compare with the HDF5")
        return r.failures
    sources = json.loads(sources_path.read_text())
    r.check(len(sources) == meta.total_episodes, f"{SOURCES_FILE} episodes", str(len(sources)))

    # Per-episode length and task string. With --max-idle-frames the converter kept only some HDF5
    # frames and listed them as "kept_frames"; dataset frame t is HDF5 frame kept_frames[t].
    kept = {src["episode_index"]: np.asarray(src.get("kept_frames", range(src["num_samples"]))) for src in sources}
    bad_len, bad_task = [], []
    for src in sources:
        ep = meta.episodes[src["episode_index"]]
        if int(ep["length"]) != len(kept[src["episode_index"]]):
            bad_len.append(src["episode_index"])
        if list(ep["tasks"]) != [src["prompt"]]:
            bad_task.append(src["episode_index"])
    lengths = [len(k) for k in kept.values()]
    trimmed = sum(s["num_samples"] for s in sources) - sum(lengths)
    r.check(not bad_len, "episode lengths = HDF5 frames kept", f"mismatch in {bad_len}" if bad_len else
            f"{min(lengths)}-{max(lengths)} frames" + (f" ({trimmed} idle frames trimmed)" if trimmed else ""))
    r.check(not bad_task, "one task per episode = HDF5 prompt", f"mismatch in {bad_task}" if bad_task else
            f"{len(meta.tasks)} distinct task(s)")

    if hdf5_dir is None:
        print("(no --hdf5-dir: skipping the comparison with the HDF5 data)")
        return r.failures

    # State/action: compare the whole parquet columns with the HDF5, every episode.
    hf = ds.hf_dataset.with_format("numpy")
    ep_col = np.asarray(hf["episode_index"])
    state_col = np.stack(hf[C.DATASET_STATE_KEY])
    action_col = np.stack(hf[C.DATASET_ACTION_KEY])
    worst = 0.0
    for src in sources:
        h5 = str(Path(hdf5_dir) / src["file"])
        arrays, _ = S.load_episode(h5, src["demo"], keys=(S.JOINT_POS, S.JOINT_POS_TARGETS))
        rows, idx = ep_col == src["episode_index"], kept[src["episode_index"]]
        worst = max(worst, float(np.abs(state_col[rows] - arrays[S.JOINT_POS][idx].astype(np.float32)).max()),
                    float(np.abs(action_col[rows] - arrays[S.JOINT_POS_TARGETS][idx].astype(np.float32)).max()))
    r.check(worst == 0.0, "state/action = HDF5 joint_pos / joint_pos_targets", f"max abs diff {worst:g} rad")

    # [start:frames]
    # Video frames: decode N random (episode, t) samples through the dataset and compare.
    rng = np.random.default_rng(seed)
    picks = [(src, int(rng.integers(len(kept[src["episode_index"]]))))
             for src in (sources[i] for i in rng.integers(len(sources), size=num_samples))]
    errors = {cam.dataset_key: [] for cam in C.CAMERAS}
    misaligned, judged = [], 0
    for src, t in picks:
        h5 = str(Path(hdf5_dir) / src["file"])
        arrays, _ = S.load_episode(h5, src["demo"], keys=tuple(f"obs/{c.obs_key}" for c in C.CAMERAS))
        item = ds[int(meta.episodes[src["episode_index"]]["dataset_from_index"]) + t]
        assert int(item["frame_index"]) == t and int(item["episode_index"]) == src["episode_index"]
        h = int(kept[src["episode_index"]][t])                # the HDF5 frame behind dataset frame t
        for cam in C.CAMERAS:
            ref = arrays[f"obs/{cam.obs_key}"]
            frame = to_uint8_hwc(item[cam.dataset_key])
            err = mae(frame, ref[h])
            errors[cam.dataset_key].append(err)
            # Time alignment, judged only where the scene moved between steps by clearly more than
            # the codec error (in still phases neighbouring frames are equally close).
            for u in (h - 1, h + 1):
                if 0 <= u < len(ref) and mae(ref[u], ref[h]) > 2 * err + 1:
                    judged += 1
                    if mae(frame, ref[u]) <= err:
                        misaligned.append((src["episode_index"], t, u, cam.name))
    # [end:frames]
    for key, errs in errors.items():
        errs = np.asarray(errs)
        r.check(errs.max() <= max_mae, f"{key} decoded vs HDF5 ({len(errs)} frames)",
                f"MAE mean {errs.mean():.2f}, median {np.median(errs):.2f}, max {errs.max():.2f} "
                f"(0-255 scale, tolerance {max_mae})")
    r.check(not misaligned, "decoded frame t is closest to HDF5 frame t (not t-1 / t+1)",
            f"{misaligned[:5]}" if misaligned else f"{judged} comparisons with a moving scene")
    return r.failures


def main():
    parser = argparse.ArgumentParser(description="Validate a LeRobot dataset converted from fastener demo HDF5.")
    parser.add_argument("root", help="dataset directory, e.g. /workspace/data/lerobot/fastener_m6")
    parser.add_argument("--repo-id", default=None, help="default local/<root dir name>")
    parser.add_argument("--hdf5-dir", default=None, help="directory with the source HDF5 files (for the comparison)")
    parser.add_argument("--expect-episodes", type=int, default=None)
    parser.add_argument("--num-samples", type=int, default=32, help="frames to decode and compare (default 32)")
    parser.add_argument("--max-mae", type=float, default=DEFAULT_MAX_MAE,
                        help=f"max mean abs pixel error per frame, 0-255 (default {DEFAULT_MAX_MAE})")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    failures = validate(args.root, args.repo_id or f"local/{Path(args.root).name}", args.hdf5_dir,
                        args.expect_episodes, args.num_samples, args.max_mae, args.seed)
    print(f"\n{'OK' if not failures else f'{failures} check(s) FAILED'}")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
