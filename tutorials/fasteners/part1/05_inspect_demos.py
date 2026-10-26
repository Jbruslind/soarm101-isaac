"""F5 checkpoint: inspect a fastener demo file and draw what one demonstration looks like.

No simulator and no GPU: it reads the HDF5 file that ``generate_demos.py`` or Isaac Lab's
``record_demos.py`` wrote, checks it with :mod:`soarm_tasks.demo_schema`, prints one
``[F5] PASS`` (or ``[F5] FAIL``) line, and saves two pictures of one demo under ``--out``:

* ``F5_<name>_frames.png``: the overhead and wrist camera at the six key moments of the demo
  (start, jaws close, lifted, over the bin, release, end).
* ``F5_<name>_timeline.png``: EE height, bolt height and gripper angle over time, with the same
  moments marked.

Run it in the isaac-sim container (it only needs ``h5py``, ``numpy`` and ``matplotlib``)::

    ./scripts/isaac.sh --headless /isaac-sim/python.sh /tutorials/fasteners/part1/05_inspect_demos.py \\
        --file /data/demos/fastener_m6/scripted.hdf5 --min-demos 50
"""

# [start:args]
import argparse
import os
import sys

import h5py
import matplotlib

matplotlib.use("Agg")  # no display needed: the pictures go to files
import matplotlib.pyplot as plt
import numpy as np

parser = argparse.ArgumentParser(description="F5: check a demo file and draw one demonstration.")
parser.add_argument("--file", default="/data/demos/fastener_m6/scripted.hdf5", help="Demo HDF5 file.")
parser.add_argument("--demo", type=int, default=0, help="Index of the demo to draw (demo_<k>).")
parser.add_argument("--min-demos", type=int, default=1, help="Pass threshold: successful demos in the file.")
parser.add_argument("--out", default="/data/tutorials/F5", help="Directory for the pictures.")
args = parser.parse_args()

# The repository's environments are mounted read-only at /isaac_envs in the isaac-sim container.
sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))
from soarm_tasks import demo_schema as S  # noqa: E402
from soarm_tasks.work_cell import BIN  # noqa: E402
# [end:args]

HZ = 30.0  # control rate of the contract (soarm_common.contract.CONTROL_HZ)


# [start:moments]
def key_moments(g: h5py.Group) -> dict[str, int]:
    """Step index of the six moments every pick-and-place demo goes through."""
    grip_target = g["ee_pose_actions"][:, 7]          # rad: > 0.5 open, < 0 close
    bolt = g["obs/fastener_pose"][:, :3]              # bolt position in the robot base frame (m)
    closed = grip_target < 0.5
    t_lift = int(np.argmax(bolt[:, 2] > 0.03))        # bolt 3 cm above the table
    starts = np.flatnonzero(closed[1:t_lift + 1] & ~closed[:t_lift]) + 1
    t_close = int(starts[-1]) if len(starts) else int(np.argmax(closed))  # the close that lifted it
    near_bin = np.all(np.abs(bolt[:, :2] - np.array(BIN.center)) < np.array(BIN.inner_size) / 2, axis=1)
    t_over = int(np.argmax(near_bin & (bolt[:, 2] > 0.03)))
    t_release = t_lift + int(np.argmax(~closed[t_lift:]))  # first "open" command after the lift
    return {"start": 0, "jaws close": t_close, "lifted": t_lift, "over the bin": t_over,
            "release": t_release, "end": len(grip_target) - 1}
# [end:moments]


def draw_frames(g: h5py.Group, moments: dict[str, int], title: str, path: str) -> None:
    fig, axes = plt.subplots(2, len(moments), figsize=(2.2 * len(moments), 4.9))
    for col, (label, t) in enumerate(moments.items()):
        for row, key in enumerate(("obs/overhead_rgb", "obs/wrist_rgb")):
            ax = axes[row, col]
            ax.imshow(g[key][t])
            ax.set_xticks([])
            ax.set_yticks([])
            if row == 0:
                ax.set_title(f"{label}\nt = {t / HZ:.1f} s", fontsize=10)
            if col == 0:
                ax.set_ylabel(("overhead", "wrist")[row], fontsize=10)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def draw_timeline(g: h5py.Group, moments: dict[str, int], title: str, path: str) -> None:
    t = np.arange(g["obs/ee_pose"].shape[0]) / HZ
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10.5, 4.8), sharex=True)
    ax1.plot(t, 100 * g["obs/ee_pose"][:, 2], label="gripper (EE) height")
    ax1.plot(t, 100 * g["obs/fastener_pose"][:, 2], label="bolt height")
    ax1.set_ylabel("height above table (cm)")
    ax1.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    ax2.plot(t, g["ee_pose_actions"][:, 7], "--", label="gripper command")
    ax2.plot(t, g["obs/gripper_pos"][:, 0], label="gripper angle")
    ax2.set_ylabel("gripper (rad)")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=9)
    for label, step in moments.items():
        for ax in (ax1, ax2):
            ax.axvline(step / HZ, color="0.6", lw=0.8)
        ax1.annotate(label, (step / HZ, ax1.get_ylim()[1]), rotation=90, va="top", ha="right", fontsize=8)
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main() -> int:
    # [start:check]
    problems = S.validate(args.file)                   # layout, dims, images, attributes
    info = S.file_info(args.file)
    with h5py.File(args.file, "r") as f:
        demos = [f["data"][n] for n in S.episode_names(args.file)]
        successes = info["successes"]
        lengths = np.array([g.attrs["num_samples"] for g in demos])
        prompts = sorted({str(g.attrs.get("prompt", "")) for g in demos})
        print(f"[F5] file={args.file} env={info['env_name']} demos={info['demos']} successes={successes} "
              f"sources={info['sources']} size={info['bytes'] / 1e6:.0f} MB")
        print(f"[F5] episode length: mean {lengths.mean() / HZ:.1f} s, min {lengths.min() / HZ:.1f} s, "
              f"max {lengths.max() / HZ:.1f} s; prompts: {prompts}")
    # [end:check]

        g = demos[args.demo]
        moments = key_moments(g)
        print("[F5] demo_%d moments (s): " % args.demo
              + ", ".join(f"{k} {v / HZ:.1f}" for k, v in moments.items()))
        os.makedirs(args.out, exist_ok=True)
        name = os.path.splitext(os.path.basename(args.file))[0]
        title = f"{name} demo_{args.demo}: \"{g.attrs.get('prompt', '')}\" (source: {g.attrs.get('source', '?')})"
        frames = os.path.join(args.out, f"F5_{name}_frames.png")
        timeline = os.path.join(args.out, f"F5_{name}_timeline.png")
        draw_frames(g, moments, title, frames)
        draw_timeline(g, moments, title, timeline)
        print(f"[F5] wrote {frames}")
        print(f"[F5] wrote {timeline}")

    ok = not problems and successes >= args.min_demos
    for p in problems[:10]:
        print(f"[F5] schema problem: {p}")
    print(f"[F5] {'PASS' if ok else 'FAIL'} successes={successes} (need >= {args.min_demos}) "
          f"schema_problems={len(problems)}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
