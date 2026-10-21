"""F6 checkpoint: does the fine-tuned policy predict the right joint targets on demos it never saw?

Runs in the soarm-lerobot container (lerobot 0.6.1), not in Isaac Sim. For every checkpoint
given with ``--policy LABEL=DIR`` it:

1. loads the policy and its saved pre/post-processors (normalization, camera renames),
2. plays every frame of a held-out LeRobot dataset into ``policy.select_action``, exactly as a
   robot would at 30 Hz (open loop: the policy sees the *recorded* images, not its own result),
3. compares each predicted action with the recorded one (mean absolute error per joint).

It prints one ``[F6] PASS`` (or ``[F6] FAIL``) line and writes two pictures under ``--out``:
``F6_loss.png`` (training loss of each run, from its TensorBoard log) and ``F6_open_loop.png``
(predicted vs recorded joint targets for one held-out episode). Usage, from docker/::

    docker compose --profile train-lerobot run --rm lerobot python /tutorials/fasteners/part1/06_check_policy.py \\
        --policy ACT=/workspace/data/checkpoints/act_fastener_m6_10k \\
        --policy SmolVLA=/workspace/data/checkpoints/smolvla_fastener_m6_10k \\
        --dataset /workspace/data/lerobot/fastener_m6_heldout
"""

# [start:args]
import argparse
import json
import os
import sys
import time

import matplotlib

matplotlib.use("Agg")  # no display needed: the pictures go to files
import matplotlib.pyplot as plt
import numpy as np
import torch

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.policies.factory import get_policy_class, make_pre_post_processors

parser = argparse.ArgumentParser(description="F6: check fine-tuned policies on held-out demos.")
parser.add_argument("--policy", action="append", required=True, metavar="LABEL=DIR",
                    help="A training run (data/checkpoints/<job>) or a pretrained_model dir. Repeatable.")
parser.add_argument("--dataset", default="/workspace/data/lerobot/fastener_m6_heldout", help="Held-out dataset root.")
parser.add_argument("--episode", type=int, default=0, help="Episode to draw in F6_open_loop.png.")
parser.add_argument("--episodes", type=int, default=0, help="Only check the first N episodes (0: all).")
parser.add_argument("--n-action-steps", type=int, default=0,
                    help="Re-plan after this many steps of each predicted chunk (0: the trained value).")
parser.add_argument("--sweep", default="", metavar="N,N,...",
                    help="Also measure the error for these re-plan intervals and draw F6_replan.png, e.g. 10,25,50,100.")
parser.add_argument("--max-error", type=float, default=0.1, help="Pass threshold: mean abs error (rad).")
parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
parser.add_argument("--out", default="/workspace/data/tutorials/F6", help="Directory for the pictures.")
args = parser.parse_args()
# [end:args]

JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]


def model_dir(path: str) -> str:
    """Accept a run dir (data/checkpoints/<job>) as well as its pretrained_model dir."""
    last = os.path.join(path, "checkpoints", "last", "pretrained_model")
    return last if os.path.isdir(last) else path


# [start:load]
def load(path: str):
    """Policy plus the processors saved next to it (same normalization and camera names as in training)."""
    with open(os.path.join(path, "config.json")) as f:
        policy_type = json.load(f)["type"]                     # "act", "smolvla", ...
    policy = get_policy_class(policy_type).from_pretrained(path).to(args.device).eval()
    if args.n_action_steps:                                    # how much of each chunk runs before re-planning
        policy.config.n_action_steps = args.n_action_steps
    with open(os.path.join(path, "train_config.json")) as f:
        rename_map = json.load(f).get("rename_map") or {}      # e.g. overhead -> camera1 for SmolVLA
    pre, post = make_pre_post_processors(policy.config, pretrained_path=path, preprocessor_overrides={
        "device_processor": {"device": args.device},
        "rename_observations_processor": {"rename_map": rename_map}})
    return policy, pre, post
# [end:load]


# [start:rollout]
@torch.no_grad()
def predict_episode(policy, pre, post, ds: LeRobotDataset, ep: int) -> tuple[np.ndarray, np.ndarray]:
    """Predicted and recorded actions for every frame of one episode, fed in order like a live robot."""
    start = ds.meta.episodes["dataset_from_index"][ep]
    end = ds.meta.episodes["dataset_to_index"][ep]
    policy.reset()                                             # empty the action queue (chunking)
    predicted, recorded = [], []
    for i in range(start, end):
        frame = ds[i]                                          # images, state, task string, recorded action
        obs = {k: v for k, v in frame.items() if k.startswith("observation.")}
        obs["task"] = frame["task"]
        action = post(policy.select_action(pre(obs)))          # normalized -> radians
        predicted.append(action.squeeze(0).float().cpu().numpy())
        recorded.append(frame["action"].numpy())
    return np.array(predicted), np.array(recorded)
# [end:rollout]


def loss_curve(run_dir: str) -> tuple[np.ndarray, np.ndarray] | None:
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    tb = os.path.join(run_dir, "tensorboard")
    if not os.path.isdir(tb):
        return None
    acc = EventAccumulator(tb)
    acc.Reload()
    if "train/loss" not in acc.Tags().get("scalars", []):
        return None
    ev = acc.Scalars("train/loss")
    return np.array([e.step for e in ev]), np.array([e.value for e in ev])


def main() -> int:
    ds = LeRobotDataset("local/heldout", root=args.dataset)
    print(f"[F6] held-out dataset {args.dataset}: {ds.meta.total_episodes} episodes, "
          f"{ds.meta.total_frames} frames, task {list(ds.meta.tasks.index)}")
    os.makedirs(args.out, exist_ok=True)
    results, curves, drawn, sweep = {}, {}, {}, {}
    for spec in args.policy:
        label, path = spec.split("=", 1)
        policy, pre, post = load(model_dir(path))
        t0, errors = time.time(), []
        for ep in range(args.episodes or ds.meta.total_episodes):
            pred, rec = predict_episode(policy, pre, post, ds, ep)
            errors.append(np.abs(pred - rec))
            if ep == args.episode:
                drawn[label] = (pred, rec)
        err = np.concatenate(errors)                           # (frames, 6) abs error in rad
        results[label] = err.mean(axis=0)
        curve = loss_curve(path)
        if curve is not None:
            curves[label] = curve
        print(f"[F6] {label}: chunk {policy.config.chunk_size} steps, re-plans every "
              f"{policy.config.n_action_steps} steps ({policy.config.n_action_steps / ds.meta.fps:.2f} s)")
        print(f"[F6] {label}: mean abs error {err.mean():.3f} rad ({np.degrees(err.mean()):.1f} deg) over "
              f"{len(err)} frames, {len(err) / (time.time() - t0):.0f} frames/s on {args.device}")
        print(f"[F6] {label} per joint (rad): "
              + ", ".join(f"{j} {e:.3f}" for j, e in zip(JOINTS, results[label])))
        if curve is not None:
            print(f"[F6] {label} training loss: {curve[1][0]:.3f} at step {curve[0][0]} -> "
                  f"{curve[1][-1]:.4f} at step {curve[0][-1]}")
        # [start:sweep]
        for n in [int(v) for v in args.sweep.split(",") if v]:
            if n > policy.config.chunk_size:                  # cannot run more steps than one chunk holds
                print(f"[F6] {label} re-plan every {n} steps: skipped (chunk is {policy.config.chunk_size})")
                continue
            policy.config.n_action_steps = n                  # takes effect at the next policy.reset()
            e = np.concatenate([np.abs(np.subtract(*predict_episode(policy, pre, post, ds, ep)))
                                for ep in range(args.episodes or ds.meta.total_episodes)])
            sweep.setdefault(label, {})[n] = e.mean()
            print(f"[F6] {label} re-plan every {n} steps ({n / ds.meta.fps:.2f} s): mean abs error {e.mean():.3f} rad")
        # [end:sweep]
        del policy
        torch.cuda.empty_cache()

    # [start:plots]
    if curves:
        # one panel per run: ACT (L1 + KL) and SmolVLA (flow matching) use different losses
        fig, axes = plt.subplots(1, len(curves), figsize=(4.6 * len(curves), 3.4), squeeze=False)
        for ax, (label, (steps, loss)), color in zip(axes[0], curves.items(), ("C0", "C1", "C2", "C3")):
            ax.plot(steps, loss, color=color)
            ax.set_yscale("log")
            ax.set_xlabel("training step")
            ax.set_ylabel("training loss (log scale)")
            ax.set_title(f"{label}: {loss[0]:.3g} -> {loss[-1]:.3g}", fontsize=10)
            ax.grid(alpha=0.3)
        fig.suptitle("Training loss on the M6 demos (each policy has its own loss: compare shapes, not heights)",
                     fontsize=10)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "F6_loss.png"), dpi=110)
        plt.close(fig)
    if sweep:
        fig, ax = plt.subplots(figsize=(7, 3.6))
        width = 0.8 / len(sweep)
        ns = sorted({n for errs in sweep.values() for n in errs})
        for k, (label, errs) in enumerate(sweep.items()):
            xs = [ns.index(n) + k * width for n in errs]
            bars = ax.bar(xs, list(errs.values()), width, label=label)
            ax.bar_label(bars, fmt="%.3f", fontsize=8)
        ax.set_xticks(np.arange(len(ns)) + width * (len(sweep) - 1) / 2,
                      [f"every {n} steps\n({n / ds.meta.fps:.2f} s)" for n in ns])
        ax.axhline(args.max_error, color="0.4", ls="--", lw=1)
        ax.text(-0.4, args.max_error, f"pass line {args.max_error} rad", va="bottom", ha="left", fontsize=8)
        ax.set_ylabel("mean abs error (rad)")
        ax.set_title("How often the policy re-plans vs error on held-out demos")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(args.out, "F6_replan.png"), dpi=110)
        plt.close(fig)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True)
    rec = next(iter(drawn.values()))[1]
    t = np.arange(len(rec)) / ds.meta.fps
    for j, ax in enumerate(axes.T.ravel()):
        ax.plot(t, rec[:, j], color="black", lw=2.2, label="recorded (expert)")
        for label, (pred, _) in drawn.items():
            ax.plot(t, pred[:, j], lw=1.4, label=f"{label} prediction")
        ax.set_title(JOINTS[j], fontsize=10, loc="left")
        ax.set_ylabel("rad")
        ax.grid(alpha=0.3)
    axes[0, 0].legend(fontsize=8)
    for ax in axes[-1]:
        ax.set_xlabel("time (s)")
    fig.suptitle(f"{os.path.basename(args.dataset.rstrip('/'))} episode {args.episode}: "
                 "joint targets the policy predicts vs the expert's", fontsize=11)
    fig.tight_layout()
    fig.savefig(os.path.join(args.out, "F6_open_loop.png"), dpi=110)
    plt.close(fig)
    # [end:plots]
    print(f"[F6] wrote {args.out}/F6_loss.png and {args.out}/F6_open_loop.png")

    best = min(results, key=lambda k: results[k].mean())
    ok = all(r.mean() <= args.max_error for r in results.values())
    print(f"[F6] {'PASS' if ok else 'FAIL'} "
          + " ".join(f"{k}={v.mean():.3f}rad" for k, v in results.items())
          + f" (need <= {args.max_error}); lowest error: {best}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
