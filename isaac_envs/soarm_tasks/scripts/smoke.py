"""Reset + step smoke test and throughput / VRAM numbers for the Soarm fastener tasks.

Builds the task with ``--num_envs`` envs, resets, then steps it ``--steps`` times holding the
robot's reset joint targets (JointAbs/State) or its reset EE pose (IK-Abs), and prints env-steps
per second and GPU memory. Run in the isaac-sim container::

    /isaac-sim/python.sh /isaac_envs/soarm_tasks/scripts/smoke.py --task Soarm-PickPlace-Fastener-JointAbs-v0 \
        --num_envs 16 [--no-cameras]

GPU memory: ``device_used`` is the whole GPU's used memory (nvidia-smi, includes other
processes) at its peak during the run minus before Isaac started; ``torch_peak`` is PyTorch's
own allocations only (not PhysX / RTX).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time


def gpu_used_mib() -> int:
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits", "-i", "0"],
                             capture_output=True, text=True, check=True).stdout
        return int(out.strip().splitlines()[0])
    except Exception:  # noqa: BLE001
        return -1


BASELINE_MIB = gpu_used_mib()

from isaaclab.app import add_launcher_args, launch_simulation  # noqa: E402

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="Soarm-PickPlace-Fastener-JointAbs-v0")
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--steps", type=int, default=300)
parser.add_argument("--no-cameras", action="store_true")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_cameras and "State" not in args.task
args.require_kit = True
sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))


def main():
    with launch_simulation(None, args):
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        import soarm_tasks  # noqa: F401
        from soarm_tasks.mdp import language_instructions
        from soarm_tasks.pick_place_env_cfg import cameras_enabled

        cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        cfg.seed = 0
        if args.no_cameras:
            cfg.disable_cameras()
        t0 = time.time()
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        t_build = time.time() - t0
        t0 = time.time()
        obs, _ = env.reset(seed=0)
        t_reset = time.time() - t0
        print(f"[smoke] {args.task} num_envs={u.num_envs} cameras={cameras_enabled(cfg)} "
              f"action_dim={u.action_manager.total_action_dim} build {t_build:.1f}s reset {t_reset:.2f}s", flush=True)
        print(f"[smoke] obs groups: { {g: (list(v) if isinstance(v, dict) else tuple(v.shape)) for g, v in obs.items()} }",
              flush=True)
        if "images" in obs:
            for k, v in obs["images"].items():
                print(f"[smoke] image {k}: {tuple(v.shape)} {v.dtype} mean {v.float().mean():.1f}", flush=True)
        print(f"[smoke] prompts: {sorted(set(language_instructions(u)))}", flush=True)

        robot = u.scene["robot"]
        if "IK" in args.task:
            ee_id = robot.find_bodies("gripper_frame_link")[0][0]
            from isaaclab.utils.math import subtract_frame_transforms
            root = robot.data.root_link_pose_w.torch
            ee = robot.data.body_link_pose_w.torch[:, ee_id]
            pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], ee[:, :3], ee[:, 3:7])
            action = torch.cat([pos, quat, robot.data.joint_pos.torch[:, -1:]], dim=-1)
        else:
            action = robot.data.joint_pos.torch.clone()
        torch.cuda.synchronize()
        peak = gpu_used_mib()
        resets = 0
        t0 = time.time()
        for i in range(args.steps):
            obs, _, term, trunc, _ = env.step(action)
            resets += int((term | trunc).sum())
            if i % 50 == 0:
                peak = max(peak, gpu_used_mib())
        torch.cuda.synchronize()
        dt = time.time() - t0
        peak = max(peak, gpu_used_mib())
        print(f"[smoke] RESULT task={args.task} num_envs={u.num_envs} cameras={cameras_enabled(cfg)} "
              f"steps={args.steps} {args.steps / dt:.1f} steps/s = {args.steps * u.num_envs / dt:.0f} env-steps/s "
              f"(real-time factor {args.steps / dt / 30:.2f} at 30 Hz); resets during run {resets}; "
              f"device_used peak {peak} MiB (+{peak - BASELINE_MIB} MiB over {BASELINE_MIB} before); "
              f"torch_peak {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB", flush=True)
        env.close()


if __name__ == "__main__":
    main()
