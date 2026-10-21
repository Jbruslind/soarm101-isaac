"""Evaluate the scripted IK expert on Soarm-PickPlace-Fastener-IK-Abs-v0 over N seeded episodes.

Prints one line per episode, then the success rate with a 95 % Wilson interval and the
per-stage rates (grasped / lifted / in_bin reached at any step of the episode), plus throughput
and GPU memory. Run in the isaac-sim container (from docker/):

    docker compose --profile collect run --rm isaac-sim /isaac-sim/python.sh \
        /isaac_envs/soarm_tasks/scripts/scripted_eval.py --episodes 20 --num_envs 4 --seed 0

``--no-cameras`` skips rendering (faster); ``--json PATH`` writes the summary.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="Soarm-PickPlace-Fastener-IK-Abs-v0")
parser.add_argument("--episodes", type=int, default=20)
parser.add_argument("--num_envs", type=int, default=1)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--no-cameras", action="store_true", help="drop both cameras (state-only)")
parser.add_argument("--fastener", default=None, help="fastener name, e.g. hex_bolt_M6x20 (default: task's)")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override ExpertParams fields, e.g. grasp_z=0.006 closed=-0.17")
parser.add_argument("--json", default=None, help="write the summary as JSON here")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_cameras
args.require_kit = True
sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))


# [start:wilson]
def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """95 % Wilson score interval of a success rate k / n."""
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)
# [end:wilson]


def main():
    with launch_simulation(None, args):
        import gymnasium as gym
        import torch
        from isaaclab_tasks.utils import parse_env_cfg

        import soarm_tasks  # noqa: F401
        from soarm_tasks.expert import PHASE_NAMES, ExpertParams, FastenerPickExpert
        from soarm_tasks.work_cell import BIN

        cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        cfg.seed = args.seed
        if args.fastener:
            cfg.fastener_names = (args.fastener,)
            cfg.__post_init__()
        if args.no_cameras:
            cfg.disable_cameras()
        params = ExpertParams()
        for kv in args.set:
            k, v = kv.split("=")
            setattr(params, k, type(getattr(params, k))(float(v)))
        print(f"[eval] task={args.task} num_envs={args.num_envs} seed={args.seed} "
              f"cameras={not args.no_cameras} params={params}", flush=True)

        t0 = time.time()
        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        print(f"[eval] env built in {time.time() - t0:.1f}s; prompt: {cfg.language_instruction!r}", flush=True)
        torch.cuda.reset_peak_memory_stats()
        obs, _ = env.reset(seed=args.seed)
        expert = FastenerPickExpert(u.num_envs, u.device, BIN.center, params)

        n = u.num_envs
        stages = {k: torch.zeros(n, dtype=torch.bool, device=u.device) for k in ("grasped", "lifted", "in_bin")}
        jaw_at_lift = torch.full((n,), float("nan"), device=u.device)
        ep_steps = torch.zeros(n, dtype=torch.long, device=u.device)
        # Each env contributes the same number of episodes (its first ones): counting the first
        # completions instead would favour quick successes over 600-step failures.
        quota = [args.episodes // n + (1 if i < args.episodes % n else 0) for i in range(n)]
        done_count = [0] * n
        z_max = torch.zeros(n, device=u.device)
        drop_phase = torch.full((n,), -1, dtype=torch.long, device=u.device)  # phase when a lifted bolt fell
        results, steps, t_start = [], 0, time.time()
        while len(results) < args.episodes:
            pol = obs["policy"]
            action = expert.act(pol["ee_pose"][:, :3], pol["fastener_pose"])
            for k in stages:
                stages[k] |= obs["subtask_terms"][k].bool().view(-1)
            lifting = (expert.phase == PHASE_NAMES.index("lift")) & torch.isnan(jaw_at_lift)
            jaw_at_lift[lifting] = pol["gripper_pos"][lifting, 0]
            z_max = torch.maximum(z_max, pol["fastener_pose"][:, 2])
            last_pos = pol["fastener_pose"][:, :3].clone()
            phase_when_dropped = torch.where(
                (pol["fastener_pose"][:, 2] < 0.02) & (z_max > 0.05) & (drop_phase < 0), expert.phase, drop_phase)
            drop_phase = phase_when_dropped
            obs, _, terminated, truncated, _ = env.step(action)
            steps += 1
            ep_steps += 1
            done = (terminated | truncated).nonzero().squeeze(-1)
            if len(done):
                success = u.termination_manager.get_term("success")
                for i in done.tolist():
                    if done_count[i] >= quota[i]:
                        continue
                    done_count[i] += 1
                    r = {"episode": len(results), "env": i, "success": bool(success[i]),
                         "steps": int(ep_steps[i]),
                         **{k: bool(v[i]) for k, v in stages.items()},
                         "max_phase": PHASE_NAMES[int(expert.max_phase[i])],
                         "jaw_at_lift": round(float(jaw_at_lift[i]), 3),
                         "fastener_z_max": round(float(z_max[i]), 3),
                         "final_pos": [round(x, 3) for x in last_pos[i].tolist()],
                         "fell_in_phase": PHASE_NAMES[int(drop_phase[i])] if drop_phase[i] >= 0 else None}
                    results.append(r)
                    print(f"[eval] ep {r['episode'] + 1:3d} env {i:2d}: {'PLACED' if r['success'] else 'failed'}"
                          f"  grasped={r['grasped']:d} lifted={r['lifted']:d} in_bin={r['in_bin']:d}"
                          f"  reached={r['max_phase']} jaw@lift={r['jaw_at_lift']} z_max={r['fastener_z_max']} steps={r['steps']}"
                          + ("" if r["success"] else f" final={r['final_pos']} fell_in={r['fell_in_phase']}"), flush=True)
                expert.reset(done)
                for v in stages.values():
                    v[done] = False
                jaw_at_lift[done] = float("nan")
                ep_steps[done] = 0
                z_max[done] = 0.0
                drop_phase[done] = -1
        wall = time.time() - t_start
        k = sum(r["success"] for r in results)
        lo, hi = wilson(k, len(results))
        summary = {
            "task": args.task, "fastener": cfg.fastener_names, "seed": args.seed, "num_envs": n,
            "cameras": not args.no_cameras, "episodes": len(results), "success": k,
            "success_rate": k / len(results), "wilson95": [round(lo, 3), round(hi, 3)],
            **{f"{s}_rate": sum(r[s] for r in results) / len(results) for s in ("grasped", "lifted", "in_bin")},
            "env_steps_per_s": round(steps * n / wall, 1), "steps": steps,
            "torch_peak_mem_gb": round(torch.cuda.max_memory_allocated() / 1024**3, 2),
            "params": vars(params), "results": results,
        }
        print(f"[eval] SUCCESS {k}/{len(results)} = {100 * k / len(results):.0f}% "
              f"(95% Wilson CI {100 * lo:.0f}-{100 * hi:.0f}%)  grasped {summary['grasped_rate']:.2f} "
              f"lifted {summary['lifted_rate']:.2f} in_bin {summary['in_bin_rate']:.2f}  "
              f"{summary['env_steps_per_s']} env-steps/s", flush=True)
        if args.json:
            with open(args.json, "w") as f:
                json.dump(summary, f, indent=1)
        env.close()


if __name__ == "__main__":
    main()
