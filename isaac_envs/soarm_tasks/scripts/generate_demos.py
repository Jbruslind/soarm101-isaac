"""Generate scripted demos for Soarm-PickPlace-Fastener-IK-Abs-v0 into an Isaac Lab HDF5 file.

The batched scripted expert (``soarm_tasks.expert``) drives ``--num_envs`` envs. Isaac Lab's
recorder manager, with ``soarm_tasks.recorders.DemoRecorderManagerCfg``, records each episode:
EE-pose actions, applied joint targets, joint state, both camera images, initial/step states, and
seed / fastener / prompt as attributes (layout: ``soarm_tasks.demo_schema``). An episode counts
as a success, like in ``record_demos.py``, once the task's success term has held for
``--num_success_steps`` steps; it is then written and its env reset. Drops and timeouts are failed
episodes, kept only with ``--keep_failures`` (in ``<name>_failed.hdf5``).

Run in the isaac-sim container (from docker/):

    docker compose --profile collect run --rm isaac-sim /isaac-sim/python.sh \\
        /isaac_envs/soarm_tasks/scripts/generate_demos.py --num_demos 50 --num_envs 4 --seed 0 \\
        --dataset_file /data/demos/fastener_m6/scripted.hdf5

Replay the file with Isaac Lab's ``replay_demos.py`` (see docs/ISAAC_SIM_ENVIRONMENTS.md).
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
parser.add_argument("--task", default="Soarm-PickPlace-Fastener-IK-Abs-v0")
parser.add_argument("--num_demos", type=int, default=50, help="successful demos to write")
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fastener", default=None, help="fastener name, e.g. hex_bolt_M6x20 (default: task's)")
parser.add_argument("--dataset_file", default="/data/demos/fastener_m6/scripted.hdf5")
parser.add_argument("--num_success_steps", type=int, default=10,
                    help="steps the success term must hold before a demo is written (record_demos.py: 10)")
parser.add_argument("--keep_failures", action="store_true", help="also write failed episodes to <name>_failed.hdf5")
parser.add_argument("--no-cameras", action="store_true", help="drop both cameras (no images in the file)")
parser.add_argument("--max_episodes", type=int, default=0, help="stop after this many finished episodes (0: 4x num_demos)")
parser.add_argument("--json", default=None, help="write a run summary here")
# corrective demonstrations (F7): see ExpertParams.retry / miss_prob / perturb_prob
parser.add_argument("--retry", action="store_true", help="expert retries after a missed grasp")
parser.add_argument("--miss_prob", type=float, default=0.0, help="chance per episode of a deliberate first miss (needs --retry)")
parser.add_argument("--perturb_prob", type=float, default=0.0, help="chance per episode of a goal displacement while approaching")
parser.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE",
                    help="override other ExpertParams fields, e.g. hover_tol=0.015")
add_launcher_args(parser)
args = parser.parse_args()
args.enable_cameras = not args.no_cameras
args.require_kit = True
sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))


def main():
    with launch_simulation(None, args):
        import gymnasium as gym
        import torch
        from isaaclab.managers import DatasetExportMode
        from isaaclab_tasks.utils import parse_env_cfg

        import soarm_tasks  # noqa: F401
        from soarm_tasks.expert import ExpertParams, FastenerPickExpert
        from soarm_tasks.recorders import DemoRecorderManagerCfg
        from soarm_tasks.work_cell import BIN

        # [start:cfg]
        cfg = parse_env_cfg(args.task, num_envs=args.num_envs)
        cfg.seed = args.seed
        cfg.env_name = args.task  # stored in the file; replay_demos.py reads it
        if args.fastener:
            cfg.fastener_names = (args.fastener,)
            cfg.__post_init__()
        if args.no_cameras:
            cfg.disable_cameras()
        # Success is checked here (held for num_success_steps), not by the termination manager.
        success_term = cfg.terminations.success
        cfg.terminations.success = None
        out_dir, out_name = os.path.split(os.path.abspath(args.dataset_file))
        corrective = args.retry or args.miss_prob > 0 or args.perturb_prob > 0
        cfg.recorders = DemoRecorderManagerCfg(
            dataset_export_dir_path=out_dir,
            dataset_filename=os.path.splitext(out_name)[0],
            dataset_export_mode=(DatasetExportMode.EXPORT_SUCCEEDED_FAILED_IN_SEPARATE_FILES if args.keep_failures
                                 else DatasetExportMode.EXPORT_SUCCEEDED_ONLY),
        )
        if corrective:  # labelled in the file so the LeRobot conversion can tell them apart
            cfg.recorders.record_episode_meta.source = "scripted_corrective"
        # [end:cfg]
        os.makedirs(out_dir, exist_ok=True)
        print(f"[demos] task={args.task} num_envs={args.num_envs} seed={args.seed} "
              f"cameras={not args.no_cameras} retry={args.retry} miss_prob={args.miss_prob} "
              f"perturb_prob={args.perturb_prob} set={args.set} -> {args.dataset_file}", flush=True)

        env = gym.make(args.task, cfg=cfg)
        u = env.unwrapped
        rec = u.recorder_manager
        obs, _ = env.reset(seed=args.seed)
        params = ExpertParams(retry=args.retry, miss_prob=args.miss_prob, perturb_prob=args.perturb_prob)
        for kv in args.set:
            k, v = kv.split("=")
            setattr(params, k, type(getattr(params, k))(float(v)))
        expert = FastenerPickExpert(u.num_envs, u.device, BIN.center, params)
        held = torch.zeros(u.num_envs, dtype=torch.long, device=u.device)
        max_episodes = args.max_episodes or 4 * args.num_demos
        steps, t0 = 0, time.time()

        # [start:loop]
        while rec.exported_successful_episode_count < args.num_demos:
            if rec.exported_successful_episode_count + rec.exported_failed_episode_count >= max_episodes:
                print(f"[demos] stopping after {max_episodes} episodes", flush=True)
                break
            pol = obs["policy"]
            action = expert.act(pol["ee_pose"][:, :3], pol["fastener_pose"])
            obs, _, terminated, truncated, _ = env.step(action)  # drops/timeouts reset (and export) here
            steps += 1
            failed = (terminated | truncated).nonzero().squeeze(-1)
            expert.reset(failed)
            held[failed] = 0
            ok = success_term.func(u, **success_term.params)
            held = torch.where(ok, held + 1, torch.zeros_like(held))
            done = (held >= args.num_success_steps).nonzero().squeeze(-1)
            done = done[: args.num_demos - rec.exported_successful_episode_count]
            if len(done):
                rec.set_success_to_episodes(done, torch.ones(len(done), 1, dtype=torch.bool, device=u.device))
                rec.export_episodes(done)
                obs, _ = u.reset(env_ids=done)  # the new episodes start recording here
                expert.reset(done)
                held[done] = 0
                n_ok, n_fail = rec.exported_successful_episode_count, rec.exported_failed_episode_count
                print(f"[demos] {n_ok}/{args.num_demos} written  (failed {n_fail}, "
                      f"success rate {n_ok / (n_ok + n_fail):.0%}, {time.time() - t0:.0f}s)", flush=True)
        # [end:loop]
        wall = time.time() - t0
        n_ok, n_fail = rec.exported_successful_episode_count, rec.exported_failed_episode_count
        env.close()  # closes the HDF5 file
        report(n_ok, n_fail, wall, steps)  # inside the launch context: Kit ends the process on exit


def report(n_ok: int, n_fail: int, wall: float, steps: int):
    from soarm_tasks import demo_schema

    info = demo_schema.file_info(args.dataset_file)
    problems = demo_schema.validate(args.dataset_file, require_images=not args.no_cameras)
    summary = {
        "task": args.task, "seed": args.seed, "num_envs": args.num_envs, "cameras": not args.no_cameras,
        "written": n_ok, "failed": n_fail, "success_rate": n_ok / max(1, n_ok + n_fail),
        "wall_s": round(wall, 1), "env_steps_per_s": round(steps * args.num_envs / wall, 1),
        "file": args.dataset_file, "file_mb": round(info["bytes"] / 2**20, 1), "steps_in_file": info["steps"],
        "schema_problems": problems,
    }
    print(f"[demos] DONE {n_ok} demos written, {n_fail} failed (success rate {summary['success_rate']:.1%}) "
          f"in {wall:.0f}s; {summary['file_mb']} MB, {info['steps']} steps; schema "
          f"{'OK' if not problems else problems[:5]}", flush=True)
    if args.json:
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=1)


if __name__ == "__main__":
    main()
