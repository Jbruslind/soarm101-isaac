"""F0 checkpoint: check that Isaac Sim, Isaac Lab, the GPU and the SO-ARM101 asset work together.

Tutorial page: docs/tutorials/fasteners/F0_setup.md

The script launches Isaac Sim through Isaac Lab, builds the SO-ARM101 reach environment from
``isaac_envs/soarm_reach_env.py``, drives ``shoulder_pan`` for a few policy steps to prove that
actions reach the robot, times a short random rollout, and prints one ``[F0] PASS`` (or
``[F0] FAIL``) line with the versions and the GPU name. With a window open (``--gui``) it keeps
the robot moving until you close the window, so you can look at it in Moonlight.

Usage (from the repository root on the host):

    ./scripts/isaac.sh --headless /isaac-sim/python.sh /tutorials/fasteners/part1/00_check_setup.py
    ./scripts/isaac.sh --gui      /isaac-sim/python.sh /tutorials/fasteners/part1/00_check_setup.py

Exit code 0 means PASS, 1 means FAIL.
"""

from __future__ import annotations

# [start:args]
import argparse
import os
import sys
import time

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="F0: check the Isaac Sim / Isaac Lab / SO-ARM101 setup.")
parser.add_argument("--steps", type=int, default=100, help="Random-action steps to time after the motion check.")
parser.add_argument("--min-pan", type=float, default=0.3, help="Pass threshold: shoulder_pan travel (rad).")
add_launcher_args(parser)
args = parser.parse_args()

# scripts/isaac.sh --gui sets HEADLESS=0. Isaac Lab 3.0 opens a window only for a Kit visualizer.
if os.environ.get("HEADLESS") == "0" and not args.visualizer:
    args.visualizer = "kit"
gui = "kit" in str(args.visualizer or "")
# The env config is built inside the launched app, so tell the launcher it needs Kit (PhysX + RTX).
args.require_kit = True

# The repository's environments are mounted read-only at /isaac_envs in the isaac-sim container.
sys.path.insert(0, os.environ.get("SOARM_ISAAC_ENVS", "/isaac_envs"))
# [end:args]


def versions() -> dict[str, str]:
    """Isaac Sim, Isaac Lab and GPU identity (call inside the launched app)."""
    # [start:versions]
    import isaaclab
    import torch
    from isaacsim.core.version import get_version

    core, prerelease = get_version()[:2]  # read from /isaac-sim/VERSION
    # Isaac Lab's release is in the repo's VERSION file; isaaclab.__version__ is the extension version.
    lab_version_file = os.path.join(os.environ.get("ISAACLAB_PATH", "/workspace/isaaclab"), "VERSION")
    with open(lab_version_file, encoding="utf-8") as f:
        lab_release = f.read().strip()
    props = torch.cuda.get_device_properties(0)
    return {
        "isaac_sim": core,
        "isaac_sim_build": core + (f"-{prerelease}" if prerelease else ""),
        "isaac_lab": lab_release,
        "isaaclab_ext": isaaclab.__version__,
        "gpu": props.name.replace(" ", "_"),
        "vram_gb": f"{props.total_memory / 1024**3:.1f}",
    }
    # [end:versions]


def main() -> None:
    with launch_simulation(None, args):
        import torch

        from soarm_reach_env import SoarmReachEnv, SoarmReachEnvCfg

        info = versions()
        print(f"[F0] isaac_sim_build={info['isaac_sim_build']} isaaclab_ext={info['isaaclab_ext']} "
              f"gpu={info['gpu']} vram_gb={info['vram_gb']}", flush=True)

        # [start:env]
        cfg = SoarmReachEnvCfg()
        env = SoarmReachEnv(cfg, render_mode="human" if gui else None)
        obs, _ = env.reset()
        print(f"[F0] reach env built: num_envs={env.num_envs} device={env.device} "
              f"obs={tuple(obs['policy'].shape)} actions={cfg.action_space}", flush=True)
        # [end:env]

        # [start:motion]
        # Command +action_scale rad/step on shoulder_pan (action index 0) for 20 policy steps.
        pan0 = env.robot.data.joint_pos.torch[0, 0].item()
        action = torch.zeros(env.num_envs, cfg.action_space, device=env.device)
        action[:, 0] = 1.0
        for _ in range(20):
            env.step(action)
        pan_travel = env.robot.data.joint_pos.torch[0, 0].item() - pan0
        print(f"[F0] shoulder_pan travel after 20 steps: {pan_travel:+.3f} rad "
              f"(need >= {args.min_pan:.2f})", flush=True)
        # [end:motion]

        # [start:timing]
        t0 = time.perf_counter()
        for _ in range(args.steps):
            action = 0.3 * torch.randn(env.num_envs, cfg.action_space, device=env.device)
            env.step(action)  # DirectRLEnv resets finished episodes by itself
        steps_per_s = args.steps / (time.perf_counter() - t0)
        # [end:timing]

        # [start:verdict]
        ok = pan_travel >= args.min_pan
        verdict = "PASS" if ok else "FAIL"
        print(f"[F0] {verdict} isaac_sim={info['isaac_sim']} isaac_lab={info['isaac_lab']} "
              f"gpu={info['gpu']} pan_travel={pan_travel:.3f}rad steps_per_s={steps_per_s:.1f} "
              f"mode={'gui' if gui else 'headless'}", flush=True)
        # [end:verdict]

        if gui and ok:
            # [start:keep-open]
            import omni.kit.app

            print("[F0] Window open: watch the arm in Moonlight, close the window to exit.", flush=True)
            app = omni.kit.app.get_app()
            while app.is_running():
                env.step(0.3 * torch.randn(env.num_envs, cfg.action_space, device=env.device))
            # [end:keep-open]

        env.close()
        if not ok:
            # Inside the with block, so launch_simulation passes the code on to Kit's exit.
            sys.exit(1)


if __name__ == "__main__":
    main()
