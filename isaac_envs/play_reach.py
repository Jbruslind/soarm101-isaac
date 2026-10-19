"""Run the SO-ARM101 reach environment with a GUI so the third-person viewport camera follows the robot.

Use this to visually verify robot behavior during testing. The viewport camera is configured
in SoarmReachEnvCfg.sim.default_visualizer_cfg to track the robot.

Usage (inside the Isaac Sim / Isaac Lab container):
    /isaac-sim/python.sh /isaac_envs/play_reach.py --viz kit

From the host, watched over Moonlight (Sunshine display, docs/SUNSHINE_STREAMING.md):
    ./scripts/play_reach.sh --gui

With HEADLESS=0 (set by --gui) and no --viz, the Kit visualizer is selected automatically.
"""

from __future__ import annotations

import argparse
import os

from isaaclab.app import add_launcher_args, launch_simulation

parser = argparse.ArgumentParser(description="Play SO-ARM101 reach env with third-person camera")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
parser.add_argument("--use-camera", action="store_true", help="Enable wrist camera (optional)")
add_launcher_args(parser)
args = parser.parse_args()

# Isaac Lab 3.0 only opens a window for a Kit visualizer.
if os.environ.get("HEADLESS") == "0" and not args.visualizer:
    args.visualizer = "kit"
if args.use_camera:
    args.enable_cameras = True
# The env config is built inside the launched app, so tell the launcher this
# PhysX/RTX task needs Kit rather than letting it infer that from a config.
args.require_kit = True


def main():
    with launch_simulation(None, args):
        import omni.kit.app
        import torch

        from soarm_reach_env import SoarmReachEnv, SoarmReachEnvCfg

        cfg = SoarmReachEnvCfg()
        cfg.num_envs = args.num_envs
        cfg.scene.num_envs = args.num_envs
        cfg.use_camera = args.use_camera

        env = SoarmReachEnv(cfg, render_mode="human")
        obs, info = env.reset()

        print("SO-ARM101 reach env running. Viewport camera follows the robot. Close the window to exit.", flush=True)

        app = omni.kit.app.get_app()
        while app.is_running():
            # Simple random policy for demonstration
            action = torch.randn(cfg.num_envs, cfg.action_space, device=env.device) * 0.3
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated.any() or truncated.any():
                obs, info = env.reset()

        env.close()


if __name__ == "__main__":
    main()
