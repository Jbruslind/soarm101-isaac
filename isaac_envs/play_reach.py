"""Run the SO-ARM101 reach environment with a GUI so the third-person viewport camera follows the robot.

Use this to visually verify robot behavior during testing. The viewport camera is configured
in SoarmReachEnvCfg.viewer to track the robot (asset_root).

Usage (inside Isaac Sim / Isaac Lab container, with display or livestream):
    /isaac-sim/python.sh /isaac_envs/play_reach.py

From host with Docker (streaming):
    cd docker && docker compose --profile collect run --rm -e LIVESTREAM=2 isaac-sim \
        /isaac-sim/python.sh /isaac_envs/play_reach.py
    Then connect via the Isaac Sim WebRTC streaming client to see the viewport.
"""

from __future__ import annotations

import argparse

# Launch the app with a window (not headless) so the viewport is visible.
# For container without display, set LIVESTREAM=2 and use the WebRTC client.
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Play SO-ARM101 reach env with third-person camera")
parser.add_argument("--num-envs", type=int, default=1, help="Number of environments")
parser.add_argument("--use-camera", action="store_true", help="Enable wrist camera (optional)")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

import torch

from soarm_reach_env import SoarmReachEnv, SoarmReachEnvCfg


def main():
    cfg = SoarmReachEnvCfg()
    cfg.num_envs = args.num_envs
    cfg.use_camera = args.use_camera

    env = SoarmReachEnv(cfg, render_mode="human")
    obs, info = env.reset()

    print("SO-ARM101 reach env running. Viewport camera follows the robot. Close the window to exit.", flush=True)

    while simulation_app.is_running():
        # Simple random policy for demonstration
        action = torch.randn(cfg.num_envs, cfg.action_space, device=env.device) * 0.3
        obs, reward, terminated, truncated, info = env.step(action)
        if terminated.any() or truncated.any():
            obs, info = env.reset()

    env.close()
    simulation_app.close()


if __name__ == "__main__":
    main()
