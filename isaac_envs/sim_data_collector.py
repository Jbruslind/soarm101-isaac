"""Collect episodes from Isaac Lab environments and save in LeRobot v3.0 format.

Uses an IK-based scripted policy to generate smooth, goal-directed reaching
demonstrations.  Captures joint states, camera images, and language
instructions, and writes episodes to disk as Parquet + MP4 + JSON metadata.

Usage (inside Isaac Sim container):
    /isaac-sim/python.sh sim_data_collector.py \
        --env reach \
        --num-episodes 50 \
        --output-dir /data/episodes
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError:
    pa = None
    pq = None

try:
    import av
except ImportError:
    av = None


FPS = 30
IMAGE_SIZE = 224

LANGUAGE_INSTRUCTIONS = {
    "reach": "move the robot arm to the green target",
    "pick": "pick up the red cube and place it at the green target",
}


class LeRobotWriter:
    """Writes episode data in LeRobot v3.0 format."""

    def __init__(self, output_dir: str, fps: int = FPS, use_camera: bool = True):
        self.output_dir = Path(output_dir)
        self.fps = fps
        self.use_camera = use_camera
        self.episodes: list[dict] = []
        self.all_frames: list[dict] = []
        self.video_writers: dict[str, object] = {}
        self.episode_idx = 0
        self.global_frame_idx = 0

        (self.output_dir / "meta").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "meta" / "episodes").mkdir(exist_ok=True)
        (self.output_dir / "data").mkdir(exist_ok=True)
        (self.output_dir / "videos").mkdir(exist_ok=True)

    def start_episode(self, task: str = "reach_target"):
        self._current_frames: list[dict] = []
        self._current_task = task
        self._episode_start_frame = self.global_frame_idx
        self._video_containers: dict = {}

    def add_frame(
        self,
        state: np.ndarray,
        action: np.ndarray,
        language_instruction: str = "",
        images: dict[str, np.ndarray] | None = None,
        timestamp: float | None = None,
    ):
        frame = {
            "episode_index": self.episode_idx,
            "frame_index": len(self._current_frames),
            "index": self.global_frame_idx,
            "timestamp": timestamp or (len(self._current_frames) / self.fps),
            "language_instruction": language_instruction,
        }

        for i, name in enumerate([
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]):
            frame[f"observation.state.{name}"] = float(state[i]) if i < len(state) else 0.0

        for i, name in enumerate([
            "shoulder_pan", "shoulder_lift", "elbow_flex",
            "wrist_flex", "wrist_roll", "gripper",
        ]):
            frame[f"action.{name}"] = float(action[i]) if i < len(action) else 0.0

        frame["observation.state"] = state.tolist()
        frame["action"] = action.tolist()

        self._current_frames.append(frame)
        self.global_frame_idx += 1

        if images and av is not None:
            for cam_name, img_array in images.items():
                self._write_video_frame(cam_name, img_array)

    def _write_video_frame(self, cam_name: str, img: np.ndarray):
        video_key = f"{self.episode_idx}_{cam_name}"
        if video_key not in self._video_containers:
            video_path = (
                self.output_dir
                / "videos"
                / f"observation.images.{cam_name}_episode_{self.episode_idx:06d}.mp4"
            )
            container = av.open(str(video_path), mode="w")
            stream = container.add_stream("h264", rate=self.fps)
            stream.width = img.shape[1]
            stream.height = img.shape[0]
            stream.pix_fmt = "yuv420p"
            self._video_containers[video_key] = (container, stream)

        container, stream = self._video_containers[video_key]
        frame = av.VideoFrame.from_ndarray(img, format="rgb24")
        for packet in stream.encode(frame):
            container.mux(packet)

    def end_episode(self):
        for key, (container, stream) in self._video_containers.items():
            for packet in stream.encode():
                container.mux(packet)
            container.close()
        self._video_containers.clear()

        ep_len = len(self._current_frames)
        self.episodes.append({
            "episode_index": self.episode_idx,
            "tasks": [self._current_task],
            "length": ep_len,
        })

        self.all_frames.extend(self._current_frames)
        self.episode_idx += 1

    def save(self):
        """Write Parquet data and JSON metadata to disk."""
        if not self.all_frames:
            print("No frames to save.")
            return

        if pa is not None and pq is not None:
            table = pa.Table.from_pylist(self.all_frames)
            pq.write_table(table, self.output_dir / "data" / "train-00000-of-00001.parquet")

        for ep in self.episodes:
            ep_path = self.output_dir / "meta" / "episodes" / f"{ep['episode_index']:06d}.json"
            with open(ep_path, "w") as f:
                json.dump(ep, f, indent=2)

        features = {
            "observation.state": {"dtype": "float32", "shape": [6]},
            "action": {"dtype": "float32", "shape": [6]},
            "language_instruction": {"dtype": "string", "shape": [1]},
        }
        if self.use_camera:
            for cam_name in ("wrist", "third_person"):
                features[f"observation.images.{cam_name}"] = {
                    "dtype": "video",
                    "shape": [IMAGE_SIZE, IMAGE_SIZE, 3],
                    "video_info": {
                        "video.fps": float(self.fps),
                        "video.codec": "h264",
                        "video.pix_fmt": "yuv420p",
                        "has_audio": False,
                    },
                }

        info = {
            "codebase_version": "v3.0",
            "robot_type": "so_arm101",
            "fps": self.fps,
            "total_episodes": len(self.episodes),
            "total_frames": self.global_frame_idx,
            "features": features,
        }
        with open(self.output_dir / "meta" / "info.json", "w") as f:
            json.dump(info, f, indent=2)

        if self.all_frames:
            states = np.array([f["observation.state"] for f in self.all_frames], dtype=np.float32)
            actions = np.array([f["action"] for f in self.all_frames], dtype=np.float32)
            stats = {
                "observation.state": {
                    "mean": states.mean(axis=0).tolist(),
                    "std": np.maximum(states.std(axis=0), 1e-6).tolist(),
                    "min": states.min(axis=0).tolist(),
                    "max": states.max(axis=0).tolist(),
                },
                "action": {
                    "mean": actions.mean(axis=0).tolist(),
                    "std": np.maximum(actions.std(axis=0), 1e-6).tolist(),
                    "min": actions.min(axis=0).tolist(),
                    "max": actions.max(axis=0).tolist(),
                },
            }
            with open(self.output_dir / "meta" / "stats.json", "w") as f:
                json.dump(stats, f, indent=2)

        print(f"Saved {len(self.episodes)} episodes, {self.global_frame_idx} frames "
              f"to {self.output_dir}")


def _setup_ik_controller(robot, device):
    """Create a DifferentialIKController targeting the SO-ARM101 end-effector.

    Returns (ik_controller, arm_joint_ids, ee_body_id, ee_jacobi_idx).
    """
    from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

    arm_joint_ids, _ = robot.find_joints(
        ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"]
    )
    ee_body_ids, _ = robot.find_bodies(["gripper_frame_link"])
    ee_body_id = ee_body_ids[0]

    # For a fixed-base robot, the Jacobian omits the root body, so the
    # index into the Jacobian tensor is one less than the body index.
    ee_jacobi_idx = ee_body_id - 1

    diff_ik_cfg = DifferentialIKControllerCfg(
        command_type="position",
        use_relative_mode=False,
        ik_method="dls",
    )
    ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)

    return ik_controller, arm_joint_ids, ee_body_id, ee_jacobi_idx


def collect_episodes(
    env_type: str,
    num_episodes: int,
    output_dir: str,
    use_camera: bool,
    wait_for_key: bool,
):
    """Run data collection loop with an IK-based scripted policy."""

    livestream = os.environ.get("LIVESTREAM") in ("1", "2")

    # AppLauncher is always used: cameras need the rendering pipeline, and
    # livestream needs the WebRTC extension.  Both require AppLauncher.
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=True,
        enable_cameras=use_camera or livestream,
    )
    sim_app = app_launcher.app

    if livestream:
        import carb
        carb.settings.get_settings().set_bool("/app/livestream/allowResize", True)

    render_mode = "human" if livestream else None

    print("SimulationApp started. Loading environment (this can take 1-2 min on first run)...", flush=True)
    if env_type == "reach":
        from soarm_reach_env import SoarmReachEnv, SoarmReachEnvCfg
        cfg = SoarmReachEnvCfg()
        cfg.use_camera = use_camera
        cfg.use_third_person_camera = use_camera
        cfg.num_envs = 1
        env = SoarmReachEnv(cfg, render_mode=render_mode)
    elif env_type == "pick":
        from soarm_pick_env import SoarmPickEnv, SoarmPickEnvCfg
        cfg = SoarmPickEnvCfg()
        cfg.use_camera = use_camera
        cfg.use_third_person_camera = use_camera
        cfg.num_envs = 1
        env = SoarmPickEnv(cfg, render_mode=render_mode)
    else:
        raise ValueError(f"Unknown env type: {env_type}")

    # -- IK controller setup --------------------------------------------------
    from isaaclab.utils.math import subtract_frame_transforms

    robot = env.robot
    ik_controller, arm_joint_ids, ee_body_id, ee_jacobi_idx = _setup_ik_controller(
        robot, env.device
    )

    lang = LANGUAGE_INSTRUCTIONS.get(env_type, "move the robot arm")

    print("Environment ready.", flush=True)
    if wait_for_key:
        public_ip = os.environ.get("PUBLIC_IP", "127.0.0.1")
        if livestream:
            print(
                f"\nWebRTC streaming is active. Connect with the Isaac Sim WebRTC Streaming Client:\n"
                f"  Server address: {public_ip}\n"
                f"Press Enter to start collection...",
                flush=True,
            )
        else:
            print(
                "Waiting for input before collecting episodes.\n"
                "Press Enter to start collection...",
                flush=True,
            )
        try:
            input()
        except EOFError:
            pass

    print("Collecting episodes with IK-based scripted policy...", flush=True)
    writer = LeRobotWriter(output_dir, use_camera=use_camera)

    for ep in range(num_episodes):
        obs, info = env.reset()
        target_w = env.target_pos.clone()

        ik_controller.reset()

        writer.start_episode(task=lang)
        done = False
        step = 0

        while not done:
            # Retrieve quantities from the physics engine
            jacobian_full = robot.root_physx_view.get_jacobians()
            jacobian = jacobian_full[:, ee_jacobi_idx, :, arm_joint_ids]

            # body_state_w is (num_envs, num_bodies, 13) = pos(3) + quat(4) + vel(6)
            ee_state_w = robot.data.body_state_w[:, ee_body_id]
            ee_pos_w = ee_state_w[:, 0:3]
            ee_quat_w = ee_state_w[:, 3:7]

            # root_state_w is (num_envs, 13)
            root_state_w = robot.data.root_state_w
            root_pos_w = root_state_w[:, 0:3]
            root_quat_w = root_state_w[:, 3:7]

            arm_joint_pos = robot.data.joint_pos[:, arm_joint_ids]

            # Transform EE pose from world frame to robot base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w,
                ee_pos_w, ee_quat_w,
            )

            # Transform target from world frame to base frame
            identity_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=env.device)
            target_b, _ = subtract_frame_transforms(
                root_pos_w, root_quat_w,
                target_w, identity_quat,
            )

            # Compute IK: desired arm joint positions to reach the target
            # set_command requires ee_quat even for position-only mode
            ik_controller.set_command(target_b, ee_quat=ee_quat_b)
            arm_joint_target = ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, arm_joint_pos
            )

            # Build full 6-DOF joint target (arm + gripper held at 0)
            full_joint_target = robot.data.joint_pos.clone()
            full_joint_target[:, arm_joint_ids] = arm_joint_target

            # Convert to delta-action that the environment expects
            current_pos = robot.data.joint_pos
            delta = (full_joint_target - current_pos) / cfg.action_scale
            action = delta.clamp(-1.0, 1.0)

            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated.any() or truncated.any()

            # Record absolute joint-position targets as the action
            state_np = obs["policy"][0, :6].cpu().numpy()
            action_np = full_joint_target[0].cpu().numpy()

            images = {}
            if use_camera and "wrist_rgb" in obs:
                img = obs["wrist_rgb"][0].cpu().numpy()
                images["wrist"] = img[:, :, :3] if img.shape[-1] == 4 else img
            if use_camera and "third_person_rgb" in obs:
                img = obs["third_person_rgb"][0].cpu().numpy()
                images["third_person"] = img[:, :, :3] if img.shape[-1] == 4 else img

            writer.add_frame(
                state=state_np,
                action=action_np,
                language_instruction=lang,
                images=images,
            )
            step += 1

        writer.end_episode()

        ee_pos_final = robot.data.body_state_w[:, ee_body_id, 0:3]
        dist = torch.norm(ee_pos_final - target_w, dim=-1).item()
        status = "REACHED" if dist < cfg.success_threshold else f"dist={dist:.3f}m"
        print(f"Episode {ep + 1}/{num_episodes}: {step} steps  ({status})", flush=True)

    writer.save()
    env.close()
    sim_app.close()


def main():
    parser = argparse.ArgumentParser(description="Collect SO-ARM101 sim episodes")
    parser.add_argument("--env", choices=["reach", "pick"], default="reach")
    parser.add_argument("--num-episodes", type=int, default=50)
    parser.add_argument("--output-dir", default="/data/episodes")
    parser.add_argument(
        "--no-camera",
        action="store_true",
        help="Disable camera capture (not recommended for VLA training).",
    )
    parser.add_argument(
        "--wait-for-key",
        action="store_true",
        help="Wait for Enter after the app + environment are loaded (useful for WebRTC viewing).",
    )
    args = parser.parse_args()

    collect_episodes(
        args.env,
        args.num_episodes,
        args.output_dir,
        use_camera=not args.no_camera,
        wait_for_key=args.wait_for_key,
    )


if __name__ == "__main__":
    main()
