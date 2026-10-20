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


def _setup_ik_controller(robot, device, command_type: str = "position"):
    """Create a DifferentialIKController targeting the SO-ARM101 end-effector.

    ``command_type`` is "position" (reach) or "pose" (pick, which needs a top-down gripper).

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
        command_type=command_type,
        use_relative_mode=False,
        ik_method="dls",
    )
    ik_controller = DifferentialIKController(diff_ik_cfg, num_envs=1, device=device)

    return ik_controller, arm_joint_ids, ee_body_id, ee_jacobi_idx


def respect_joint_limits(
    q_target: torch.Tensor,
    q: torch.Tensor,
    jacobian: torch.Tensor,
    pos_err: torch.Tensor,
    limits: torch.Tensor,
    lam: float = 0.01,
) -> torch.Tensor:
    """Clamp an IK joint target to the joint limits and re-solve the position with the other joints.

    The pose IK doesn't know the joint limits. Close to the base, a top-down gripper needs
    wrist_flex past its 95 deg limit; the IK keeps asking for more wrist_flex, the joint stays
    at the limit, and the arm stalls 2-4 cm short of the cube. Here the joints past a limit are
    held at it, and the remaining joints track the position error by damped least squares, using
    the position rows of ``jacobian``. The gripper then tilts a few degrees instead of missing
    the cube.

    Shapes: q_target, q (N, J); jacobian (N, 6, J); pos_err (N, 3); limits (N, J, 2).
    """
    clamped = q_target.clamp(limits[..., 0], limits[..., 1])
    hit = clamped != q_target
    if not hit.any():
        return q_target
    out = clamped.clone()
    for i in range(q.shape[0]):
        free = ~hit[i]
        if not free.any():
            continue
        jp = jacobian[i, :3]
        # position change still needed after the clamped joints move to their (clamped) targets
        err = pos_err[i] - jp[:, ~free] @ (clamped[i, ~free] - q[i, ~free])
        jf = jp[:, free]
        dq = jf.T @ torch.linalg.solve(jf @ jf.T + lam**2 * torch.eye(3, device=q.device), err)
        out[i, free] = (q[i, free] + dq).clamp(limits[i, free, 0], limits[i, free, 1])
    return out


class ScriptedPick:
    """Waypoint state machine for the pick task, driven by differential IK in pose mode.

    Phases: hover above the cube with the gripper open, descend, close, lift, carry over
    the place target, lower, open.  ``step`` returns the EE goal pose (world frame) and the
    gripper joint goal for the current phase.  The gripper is held top-down (gripper_frame_link
    +z pointing down) with its x axis pointing back at the robot, so the jaws open roughly
    radially. The wrist is turned (``dyaw``) so the jaws close square on the cube faces.
    Position-only IK lets the wrist pitch drift, and the jaws then land on top of the cube.
    The collector clamps the IK to the joint limits with :func:`respect_joint_limits`.

    Measured (GitHub issue #28; 3 cm cube, default spawn area, ``SoarmReachEnvCfg`` PhysX
    settings with ``solve_articulation_contact_last``): 20/20 placed headless without cameras,
    5/5 with cameras. Earlier runs while tuning, without the joint-limit clamp, placed
    15-18/20; the misses were cubes near the base. The grasp holds (the jaw stalls at
    0.23-0.25 rad) and the cube is lifted ~8 cm. With the contact-last flag off, 1/10 placed.
    """

    HOVER_Z = 0.08      # m above the table while moving
    GRASP_Z = 0.010     # gripper_frame_link height when grasping. The jaw tips reach ~7 mm
                        # below that frame, so lower and they drag on the table and stall the IK
    PLACE_Z = 0.012
    # Cube centre along the gripper x axis (towards the fixed jaw is +x), from the jaw meshes:
    # the fixed jaw's inner face is at x = +2..+4 mm over the cube's height, so a 3 cm cube
    # touching it sits at x = -13 mm. -16 mm leaves the fixed jaw ~3 mm clear while descending;
    # the closing moving jaw then pushes the cube against it.
    GRASP_OFFSET = 0.016
    MAX_SPEED = 0.01    # m per step the commanded EE goal may move
    HOVER_TOL = 0.004   # m, EE position tolerance above the cube (other phases use ``tol``)
    OPEN = 1.0          # gripper joint goal (rad) while approaching and releasing
    # Gripper goal while holding: below the angle where the jaws touch the cube, so the jaw
    # keeps squeezing. The jaw meshes give a 30 mm gap at ~0.2 rad at the tips; with the
    # collision shapes the closing jaw stalls on the 3 cm cube at 0.23-0.25 rad (measured).
    # The old goal of 0 was not the problem by itself: the env's action clamp limits the jaw
    # target to 0.05 rad past its position, ~4 N m of squeeze either way. The grasp failed
    # because the PhysX solver let the jaw drive push through the contacts (fixed with
    # solve_articulation_contact_last in SoarmReachEnvCfg.sim) and because the jaws closed on
    # the cube at an angle (fixed by aligning the wrist with the cube faces).
    GRIP_CONTACT = 0.24  # rad, jaws on the 3 cm cube
    GRIP_SQUEEZE = 0.14  # rad below contact (goal 0.10 rad)
    # (name, gripper goal ("open"/"closed"), steps to hold after the EE is within tol,
    # max steps in phase)
    PHASES = [
        ("hover", "open", 3, 90), ("descend", "open", 5, 45), ("close", "closed", 45, 45),
        ("lift", "closed", 0, 40), ("carry", "closed", 0, 60), ("lower", "closed", 5, 40),
        ("release", "open", 15, 15),
    ]

    def __init__(
        self,
        cube_w: torch.Tensor,
        place_w: torch.Tensor,
        tol: float = 0.01,
        grip_contact: float | None = None,
        grip_squeeze: float | None = None,
        grasp_offset: float | None = None,
        cube_quat_w: torch.Tensor | None = None,
    ):
        self.cube_w, self.place_w, self.tol = cube_w.clone(), place_w.clone(), tol
        # Turn the wrist so the jaws close square on the cube's faces: closing on a cube that
        # sits at an angle in the jaws catches its edges and squeezes it out. The gripper
        # x axis yaw is the cube face direction (every 90 deg) nearest the default "x towards
        # the base"; the same yaw offset is kept for the rest of the episode.
        self.dyaw = torch.zeros(cube_w.shape[0], device=cube_w.device)
        if cube_quat_w is not None:
            from isaaclab.utils.math import yaw_quat, euler_xyz_from_quat

            cube_yaw = euler_xyz_from_quat(yaw_quat(cube_quat_w))[2]
            default_yaw = torch.atan2(cube_w[:, 1], cube_w[:, 0]) + torch.pi
            quarter = torch.pi / 2
            self.dyaw = torch.remainder(cube_yaw - default_yaw + quarter / 2, quarter) - quarter / 2
        contact = self.GRIP_CONTACT if grip_contact is None else grip_contact
        squeeze = self.GRIP_SQUEEZE if grip_squeeze is None else grip_squeeze
        self.gripper_goals = {"open": self.OPEN, "closed": contact - squeeze}
        self.grasp_offset = self.GRASP_OFFSET if grasp_offset is None else grasp_offset
        self.phase, self.t, self.settled = 0, 0, 0
        self.cmd = self.grasp_pos = None

    @property
    def phase_name(self) -> str:
        return self.PHASES[self.phase][0]

    @staticmethod
    def x_yaw(pos_w: torch.Tensor, dyaw: torch.Tensor | float = 0.0) -> torch.Tensor:
        """Yaw of the gripper x axis at ``pos_w``: pointing towards the base, turned by ``dyaw``."""
        return torch.atan2(pos_w[:, 1], pos_w[:, 0]) + torch.pi + dyaw

    @staticmethod
    def top_down_quat(pos_w: torch.Tensor, dyaw: torch.Tensor | float = 0.0) -> torch.Tensor:
        """(x, y, z, w) orientation with z down and x pointing from ``pos_w`` towards the base,
        turned by ``dyaw`` about the vertical."""
        from isaaclab.utils.math import quat_from_matrix

        yaw = ScriptedPick.x_yaw(pos_w, dyaw)
        c, s, zero = torch.cos(yaw), torch.sin(yaw), torch.zeros_like(yaw)
        x_axis = torch.stack([c, s, zero], -1)
        y_axis = torch.stack([s, -c, zero], -1)
        z_axis = torch.stack([zero, zero, -torch.ones_like(yaw)], -1)
        return quat_from_matrix(torch.stack([x_axis, y_axis, z_axis], -1))

    def _goal(self, ee_pos_w: torch.Tensor) -> torch.Tensor:
        name = self.PHASES[self.phase][0]
        if name in ("close", "lift"):
            # Hold the EE where the descent ended while closing (pushing on towards an
            # unreachable goal shoves the cube away), then lift straight up from there.
            if self.grasp_pos is None:
                self.grasp_pos = self.cmd = ee_pos_w.clone()
            goal = self.grasp_pos.clone()
            if name == "lift":
                goal[:, 2] = self.HOVER_Z
            return goal
        goal = (self.cube_w if name in ("hover", "descend") else self.place_w).clone()
        yaw = self.x_yaw(goal, self.dyaw)
        # the cube sits at x = -grasp_offset in the gripper frame
        goal[:, :2] += self.grasp_offset * torch.stack([torch.cos(yaw), torch.sin(yaw)], -1)
        goal[:, 2] = {"descend": self.GRASP_Z, "lower": self.PLACE_Z,
                      "release": self.PLACE_Z}.get(name, self.HOVER_Z)
        return goal

    def step(self, ee_pos_w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, float]:
        _, gripper_state, hold, max_steps = self.PHASES[self.phase]
        gripper = self.gripper_goals[gripper_state]
        self.cmd = ee_pos_w.clone() if self.cmd is None else self.cmd
        goal = self._goal(ee_pos_w)
        # Line up tightly above the cube before descending: the fixed jaw only clears the cube by
        # ~3 mm, and a 1 cm miss lands it on top of the cube.
        tol = self.HOVER_TOL if self.phase_name == "hover" else self.tol
        if torch.norm(ee_pos_w - goal, dim=-1).item() < tol:
            self.settled += 1
        self.t += 1
        if (self.settled > hold or self.t >= max_steps) and self.phase < len(self.PHASES) - 1:
            self.phase, self.t, self.settled = self.phase + 1, 0, 0
        # Move the commanded goal at a bounded speed so the arm doesn't fling the cube.
        delta = goal - self.cmd
        self.cmd = self.cmd + delta * (self.MAX_SPEED / delta.norm(dim=-1, keepdim=True).clamp(min=self.MAX_SPEED))
        return self.cmd, self.top_down_quat(self.cmd, self.dyaw), gripper


def collect_episodes(
    env_type: str,
    num_episodes: int,
    output_dir: str,
    use_camera: bool,
    wait_for_key: bool,
    pick_kwargs: dict | None = None,
):
    """Run data collection loop with an IK-based scripted policy.

    ``pick_kwargs`` are passed to :class:`ScriptedPick` (grip_contact, grip_squeeze, grasp_offset).
    """

    livestream = os.environ.get("LIVESTREAM") in ("1", "2")
    # --gui (scripts/_gui.sh) sets HEADLESS=0 and DISPLAY to the Sunshine X display.
    gui = os.environ.get("HEADLESS") == "0" and bool(os.environ.get("DISPLAY"))

    # Kit is always started: cameras need the RTX renderer and livestream needs the
    # WebRTC extension. Isaac Lab 3.0 only opens a window for a Kit visualizer, so the
    # GUI mode requests one (the env's default_visualizer_cfg makes it follow the robot).
    from isaaclab.app import AppLauncher
    app_launcher = AppLauncher(
        headless=not gui,
        enable_cameras=use_camera or livestream or gui,
        visualizer=["kit"] if gui else None,
    )
    sim_app = app_launcher.app

    if livestream:
        import carb
        carb.settings.get_settings().set_bool("/app/livestream/allowResize", True)

    render_mode = "human" if (livestream or gui) else None

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
        robot, env.device, command_type="pose" if env_type == "pick" else "position"
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

    successes = 0
    for ep in range(num_episodes):
        obs, info = env.reset()
        target_w = env.target_pos.clone()
        pick = None
        if env_type == "pick":
            pick = ScriptedPick(env.cube_pos_w, target_w, cube_quat_w=env.cube_quat_w, **(pick_kwargs or {}))
        grip_q_lift, cube_z_max = float("nan"), 0.0
        gripper_goal = None
        target_quat_w = torch.tensor([[0.0, 0.0, 0.0, 1.0]], device=env.device)  # (x, y, z, w)

        ik_controller.reset()

        writer.start_episode(task=lang)
        done = False
        step = 0
        ee_dist = float("nan")

        while not done:
            # Retrieve quantities from the physics engine (Isaac Lab 3.0: ProxyArray -> .torch,
            # poses are pos(3) + quat(4) with quaternions in (x, y, z, w) order)
            jacobian = robot.data.body_link_jacobian_w.torch[:, ee_jacobi_idx, :, arm_joint_ids]

            ee_pose_w = robot.data.body_link_pose_w.torch[:, ee_body_id]
            ee_pos_w = ee_pose_w[:, 0:3]
            ee_quat_w = ee_pose_w[:, 3:7]

            root_pose_w = robot.data.root_link_pose_w.torch
            root_pos_w = root_pose_w[:, 0:3]
            root_quat_w = root_pose_w[:, 3:7]

            arm_joint_pos = robot.data.joint_pos.torch[:, arm_joint_ids]
            if pick is not None:
                target_w, target_quat_w, gripper_goal = pick.step(ee_pos_w)
                # Diagnostics: where the jaw stalled on the cube, and how high the cube got
                if pick.phase_name == "lift" and grip_q_lift != grip_q_lift:
                    grip_q_lift = robot.data.joint_pos.torch[0, -1].item()
                if pick.phase_name in ("lift", "carry"):
                    cube_z_max = max(cube_z_max, env.cube_pos_w[0, 2].item())
            # Measured before stepping: on the final step the env auto-resets the robot.
            ee_dist = torch.norm(ee_pos_w - target_w, dim=-1).item()

            # Transform EE pose from world frame to robot base frame
            ee_pos_b, ee_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w,
                ee_pos_w, ee_quat_w,
            )

            # Transform target from world frame to base frame
            target_b, target_quat_b = subtract_frame_transforms(
                root_pos_w, root_quat_w,
                target_w, target_quat_w,
            )

            # Compute IK: desired arm joint positions to reach the target
            if pick is not None:
                ik_controller.set_command(torch.cat([target_b, target_quat_b], dim=-1))
            else:
                # set_command requires ee_quat even for position-only mode
                ik_controller.set_command(target_b, ee_quat=ee_quat_b)
            arm_joint_target = ik_controller.compute(
                ee_pos_b, ee_quat_b, jacobian, arm_joint_pos
            )
            if pick is not None:
                arm_joint_target = respect_joint_limits(
                    arm_joint_target, arm_joint_pos, jacobian, target_b - ee_pos_b,
                    robot.data.soft_joint_pos_limits.torch[:, arm_joint_ids],
                )

            # Build full 6-DOF joint target (gripper held for reach, scripted for pick)
            full_joint_target = robot.data.joint_pos.torch.clone()
            full_joint_target[:, arm_joint_ids] = arm_joint_target
            if gripper_goal is not None:
                full_joint_target[:, -1] = gripper_goal

            # Convert to delta-action that the environment expects
            current_pos = robot.data.joint_pos.torch
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

        # Success terminates the episode inside env.step (pick: cube within place_threshold
        # of the target), so a terminated episode means success.
        if pick is not None:
            reached = bool(terminated.any())
            status = "PLACED" if reached else f"failed in phase {pick.phase_name}"
            status += f"; jaw {grip_q_lift:.2f} rad at lift, cube up to {cube_z_max * 100:.1f} cm"
        else:
            reached = bool(terminated.any()) or ee_dist < cfg.success_threshold
            status = "REACHED" if reached else f"dist={ee_dist:.3f}m"
        successes += int(reached)
        print(f"Episode {ep + 1}/{num_episodes}: {step} steps  ({status})", flush=True)

    print(f"Success rate: {successes}/{num_episodes}", flush=True)
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
    parser.add_argument("--grip-contact", type=float, default=None,
                        help="Pick: gripper angle (rad) where the jaws touch the cube "
                             f"(default {ScriptedPick.GRIP_CONTACT}).")
    parser.add_argument("--grip-squeeze", type=float, default=None,
                        help="Pick: how far (rad) below the contact angle to command the jaw "
                             f"(default {ScriptedPick.GRIP_SQUEEZE}).")
    parser.add_argument("--grasp-offset", type=float, default=None,
                        help="Pick: cube centre offset (m) along -x of gripper_frame_link "
                             f"(default {ScriptedPick.GRASP_OFFSET}).")
    args = parser.parse_args()

    collect_episodes(
        args.env,
        args.num_episodes,
        args.output_dir,
        use_camera=not args.no_camera,
        wait_for_key=args.wait_for_key,
        pick_kwargs=dict(grip_contact=args.grip_contact, grip_squeeze=args.grip_squeeze,
                         grasp_offset=args.grasp_offset),
    )


if __name__ == "__main__":
    main()
