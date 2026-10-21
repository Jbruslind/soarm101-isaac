"""Isaac Lab reach-target environment for the SO-ARM101 6-axis robot arm.

The robot must move its end-effector (gripper_frame_link) to a randomly
sampled target position in the workspace.  Observations include joint
positions, joint velocities, and (optionally) the wrist and overhead camera images.
Actions are 6 normalised joint-position deltas (5 arm joints + gripper); see
``soarm_common.contract`` for how they relate to the absolute joint targets that are
recorded and sent by policies.

Robot, cameras, timing and keys come from ``soarm_common`` (one definition shared with
``interactive_inference.py``, the data collector and the ROS 2 bridge).

Isaac Lab 3.0 / Isaac Sim 6.2: quaternions are (x, y, z, w), asset and sensor data
are ProxyArray (use ``.torch``), and the viewport camera is a Kit visualizer
(``SimulationCfg.default_visualizer_cfg``) opened with ``--viz kit``.
"""

from __future__ import annotations

import torch
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, Camera
from isaaclab_visualizers.kit import KitVisualizerCfg

from soarm_common import contract as C
from soarm_common.cameras import camera_prim_path, make_camera_cfg
from soarm_common.robot import check_joint_limits, make_physx_cfg, make_robot_cfg
from soarm_common.scene import spawn_ground_and_light

IDENTITY_QUAT = (0.0, 0.0, 0.0, 1.0)  # (x, y, z, w)
_ENV_REGEX_PATH = C.ENV_ROBOT_PRIM_PATH.rsplit("/", 1)[0]  # /World/envs/env_.*


class SoarmEnvWindow(BaseEnvWindow):
    """IsaacLab UI window whose "Follow Mode" dropdown matches the Kit visualizer config.

    Isaac Lab 3.0's ``BaseEnvWindow`` always builds that dropdown on "World", whatever
    ``KitVisualizerCfg.origin_type`` says, so the window claimed "World" while the camera was
    tracking the robot. This selects the configured entry instead.
    """

    def _build_viewer_frame(self):
        super()._build_viewer_frame()
        viz = self._get_kit_visualizer()
        if viz is None:
            return
        if viz.cfg.origin_type == "env":
            option = "Env"
        elif viz.cfg.origin_type == "asset" and viz.cfg.origin_track_path:
            option = viz.cfg.origin_track_path.partition("/")[0]
        else:
            return
        if option not in self._viewer_assets_options:
            return
        # Selecting the entry runs the UI callback, which resets origin_track_path to the bare
        # asset name; keep a configured body ("robot/<body>").
        track_path = viz.cfg.origin_track_path
        model = self.ui_window_elements["viewer_follow"]
        model.get_item_value_model().set_value(self._viewer_assets_options.index(option))
        viz.cfg.origin_track_path = track_path


@configclass
class SoarmReachEnvCfg(DirectRLEnvCfg):
    """Configuration for the SO-ARM101 reach task."""

    # Simulation (contract timing: 120 Hz physics, 30 Hz control). The viewport camera follows the
    # robot when a Kit visualizer is open (--viz kit, e.g. via scripts/isaac.sh --gui).
    sim: SimulationCfg = SimulationCfg(
        dt=C.PHYSICS_DT,
        render_interval=C.RENDER_INTERVAL,
        physics=make_physx_cfg(),
        default_visualizer_cfg=KitVisualizerCfg(
            origin_type="asset",
            origin_track_path="robot",
            eye=(0.8, 0.8, 0.5),
            lookat=(0.0, 0.0, 0.2),
        ),
    )
    decimation: int = C.DECIMATION  # policy runs at C.CONTROL_HZ (30 Hz)
    ui_window_class_type: type | None = SoarmEnvWindow

    # Episode
    episode_length_s: float = 5.0
    num_envs: int = 1

    # Observation / action spaces
    observation_space: int = C.NUM_JOINTS * 2 + 3  # joint pos + vel + target xyz
    action_space: int = C.NUM_JOINTS  # 5 arm joints + 1 gripper

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5)

    # Robot (soarm_common.robot): the spawned prim is the articulation root
    robot_cfg: ArticulationCfg = make_robot_cfg(C.ENV_ROBOT_PRIM_PATH)

    # Wrist camera (soarm_common.contract.WRIST_CAMERA), under the EE link of every env's robot
    use_wrist_camera: bool = False
    wrist_camera_cfg: CameraCfg = make_camera_cfg(
        C.WRIST_CAMERA, camera_prim_path(C.WRIST_CAMERA, C.ENV_ROBOT_PRIM_PATH, _ENV_REGEX_PATH)
    )

    # Overhead camera (soarm_common.contract.OVERHEAD_CAMERA). The camera spawner needs an existing
    # parent prim and doesn't accept regex paths, so it lives under env_0 (the collector uses
    # num_envs=1, whose env_0 origin is the world origin).
    use_overhead_camera: bool = False
    overhead_camera_cfg: CameraCfg = make_camera_cfg(
        C.OVERHEAD_CAMERA, camera_prim_path(C.OVERHEAD_CAMERA, f"{C.ENV0_PATH}/Robot", C.ENV0_PATH)
    )

    # Target position sampling bounds (meters, relative to robot base)
    target_min: Sequence[float] = (0.05, -0.15, 0.02)
    target_max: Sequence[float] = (0.25, 0.15, 0.20)

    # Reward
    reward_dist_scale: float = -10.0
    reward_success_bonus: float = 5.0
    success_threshold: float = 0.02  # 2 cm

    # Action scaling
    action_scale: float = C.ENV_DELTA_ACTION_SCALE  # rad per step for an action of 1


class SoarmReachEnv(DirectRLEnv):
    """SO-ARM101 reach-target environment."""

    cfg: SoarmReachEnvCfg

    def __init__(self, cfg: SoarmReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self._robot: Articulation = self.scene["robot"]
        self._joint_pos_target: torch.Tensor | None = None

        # Resolve the end-effector by name; body order in the imported USD puts the moving
        # jaw just before gripper_frame_link, so a fixed index is easy to get wrong.
        self._ee_body_idx = self._robot.find_bodies(C.EE_BODY_NAME)[0][0]
        check_joint_limits(self._robot)

        self._target_min = torch.tensor(cfg.target_min, device=self.device)
        self._target_max = torch.tensor(cfg.target_max, device=self.device)

    @property
    def robot(self) -> Articulation:
        """The robot articulation asset (for IK controllers, Jacobians, etc.)."""
        return self._robot

    @property
    def ee_pos_w(self) -> torch.Tensor:
        """End-effector (gripper_frame_link) position(s) in world frame, shape ``(num_envs, 3)``."""
        return self._robot.data.body_pos_w.torch[:, self._ee_body_idx, :]

    @property
    def joint_pos_target(self) -> torch.Tensor | None:
        """Absolute joint-position targets (rad) applied in the last step, shape ``(num_envs, 6)``.

        This is the contract action (``soarm_common.contract.ACTION_TYPE``) equivalent to the
        delta action passed to :meth:`step`; the data collector records it.
        """
        return self._joint_pos_target

    @property
    def target_pos(self) -> torch.Tensor:
        """Current target position(s), shape ``(num_envs, 3)``."""
        return self._target_pos

    def _setup_scene(self):
        # 1) Spawn the robot USD into the stage manually.
        self.cfg.robot_cfg.spawn.func(f"{C.ENV0_PATH}/Robot", self.cfg.robot_cfg.spawn)

        # 2) Create the Articulation wrapper *without* spawning again.
        saved_spawn = self.cfg.robot_cfg.spawn
        self.cfg.robot_cfg.spawn = None
        self._robot = Articulation(self.cfg.robot_cfg)
        self.cfg.robot_cfg.spawn = saved_spawn
        self.scene.articulations["robot"] = self._robot

        self._cameras: dict[str, Camera] = {}
        if self.cfg.use_overhead_camera:
            # The camera spawner requires the parent prim to exist.
            import omni.usd

            omni.usd.get_context().get_stage().DefinePrim(f"{C.ENV0_PATH}/Cameras", "Xform")
        for spec, use, cam_cfg in (
            (C.WRIST_CAMERA, self.cfg.use_wrist_camera, self.cfg.wrist_camera_cfg),
            (C.OVERHEAD_CAMERA, self.cfg.use_overhead_camera, self.cfg.overhead_camera_cfg),
        ):
            if not use:
                continue
            cam = Camera(cam_cfg)
            # Ensure base-class attributes exist so update() doesn't raise if init failed.
            if not hasattr(cam, "_timestamp"):
                cam._timestamp = 0.0
            if not hasattr(cam, "_is_outdated"):
                cam._is_outdated = False
            self.scene.sensors[spec.prim_name] = cam
            self._cameras[spec.obs_key] = cam

        # Ground plane and lighting (shared with interactive_inference.py)
        spawn_ground_and_light()

        # Target marker
        self._target_marker = VisualizationMarkers(
            VisualizationMarkersCfg(
                prim_path="/Visuals/target",
                markers={
                    "sphere": sim_utils.SphereCfg(
                        radius=0.015,
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=(0.0, 1.0, 0.0),
                        ),
                    ),
                },
            )
        )

    def _pre_physics_step(self, actions: torch.Tensor):
        # Joint-position deltas, applied once per policy step (action_scale rad/step)
        # and held for the `decimation` physics substeps.
        self.actions = actions.clamp(-1.0, 1.0)
        self._joint_pos_target = self._robot.data.joint_pos.torch + self.actions * self.cfg.action_scale

    def _apply_action(self) -> None:
        """Apply the joint targets computed in _pre_physics_step. Required by DirectRLEnv."""
        self._robot.actuators.target_command.set_position_index(value=self._joint_pos_target)

    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos.torch
        joint_vel = self._robot.data.joint_vel.torch

        obs = torch.cat([joint_pos, joint_vel, self._target_pos], dim=-1)

        obs_dict = {"policy": obs}

        # Camera images under the contract keys (CameraSpec.obs_key): "wrist_rgb", "overhead_rgb"
        for key, cam in self._cameras.items():
            obs_dict[key] = cam.data.output["rgb"].torch

        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self.ee_pos_w
        dist = torch.norm(ee_pos - self._target_pos, dim=-1)

        reward = self.cfg.reward_dist_scale * dist
        reward += self.cfg.reward_success_bonus * (dist < self.cfg.success_threshold).float()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos = self.ee_pos_w
        dist = torch.norm(ee_pos - self._target_pos, dim=-1)
        success = dist < self.cfg.success_threshold

        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # Randomize joint positions near zero
        num = len(env_ids)
        noise = torch.randn(num, C.NUM_JOINTS, device=self.device) * 0.1
        default_pos = torch.zeros(num, C.NUM_JOINTS, device=self.device)
        self._robot.write_joint_state_to_sim_index(
            position=default_pos + noise,
            velocity=torch.zeros(num, C.NUM_JOINTS, device=self.device),
            env_ids=env_ids,
        )

        # Randomize target position
        rand = torch.rand(num, 3, device=self.device)
        self._target_pos[env_ids] = (
            self._target_min + rand * (self._target_max - self._target_min)
        )

        # Update marker
        self._target_marker.visualize(self._target_pos[env_ids])
