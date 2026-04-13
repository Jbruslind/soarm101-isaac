"""Isaac Lab reach-target environment for the SO-ARM101 6-axis robot arm.

The robot must move its end-effector (gripper_frame_link) to a randomly
sampled target position in the workspace.  Observations include joint
positions, joint velocities, and (optionally) a wrist camera image.
Actions are 6 joint-position deltas plus 1 gripper command.
"""

from __future__ import annotations

import math
import torch
from dataclasses import dataclass, field
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.common import ViewerCfg
from isaaclab.markers import VisualizationMarkersCfg, VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.utils import configclass
from isaaclab.sensors import CameraCfg, Camera

# SO-ARM101 joint names (from URDF, shoulder-to-gripper order)
SOARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]

NUM_ARM_JOINTS = 5  # excludes gripper
NUM_JOINTS = 6      # includes gripper
ARM_JOINT_NAMES = SOARM_JOINT_NAMES[:NUM_ARM_JOINTS]
EE_BODY_NAME = "gripper_frame_link"


@configclass
class SoarmReachEnvCfg(DirectRLEnvCfg):
    """Configuration for the SO-ARM101 reach task."""

    # Simulation
    sim: SimulationCfg = SimulationCfg(dt=1 / 120.0, render_interval=2)
    decimation: int = 4  # policy runs at 30 Hz

    # Episode
    episode_length_s: float = 5.0
    num_envs: int = 1

    # Observation / action spaces
    observation_space: int = NUM_JOINTS * 2 + 3  # joint pos + vel + target xyz
    action_space: int = NUM_JOINTS  # 5 arm joints + 1 gripper

    # Scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1, env_spacing=1.5)

    # Third-person viewport camera (follows robot when run with GUI / render_mode="human")
    viewer: ViewerCfg = ViewerCfg(
        origin_type="asset_root",
        asset_name="robot",
        eye=(0.8, 0.8, 0.5),
        lookat=(0.0, 0.0, 0.2),
    )

    # Robot (USD from URDF import has root at Robot/so101_new_calib; point to it so
    # only one articulation is found — the USD may also contain so101_new_calib_01)
    robot_cfg: ArticulationCfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot/so101_new_calib",
        spawn=sim_utils.UsdFileCfg(
            usd_path="/robot_description/usd/soarm101.usd",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": 0.0,
                "elbow_flex": 0.0,
                "wrist_flex": 0.0,
                "wrist_roll": 0.0,
                "gripper": 0.0,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex",
                                  "wrist_flex", "wrist_roll"],
                velocity_limit_sim=2.0,
                effort_limit_sim=30.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                velocity_limit_sim=2.0,
                effort_limit_sim=10.0,
                stiffness=80.0,
                damping=4.0,
            ),
        },
    )

    # Camera (optional, for VLA data collection; under same articulation root)
    use_camera: bool = False
    camera_cfg: CameraCfg = CameraCfg(
        prim_path="/World/envs/env_.*/Robot/so101_new_calib/gripper_frame_link/wrist_cam",
        update_period=0.0333,  # 30 Hz
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            horizontal_aperture=2.65,
        ),
        offset=CameraCfg.OffsetCfg(
            pos=(0.05, 0.0, 0.02),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # Third-person camera (optional, for visual verification in headless runs)
    use_third_person_camera: bool = False
    third_person_camera_cfg: CameraCfg = CameraCfg(
        # Note: The camera spawner expects the parent prim to exist and does not accept regex prim paths.
        # Our current collector uses num_envs=1, so we place it under env_0.
        prim_path="/World/envs/env_0/Cameras/third_person_cam",
        update_period=0.0333,  # 30 Hz
        height=224,
        width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=1.93,
            horizontal_aperture=2.65,
        ),
        # Placed in world frame, looking roughly toward the robot workspace.
        offset=CameraCfg.OffsetCfg(
            pos=(0.8, 0.8, 0.5),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="world",
        ),
    )

    # Target position sampling bounds (meters, relative to robot base)
    target_min: Sequence[float] = (0.05, -0.15, 0.02)
    target_max: Sequence[float] = (0.25, 0.15, 0.20)

    # Reward
    reward_dist_scale: float = -10.0
    reward_success_bonus: float = 5.0
    success_threshold: float = 0.02  # 2 cm

    # Action scaling
    action_scale: float = 0.05  # radians per step


class SoarmReachEnv(DirectRLEnv):
    """SO-ARM101 reach-target environment."""

    cfg: SoarmReachEnvCfg

    def __init__(self, cfg: SoarmReachEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self._target_pos = torch.zeros(self.num_envs, 3, device=self.device)

        self._robot: Articulation = self.scene["robot"]
        self._camera: Camera | None = self.scene.sensors["wrist_cam"] if "wrist_cam" in self.scene.sensors else None
        self._third_person_camera: Camera | None = (
            self.scene.sensors["third_person_cam"] if "third_person_cam" in self.scene.sensors else None
        )

        self._target_min = torch.tensor(cfg.target_min, device=self.device)
        self._target_max = torch.tensor(cfg.target_max, device=self.device)

    @property
    def robot(self) -> Articulation:
        """The robot articulation asset (for IK controllers, Jacobians, etc.)."""
        return self._robot

    @property
    def target_pos(self) -> torch.Tensor:
        """Current target position(s), shape ``(num_envs, 3)``."""
        return self._target_pos

    def _setup_scene(self):
        # 1) Spawn the robot USD into the stage manually.
        self.cfg.robot_cfg.spawn.func(
            "/World/envs/env_0/Robot", self.cfg.robot_cfg.spawn
        )

        # 2) The URDF→USD converter can produce duplicate roots (so101_new_calib_01).
        #    They come from a reference, so we deactivate the duplicate instead of removing.
        import omni.usd
        from pxr import Usd, UsdPhysics
        stage = omni.usd.get_context().get_stage()
        robot_prim = stage.GetPrimAtPath("/World/envs/env_0/Robot")
        if robot_prim.IsValid():
            for child in robot_prim.GetChildren():
                if child.GetName() != "so101_new_calib":
                    child.SetActive(False)

        # 3) Create the Articulation wrapper *without* spawning again.
        saved_spawn = self.cfg.robot_cfg.spawn
        self.cfg.robot_cfg.spawn = None
        self._robot = Articulation(self.cfg.robot_cfg)
        self.cfg.robot_cfg.spawn = saved_spawn
        self.scene.articulations["robot"] = self._robot

        if self.cfg.use_camera:
            self._camera = Camera(self.cfg.camera_cfg)
            # Ensure base-class attributes exist so update() doesn't raise if init failed.
            if not hasattr(self._camera, "_timestamp"):
                self._camera._timestamp = 0.0
            if not hasattr(self._camera, "_is_outdated"):
                self._camera._is_outdated = False
            self.scene.sensors["wrist_cam"] = self._camera

        if self.cfg.use_third_person_camera:
            # The camera spawner requires the parent prim to exist.
            import omni.usd
            from pxr import Usd

            stage = omni.usd.get_context().get_stage()
            stage.DefinePrim("/World/envs/env_0/Cameras", "Xform")

            self._third_person_camera = Camera(self.cfg.third_person_camera_cfg)
            if not hasattr(self._third_person_camera, "_timestamp"):
                self._third_person_camera._timestamp = 0.0
            if not hasattr(self._third_person_camera, "_is_outdated"):
                self._third_person_camera._is_outdated = False
            self.scene.sensors["third_person_cam"] = self._third_person_camera

        # Ground plane (local spawner – avoids Nucleus dependency)
        ground_cfg = sim_utils.CuboidCfg(
            size=(100.0, 100.0, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
        )
        ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.01))

        # Lighting
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/light", light_cfg)

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
        clamped = actions.clamp(-1.0, 1.0)
        scaled = clamped * self.cfg.action_scale

        current_pos = self._robot.data.joint_pos
        target_pos = current_pos + scaled
        self._robot.set_joint_position_target(target_pos)

    def _apply_action(self) -> None:
        """Apply the current action (from self.actions) to the robot. Required by DirectRLEnv."""
        action = self.actions
        clamped = action.clamp(-1.0, 1.0)
        scaled = clamped * self.cfg.action_scale

        current_pos = self._robot.data.joint_pos
        target_pos = current_pos + scaled
        self._robot.set_joint_position_target(target_pos)

    def _get_observations(self) -> dict:
        joint_pos = self._robot.data.joint_pos
        joint_vel = self._robot.data.joint_vel

        obs = torch.cat([joint_pos, joint_vel, self._target_pos], dim=-1)

        obs_dict = {"policy": obs}

        if self._camera is not None:
            obs_dict["wrist_rgb"] = self._camera.data.output["rgb"]
        if self._third_person_camera is not None:
            obs_dict["third_person_rgb"] = self._third_person_camera.data.output["rgb"]

        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self._robot.data.body_pos_w[:, -2, :]  # gripper_frame_link
        dist = torch.norm(ee_pos - self._target_pos, dim=-1)

        reward = self.cfg.reward_dist_scale * dist
        reward += self.cfg.reward_success_bonus * (dist < self.cfg.success_threshold).float()
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        ee_pos = self._robot.data.body_pos_w[:, -2, :]
        dist = torch.norm(ee_pos - self._target_pos, dim=-1)
        success = dist < self.cfg.success_threshold

        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        # Randomize joint positions near zero
        num = len(env_ids)
        noise = torch.randn(num, NUM_JOINTS, device=self.device) * 0.1
        default_pos = torch.zeros(num, NUM_JOINTS, device=self.device)
        self._robot.write_joint_state_to_sim(
            default_pos + noise,
            torch.zeros(num, NUM_JOINTS, device=self.device),
            env_ids=env_ids,
        )

        # Randomize target position
        rand = torch.rand(num, 3, device=self.device)
        self._target_pos[env_ids] = (
            self._target_min + rand * (self._target_max - self._target_min)
        )

        # Update marker
        self._target_marker.visualize(self._target_pos[env_ids])
