"""Isaac Lab pick-and-place environment for the SO-ARM101.

The robot must grasp a cube from a random start position and place it at a
target location.  Extends the reach environment with an additional rigid-body
cube and a multi-phase reward.
"""

from __future__ import annotations

import torch
from typing import Sequence

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg, RigidObject
from isaaclab.utils import configclass

from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg
from isaaclab_physx.sim.spawners.materials import PhysxRigidBodyMaterialCfg

from soarm_common import contract as C
from soarm_common.robot import use_convex_decomposition_for_jaws
from soarm_reach_env import IDENTITY_QUAT, SoarmReachEnv, SoarmReachEnvCfg


@configclass
class SoarmPickEnvCfg(SoarmReachEnvCfg):
    """Configuration for the SO-ARM101 pick-and-place task."""

    episode_length_s: float = 10.0
    observation_space: int = C.NUM_JOINTS * 2 + 3 + 3 + 3  # + cube pos + target pos

    # Cube
    # Isaac Lab 3.0 matches prim-path regexes against the whole path, so ".*" would also match
    # every prim below env_0 and the spawner would copy the cube under each of them.
    # "[^/]+" keeps the wildcard to one path segment (what {ENV_REGEX_NS} expands to).
    cube_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_[^/]+/Cube",
        spawn=sim_utils.CuboidCfg(
            size=(0.03, 0.03, 0.03),
            rigid_props=PhysxRigidBodyPropertiesCfg(
                max_depenetration_velocity=1.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.02),
            collision_props=PhysxCollisionPropertiesCfg(),
            # Rubber-ish friction so the cube doesn't slide out of the tapered jaw tips
            physics_material=PhysxRigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=1.0),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 0.015)),
    )

    # Cube spawn area (closer than ~12 cm to the base the gripper can't reach down to the
    # table without hitting joint limits)
    cube_min: Sequence[float] = (0.13, -0.10, 0.015)
    cube_max: Sequence[float] = (0.25, 0.10, 0.015)
    # The place target (green marker) is sampled on the table in the same area, at least
    # min_place_dist from the cube; the reach target bounds (up to 20 cm high) don't apply.
    min_place_dist: float = 0.06

    # Reward weights
    reward_reach_scale: float = -5.0
    reward_grasp_bonus: float = 2.0
    reward_lift_scale: float = 10.0
    reward_place_bonus: float = 10.0
    grasp_threshold: float = 0.03
    # Gripper joint range is [-0.175, 1.745] rad with ~0 closed (the old -0.3 test could never
    # be true); closing on the 3 cm cube the jaw stops at ~0.3 rad.
    gripper_closed_threshold: float = 0.5
    place_threshold: float = 0.03


class SoarmPickEnv(SoarmReachEnv):
    """SO-ARM101 pick-and-place environment."""

    cfg: SoarmPickEnvCfg

    def __init__(self, cfg: SoarmPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._cube: RigidObject = self.scene["cube"]
        self._cube_min = torch.tensor(cfg.cube_min, device=self.device)
        self._cube_max = torch.tensor(cfg.cube_max, device=self.device)

    @property
    def cube_pos_w(self) -> torch.Tensor:
        """Cube position(s) in world frame, shape ``(num_envs, 3)``."""
        return self._cube.data.root_pos_w.torch

    @property
    def cube_quat_w(self) -> torch.Tensor:
        """Cube orientation(s) in world frame, (x, y, z, w), shape ``(num_envs, 4)``."""
        return self._cube.data.root_quat_w.torch

    def _setup_scene(self):
        super()._setup_scene()
        self._use_convex_decomposition_for_jaws()
        self._cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self._cube

    @staticmethod
    def _use_convex_decomposition_for_jaws():
        """Collide the gripper jaws as convex decompositions (see
        :func:`soarm_common.robot.use_convex_decomposition_for_jaws`)."""
        use_convex_decomposition_for_jaws(f"{C.ENV0_PATH}/Robot")

    def _get_observations(self) -> dict:
        # Reuse the reach observations (camera images included) and insert the cube position.
        obs_dict = super()._get_observations()
        joint_pos = self._robot.data.joint_pos.torch
        joint_vel = self._robot.data.joint_vel.torch
        obs_dict["policy"] = torch.cat([joint_pos, joint_vel, self.cube_pos_w, self._target_pos], dim=-1)
        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self.ee_pos_w
        cube_pos = self._cube.data.root_pos_w.torch

        dist_to_cube = torch.norm(ee_pos - cube_pos, dim=-1)
        dist_cube_target = torch.norm(cube_pos - self._target_pos, dim=-1)

        gripper_closed = self._robot.data.joint_pos.torch[:, -1] < self.cfg.gripper_closed_threshold
        cube_grasped = (dist_to_cube < self.cfg.grasp_threshold) & gripper_closed
        cube_lifted = cube_pos[:, 2] > 0.05
        cube_placed = dist_cube_target < self.cfg.place_threshold

        reward = self.cfg.reward_reach_scale * dist_to_cube
        reward += self.cfg.reward_grasp_bonus * cube_grasped.float()
        reward += self.cfg.reward_lift_scale * (cube_lifted & cube_grasped).float()
        reward += self.cfg.reward_place_bonus * cube_placed.float()

        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor]:
        cube_pos = self._cube.data.root_pos_w.torch
        dist_cube_target = torch.norm(cube_pos - self._target_pos, dim=-1)
        success = dist_cube_target < self.cfg.place_threshold

        time_out = self.episode_length_buf >= self.max_episode_length
        return success, time_out

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)

        num = len(env_ids)
        rand = torch.rand(num, 3, device=self.device)
        cube_pos = self._cube_min + rand * (self._cube_max - self._cube_min)
        cube_quat = torch.tensor([IDENTITY_QUAT], device=self.device).expand(num, -1)

        self._cube.write_root_pose_to_sim_index(
            root_pose=torch.cat([cube_pos, cube_quat], dim=-1),
            env_ids=env_ids,
        )
        self._cube.write_root_velocity_to_sim_index(
            root_velocity=torch.zeros(num, 6, device=self.device),
            env_ids=env_ids,
        )

        # Place target on the table, resampled until it is far enough from the cube.
        target = self._cube_min + torch.rand(num, 3, device=self.device) * (self._cube_max - self._cube_min)
        for _ in range(100):
            too_close = torch.norm(target[:, :2] - cube_pos[:, :2], dim=-1) < self.cfg.min_place_dist
            if not too_close.any():
                break
            target[too_close] = self._cube_min + torch.rand(
                int(too_close.sum()), 3, device=self.device
            ) * (self._cube_max - self._cube_min)
        self._target_pos[env_ids] = target
        self._target_marker.visualize(self._target_pos[env_ids])
