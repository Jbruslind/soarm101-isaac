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

from soarm_reach_env import (
    IDENTITY_QUAT,
    SoarmReachEnv,
    SoarmReachEnvCfg,
    NUM_JOINTS,
)


@configclass
class SoarmPickEnvCfg(SoarmReachEnvCfg):
    """Configuration for the SO-ARM101 pick-and-place task."""

    episode_length_s: float = 10.0
    observation_space: int = NUM_JOINTS * 2 + 3 + 3 + 3  # + cube pos + target pos

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
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(1.0, 0.2, 0.2),
            ),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.15, 0.0, 0.015)),
    )

    # Cube spawn area
    cube_min: Sequence[float] = (0.08, -0.10, 0.015)
    cube_max: Sequence[float] = (0.22, 0.10, 0.015)

    # Reward weights
    reward_reach_scale: float = -5.0
    reward_grasp_bonus: float = 2.0
    reward_lift_scale: float = 10.0
    reward_place_bonus: float = 10.0
    grasp_threshold: float = 0.03
    place_threshold: float = 0.03


class SoarmPickEnv(SoarmReachEnv):
    """SO-ARM101 pick-and-place environment."""

    cfg: SoarmPickEnvCfg

    def __init__(self, cfg: SoarmPickEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._cube: RigidObject = self.scene["cube"]
        self._cube_min = torch.tensor(cfg.cube_min, device=self.device)
        self._cube_max = torch.tensor(cfg.cube_max, device=self.device)

    def _setup_scene(self):
        super()._setup_scene()
        self._cube = RigidObject(self.cfg.cube_cfg)
        self.scene.rigid_objects["cube"] = self._cube

    def _get_observations(self) -> dict:
        # Reuse the reach observations (camera images included) and insert the cube position.
        obs_dict = super()._get_observations()
        joint_pos = self._robot.data.joint_pos.torch
        joint_vel = self._robot.data.joint_vel.torch
        cube_pos = self._cube.data.root_pos_w.torch
        obs_dict["policy"] = torch.cat([joint_pos, joint_vel, cube_pos, self._target_pos], dim=-1)
        return obs_dict

    def _get_rewards(self) -> torch.Tensor:
        ee_pos = self.ee_pos_w
        cube_pos = self._cube.data.root_pos_w.torch

        dist_to_cube = torch.norm(ee_pos - cube_pos, dim=-1)
        dist_cube_target = torch.norm(cube_pos - self._target_pos, dim=-1)

        gripper_closed = self._robot.data.joint_pos.torch[:, -1] < -0.3
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
