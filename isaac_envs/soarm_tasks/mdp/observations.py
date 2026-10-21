"""Observation and subtask terms. Poses are in the robot base frame (= env frame here)."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = [
    "ee_pose_b", "object_pose_b", "gripper_pos", "bin_pos_b",
    "fastener_grasped", "fastener_lifted", "fastener_in_bin",
]


def _to_base(env: ManagerBasedRLEnv, pose_w: torch.Tensor, robot_name: str = "robot") -> torch.Tensor:
    from isaaclab.utils.math import subtract_frame_transforms

    root = env.scene[robot_name].data.root_link_pose_w.torch
    pos, quat = subtract_frame_transforms(root[:, :3], root[:, 3:7], pose_w[:, :3], pose_w[:, 3:7])
    return torch.cat([pos, quat], dim=-1)


def ee_pose_b(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """End-effector (``robot_cfg.body_names[0]``, gripper_frame_link) pose: pos(3) + quat xyzw(4)."""
    robot = env.scene[robot_cfg.name]
    return _to_base(env, robot.data.body_link_pose_w.torch[:, robot_cfg.body_ids[0]], robot_cfg.name)


def object_pose_b(env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("fastener")) -> torch.Tensor:
    """Root pose of a rigid object: pos(3) + quat xyzw(4). For a fastener the root is its head's
    bearing face on the axis, and the local +z axis points from the shank to the head."""
    return _to_base(env, env.scene[object_cfg.name].data.root_link_pose_w.torch)


def gripper_pos(env: ManagerBasedRLEnv, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Gripper joint angle (rad, ~0 closed, 1.745 open), shape (N, 1)."""
    robot = env.scene[robot_cfg.name]
    return robot.data.joint_pos.torch[:, robot_cfg.joint_ids]


def bin_pos_b(env: ManagerBasedRLEnv, center: tuple[float, float], floor_z: float) -> torch.Tensor:
    """Bin floor centre (constant), shape (N, 3)."""
    return torch.tensor([[center[0], center[1], floor_z]], device=env.device).expand(env.num_envs, -1)


# --- subtask signals (also used by Isaac Lab Mimic to split demos into subtasks) -----------------

# [start:grasped]
def fastener_grasped(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("fastener"),
    max_dist: float = 0.03,
    open_below: float = 0.5,
    empty_above: float = -0.12,
) -> torch.Tensor:
    """The fastener is between the jaws and the jaws are closed on it.

    ``robot_cfg`` names the EE body and the gripper joint. True when the fastener origin is
    within ``max_dist`` of the EE frame and the gripper angle is between ``empty_above`` (jaws
    shut on nothing: they close past the fastener's width) and ``open_below``.
    """
    robot = env.scene[robot_cfg.name]
    ee = robot.data.body_link_pos_w.torch[:, robot_cfg.body_ids[0]]
    obj = env.scene[object_cfg.name].data.root_link_pos_w.torch
    q = robot.data.joint_pos.torch[:, robot_cfg.joint_ids[0]]
    return (torch.linalg.vector_norm(obj - ee, dim=-1) < max_dist) & (q < open_below) & (q > empty_above)
# [end:grasped]


def fastener_lifted(
    env: ManagerBasedRLEnv, object_cfg: SceneEntityCfg = SceneEntityCfg("fastener"), min_height: float = 0.03
) -> torch.Tensor:
    """The fastener origin is at least ``min_height`` above the table (z = 0 of the env)."""
    obj = env.scene[object_cfg.name].data.root_link_pos_w.torch
    return (obj[:, 2] - env.scene.env_origins[:, 2]) > min_height


# [start:in_bin]
def fastener_in_bin(
    env: ManagerBasedRLEnv,
    center: tuple[float, float],
    inner_size: tuple[float, float],
    top_z: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("fastener"),
) -> torch.Tensor:
    """The fastener origin is inside the bin's inner box (x, y) and below its wall top."""
    obj = env.scene[object_cfg.name].data.root_link_pos_w.torch - env.scene.env_origins
    dx = (obj[:, 0] - center[0]).abs()
    dy = (obj[:, 1] - center[1]).abs()
    return (dx < inner_size[0] / 2) & (dy < inner_size[1] / 2) & (obj[:, 2] < top_z)
# [end:in_bin]
