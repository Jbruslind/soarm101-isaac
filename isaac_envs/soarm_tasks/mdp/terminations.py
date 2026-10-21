"""Termination terms."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

from .observations import fastener_in_bin

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

__all__ = ["fastener_placed"]


# [start:success]
def fastener_placed(
    env: ManagerBasedRLEnv,
    center: tuple[float, float],
    inner_size: tuple[float, float],
    top_z: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("fastener"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_speed: float = 0.05,
    open_above: float = 0.5,
) -> torch.Tensor:
    """Success: the fastener is in the bin, at rest (< ``max_speed`` m/s), and released (gripper
    angle above ``open_above`` rad).

    ``record_demos.py`` and ``replay_demos.py`` call this term outside the termination manager,
    which would otherwise resolve ``robot_cfg`` (the gripper joint id); so resolve it here."""
    if isinstance(robot_cfg.joint_ids, slice):
        robot_cfg.resolve(env.scene)
    obj = env.scene[object_cfg.name]
    speed = torch.linalg.vector_norm(obj.data.root_link_vel_w.torch[:, :3], dim=-1)
    q = env.scene[robot_cfg.name].data.joint_pos.torch[:, robot_cfg.joint_ids[0]]
    in_bin = fastener_in_bin(env, center, inner_size, top_z, object_cfg)
    return in_bin & (speed < max_speed) & (q > open_above)
# [end:success]
