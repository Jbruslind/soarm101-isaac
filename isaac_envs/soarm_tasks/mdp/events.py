"""Event terms: start-up asset fixes, reset of the fastener, lighting randomization."""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv

__all__ = [
    "jaws_convex_decomposition", "reset_fastener_lying", "randomize_dome_light",
    "LYING_QUAT", "fastener_names", "language_instructions",
]

# Fastener lying on its side: +90 deg about x turns the fastener axis (local +z) to world -y.
LYING_QUAT = (math.sin(math.pi / 4), 0.0, 0.0, math.cos(math.pi / 4))  # (x, y, z, w)


def jaws_convex_decomposition(env: ManagerBasedEnv, env_ids, robot_cfg: SceneEntityCfg = SceneEntityCfg("robot")):
    """``prestartup``: collide the jaws of every env's robot as convex decompositions
    (:func:`soarm_common.robot.use_convex_decomposition_for_jaws`)."""
    from isaaclab.sim.utils import find_matching_prim_paths

    from soarm_common.robot import use_convex_decomposition_for_jaws

    prim_expr = env.scene.cfg.robot.prim_path.replace("{ENV_REGEX_NS}", "/World/envs/env_[^/]+")
    paths = find_matching_prim_paths(prim_expr)
    meshes = sum(use_convex_decomposition_for_jaws(path) for path in paths)
    print(f"[soarm_tasks] jaw colliders -> convexDecomposition: {meshes} meshes in {len(paths)} robots", flush=True)


# [start:reset_fastener]
def reset_fastener_lying(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    yaw_range: tuple[float, float],
    spin_range: tuple[float, float],
    z: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("fastener"),
):
    """Put the fastener on its side at a random (x, y), axis yaw and spin about its own axis.

    Orientation = yaw about world z * LYING_QUAT * spin about the fastener axis.
    """
    from isaaclab.utils.math import quat_from_angle_axis, quat_mul, sample_uniform

    asset = env.scene[asset_cfg.name]
    if env_ids is None or isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device)
    n, dev = len(env_ids), env.device
    pos = torch.stack([sample_uniform(*x_range, (n,), dev), sample_uniform(*y_range, (n,), dev),
                       torch.full((n,), z, device=dev)], dim=-1)
    pos = pos + env.scene.env_origins[env_ids]
    z_axis = torch.tensor([[0.0, 0.0, 1.0]], device=dev).expand(n, -1)
    # The fastener axis points along world -y after LYING_QUAT, i.e. at yaw -90 deg; shift so that
    # yaw is the world yaw of the axis (shank -> head).
    q_yaw = quat_from_angle_axis(sample_uniform(*yaw_range, (n,), dev) + math.pi / 2, z_axis)
    q_spin = quat_from_angle_axis(sample_uniform(*spin_range, (n,), dev), z_axis)
    q_lay = torch.tensor([LYING_QUAT], device=dev).expand(n, -1)
    quat = quat_mul(quat_mul(q_yaw, q_lay), q_spin)
    asset.write_root_pose_to_sim_index(root_pose=torch.cat([pos, quat], dim=-1), env_ids=env_ids)
    asset.write_root_velocity_to_sim_index(root_velocity=torch.zeros(n, 6, device=dev), env_ids=env_ids)
# [end:reset_fastener]


def randomize_dome_light(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    intensity_range: tuple[float, float],
    color_variation: float,
    default_color: tuple[float, float, float] = (0.75, 0.75, 0.75),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("light"),
):
    """Sample the (global) dome light's intensity and a brightness-preserving tint."""
    prim = env.scene[asset_cfg.name].prim
    prim.GetAttribute("inputs:intensity").Set(random.uniform(*intensity_range))
    offsets = [random.uniform(-color_variation, color_variation) for _ in range(3)]
    mean = sum(offsets) / 3
    prim.GetAttribute("inputs:color").Set(
        tuple(min(1.0, max(0.0, c + o - mean)) for c, o in zip(default_color, offsets)))


# --- language ---------------------------------------------------------------------------------

def fastener_names(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("fastener")) -> list[str]:
    """Per-env fastener name (e.g. 'hex_bolt_M6x20'), read from the ``fastener:*`` attributes
    each spawned fastener USD carries, so it is right for a MultiAssetSpawnerCfg mix too."""
    import omni.usd
    from pxr import Usd

    from ..language import fastener_name_from_attrs

    stage = omni.usd.get_context().get_stage()
    leaf = env.scene[asset_cfg.name].cfg.prim_path.rsplit("/", 1)[1]
    names = []
    for i in range(env.num_envs):
        root = stage.GetPrimAtPath(f"/World/envs/env_{i}/{leaf}")
        attrs = {}
        for prim in Usd.PrimRange(root):
            if prim.HasAttribute("fastener:type"):
                attrs = {k: prim.GetAttribute(f"fastener:{k}").Get() for k in ("type", "size", "lengthMm")}
                break
        names.append(fastener_name_from_attrs(attrs["type"], attrs["size"], attrs["lengthMm"]) if attrs else "")
    return names


def language_instructions(env: ManagerBasedEnv, target: str = "the bin") -> list[str]:
    """Per-env prompt, e.g. 'pick up the M6 hex bolt and place it in the bin'."""
    from ..language import instruction_for

    return [instruction_for(n, target) if n else env.cfg.language_instruction for n in fastener_names(env)]
