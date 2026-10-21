"""Procedural spawners for the work cell: an open-top bin and a plain Xform.

The config classes are pure data (importable without Kit); the spawn functions import ``pxr``
and Isaac Lab's spawners lazily, like Isaac Lab's own ``"{DIR}.module:func"`` spawners.
"""

from __future__ import annotations

from dataclasses import MISSING

from isaaclab.sim.spawners.spawner_cfg import SpawnerCfg
from isaaclab.utils import configclass

from .work_cell import BinSpec


@configclass
class BinCfg(SpawnerCfg):
    """Static open-top bin built from five collision boxes (see :class:`work_cell.BinSpec`)."""

    func: str = "soarm_tasks.spawners:spawn_bin"
    spec: BinSpec = MISSING


@configclass
class XformCfg(SpawnerCfg):
    """An empty Xform, e.g. the per-env ``Cameras`` parent the camera spawner needs."""

    func: str = "soarm_tasks.spawners:spawn_xform"


def _cloned(func):
    """Apply Isaac Lab's ``clone`` decorator lazily (it imports pxr)."""

    def wrapper(prim_path, cfg, *args, **kwargs):
        from isaaclab.sim.utils import clone

        return clone(func)(prim_path, cfg, *args, **kwargs)

    wrapper.__name__ = func.__name__
    return wrapper


@_cloned
def spawn_xform(prim_path: str, cfg: XformCfg, translation=None, orientation=None, **kwargs):
    from isaaclab.sim.utils import create_prim

    return create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)


# [start:spawn_bin]
@_cloned
def spawn_bin(prim_path: str, cfg: BinCfg, translation=None, orientation=None, **kwargs):
    """Xform at ``prim_path`` with one static, colliding box per bin part (no rigid body:
    PhysX treats colliders without a rigid body as static geometry)."""
    import isaaclab.sim as sim_utils
    from isaaclab.sim.utils import create_prim
    from isaaclab_physx.sim.schemas import PhysxCollisionCfg

    prim = create_prim(prim_path, prim_type="Xform", translation=translation, orientation=orientation)
    for name, size, center in cfg.spec.parts():
        part = sim_utils.CuboidCfg(
            size=size,
            collision_props=[sim_utils.UsdPhysicsCollisionCfg(), PhysxCollisionCfg(contact_offset=0.002, rest_offset=0.0)],
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=cfg.spec.color, roughness=0.6),
        )
        part.func(f"{prim_path}/{name}", part, translation=center)
    return prim
# [end:spawn_bin]
