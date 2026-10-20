"""Isaac Lab camera configs built from the :class:`soarm_common.contract.CameraSpec` definitions.

Needs Isaac Lab (import inside a launched Isaac Sim app).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.sensors import CameraCfg

from . import contract as C


def camera_prim_path(spec: C.CameraSpec, robot_prim_path: str, env_path: str) -> str:
    """USD prim path of ``spec``: under the EE link of ``robot_prim_path`` for wrist-mounted
    cameras, under ``<env_path>/Cameras`` for fixed ones."""
    if spec.parent == "ee":
        return f"{robot_prim_path}/{C.EE_LINK_REL_PATH}/{spec.prim_name}"
    if spec.parent == "world":
        return f"{env_path}/Cameras/{spec.prim_name}"
    raise ValueError(f"unknown camera parent {spec.parent!r}")


def make_camera_cfg(spec: C.CameraSpec, prim_path: str) -> CameraCfg:
    """RGB pinhole camera for ``spec`` at ``prim_path``, sampled once per control step."""
    return CameraCfg(
        prim_path=prim_path,
        update_period=C.CONTROL_DT,
        height=spec.height,
        width=spec.width,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=spec.focal_length,
            horizontal_aperture=spec.horizontal_aperture,
            clipping_range=spec.clipping_range,
        ),
        offset=CameraCfg.OffsetCfg(pos=spec.pos, rot=spec.rot, convention="world"),
    )
