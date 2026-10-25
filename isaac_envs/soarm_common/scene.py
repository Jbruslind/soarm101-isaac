"""Static scene shared by the envs and interactive_inference.py: ground plane and lighting.

Both must match for the cameras to render the same images in data collection and inference.
Needs Isaac Lab (import inside a launched Isaac Sim app).
"""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

GROUND_PRIM_PATH = "/World/ground"
LIGHT_PRIM_PATH = "/World/light"


def spawn_ground_and_light() -> None:
    """A 100 m grey kinematic slab with its top at z = 0, and a dome light."""
    ground_cfg = sim_utils.CuboidCfg(
        size=(100.0, 100.0, 0.02),
        rigid_props=PhysxRigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=PhysxCollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    ground_cfg.func(GROUND_PRIM_PATH, ground_cfg, translation=(0.0, 0.0, -0.01))
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func(LIGHT_PRIM_PATH, light_cfg)
