"""Geometry of the fastener pick-and-place work cell (plain Python, no Isaac imports).

All poses are in the env frame, which is also the robot base frame: the SO-ARM101 base sits at the
env origin on the table top (z = 0) and faces +x. Lengths in metres.

Part 1 uses one bin (:data:`BIN`). Part 2 adds bins by defining more :class:`BinSpec` and one scene
entity per bin (see ``pick_place_env_cfg.make_bin_cfg``).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

# [start:table]
# Table: a static box whose top is the z = 0 plane the robot stands on. It is larger than the
# overhead camera's footprint, so the camera never sees past its edge.
TABLE_SIZE = (1.0, 1.0, 0.04)
TABLE_CENTER = (0.20, 0.0, -TABLE_SIZE[2] / 2)
TABLE_COLOR = (0.10, 0.10, 0.11)  # near-black: the zinc-plated bolt (0.72 grey) stands out
# [end:table]


@dataclass
class BinSpec:
    """An open-top tray made of five static boxes (floor + four walls).

    (Not frozen: Isaac Lab configclasses walk and rewrite nested dataclass fields.)
    """

    name: str
    center: tuple[float, float]           # (x, y) of the bin centre on the table
    inner_size: tuple[float, float]       # inside length along x and y
    wall_height: float                    # wall top above the table (z)
    wall_thickness: float = 0.004
    floor_thickness: float = 0.004
    color: tuple[float, float, float] = (0.10, 0.30, 0.70)

    @property
    def floor_top(self) -> float:
        return self.floor_thickness

    def parts(self) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
        """(name, box size, box centre relative to the bin origin on the table) of each part."""
        ix, iy = self.inner_size
        t, h, f = self.wall_thickness, self.wall_height, self.floor_thickness
        ox, oy = ix + 2 * t, iy + 2 * t
        return [
            ("floor", (ox, oy, f), (0.0, 0.0, f / 2)),
            ("wall_xp", (t, oy, h), ((ix + t) / 2, 0.0, h / 2)),
            ("wall_xn", (t, oy, h), (-(ix + t) / 2, 0.0, h / 2)),
            ("wall_yp", (ix, t, h), (0.0, (iy + t) / 2, h / 2)),
            ("wall_yn", (ix, t, h), (0.0, -(iy + t) / 2, h / 2)),
        ]


# [start:bin]
# One bin to the robot's left (+y), out of the fastener spawn area. 9 x 9 cm inside, 2.5 cm walls:
# low enough that the overhead camera sees the floor, high enough to stop a dropped bolt.
BIN = BinSpec(name="bin", center=(0.20, 0.14), inner_size=(0.09, 0.09), wall_height=0.025)
# [end:bin]

# [start:spawn]
# Fastener spawn area (env frame, on the table, to the robot's right). It stays at least 12 cm
# out from the base, where the top-down gripper reaches the table without joint-limit trouble
# (same reasoning as the cube spawn area of SoarmPickEnv), and 5 cm clear of the bin.
DEFAULT_FASTENER = "hex_bolt_M6x20"
FASTENER_SPAWN_X = (0.15, 0.25)
FASTENER_SPAWN_Y = (-0.12, -0.02)
FASTENER_SPAWN_YAW = (-math.pi, math.pi)  # yaw of the bolt axis, full circle
# Height of the fastener origin at reset: the bolt lies on its side (axis horizontal), its
# origin (head bearing face, on the axis) a few mm above its rest height. It drops < 1 mm.
FASTENER_SPAWN_Z = 0.0065
# [end:spawn]
