"""Action-term configs of the teleop variant (pure configuration, importable without Kit).

The live term is :class:`soarm_tasks.mdp.actions.IntegratedPoseIKAction`. Like Isaac Lab's
``SO101PoseIKActionCfg`` it is referenced by a ``"{DIR}.actions:..."`` string, because the term
subclasses an IK action that imports ``pxr`` when it is loaded.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from isaaclab.utils import configclass
from isaaclab_tasks.contrib.stack.config.so101.pose_ik_action import SO101PoseIKActionCfg

if TYPE_CHECKING:
    from .actions import IntegratedPoseIKAction


# [start:integrated_cfg]
@configclass
class IntegratedPoseIKActionCfg(SO101PoseIKActionCfg):
    """Delta EE pose in, absolute EE pose IK out (for Se3Keyboard / Se3SpaceMouse teleop).

    The action is ``[dx, dy, dz, rx, ry, rz]`` (m and rad per control step, robot base frame), the
    format of Isaac Lab's SE(3) teleop devices. The term adds each delta to an EE *target* pose that
    it keeps per env, and gives that target to the same absolute-pose IK as the IK-Abs task.
    ``controller.use_relative_mode`` must stay False.

    After every reset the target is ``start_pos`` with the gripper down at ``start_yaw``, so the
    arm first moves to a ready pose above the table (the reset pose is upright and close to a
    singularity, where small IK steps get stuck). Keys cannot push the target more than
    ``max_lead`` from the measured EE or out of the ``workspace`` box, so holding a key against a
    limit does not wind it up.
    """

    class_type: type[IntegratedPoseIKAction] | str = "{DIR}.actions:IntegratedPoseIKAction"

    top_down: bool = True
    """Keep the gripper pointing straight down: only the yaw delta (``rz``) is used and roll/pitch
    deltas are ignored. With False the full rotation vector is applied to the target orientation."""

    start_pos: tuple[float, float, float] = (0.20, -0.04, 0.10)
    """m, target position after a reset (robot base frame): above the fastener spawn area."""

    start_yaw: float = 3.14159
    """rad, gripper x axis yaw after a reset (pi: pointing back at the base, like the expert)."""

    max_lead: float = 0.03
    """m, largest distance between the target position and the measured EE position."""

    workspace: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.05, 0.40), (-0.30, 0.30), (0.003, 0.30))
    """(x, y, z) ranges of the target position in the robot base frame (m)."""
# [end:integrated_cfg]
