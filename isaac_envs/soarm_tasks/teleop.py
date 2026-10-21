"""Keyboard and SpaceMouse teleop for ``Soarm-PickPlace-Fastener-IK-Rel-v0`` (record_demos.py).

Isaac Lab's ``Se3Keyboard`` and ``Se3SpaceMouse`` both emit ``[dx, dy, dz, rx, ry, rz, gripper]``
per control step: a delta EE pose (m, rad) and +1 open / -1 close. ``record_demos.py`` passes that
tensor straight to ``env.step``, so the teleop task takes exactly this action
(:class:`soarm_tasks.mdp.actions_cfg.IntegratedPoseIKActionCfg` turns it into an absolute EE target
for the IK-Abs controller). Same pattern as Isaac Lab's Franka ``Stack-Cube-...-IK-Rel`` tasks.

Keyboard (``--teleop_device keyboard``); hold a key to move, axes in the robot base frame:

=======================================  =======  =======
action                                   key +    key -
=======================================  =======  =======
x: away from / towards the robot base    W        S
y: to the robot's left / right           A        D
z: up / down                             Q        E
yaw of the gripper (about vertical)      C        V
toggle gripper (starts open)             K
clear held motion (stuck key)            L
reset the episode (record_demos.py)      R
=======================================  =======  =======

Roll/pitch keys (Z/X, T/G) are ignored: the task keeps the gripper pointing down (``top_down``).

SpaceMouse (``--teleop_device spacemouse``): push/tilt the cap to move/turn (same axes, analog),
left button toggles the gripper, right button resets the episode (``record_demos.py`` binds its
reset to the right button, like ``R`` on the keyboard). The SpaceMouse must be plugged
into the machine running the container and passed through (``/dev/hidraw*``); Moonlight does not
forward it.
"""

from __future__ import annotations

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.keyboard import Se3KeyboardCfg
from isaaclab.devices.spacemouse import Se3SpaceMouseCfg

# [start:sensitivity]
# Per control step (30 Hz): a held key moves the EE target 3 mm (9 cm/s) and turns it 0.03 rad
# (52 deg/s). The SpaceMouse gives up to 4 mm / 0.04 rad at full deflection.
KEYBOARD_POS_STEP = 0.003
KEYBOARD_ROT_STEP = 0.03
SPACEMOUSE_POS_STEP = 0.004
SPACEMOUSE_ROT_STEP = 0.04
# [end:sensitivity]


# [start:devices]
def make_teleop_devices(sim_device: str) -> DevicesCfg:
    """``env_cfg.teleop_devices`` of the teleop task: ``keyboard`` and ``spacemouse``."""
    return DevicesCfg(devices={
        "keyboard": Se3KeyboardCfg(pos_sensitivity=KEYBOARD_POS_STEP, rot_sensitivity=KEYBOARD_ROT_STEP,
                                   sim_device=sim_device),
        "spacemouse": Se3SpaceMouseCfg(pos_sensitivity=SPACEMOUSE_POS_STEP, rot_sensitivity=SPACEMOUSE_ROT_STEP,
                                       sim_device=sim_device),
    })
# [end:devices]
