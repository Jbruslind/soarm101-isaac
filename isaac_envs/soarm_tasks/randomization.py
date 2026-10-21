"""Domain-randomization ranges of the fastener tasks, in one place (plain Python).

Part 1 keeps them modest: the scripted expert and the first fine-tuned policies should see a
scene close to the nominal one. Part 2 (F9) widens them. Each range maps to one event term in
``pick_place_env_cfg.FastenerEventCfg``; a range of ``None`` disables that term.

Planned but not implemented yet (they get their own event terms, ranges below are the targets):
camera-mount jitter (+-3 mm / +-3 deg on both cameras) and actuator / encoder-bias
randomization (joint offset, PD gain scale).
"""

from __future__ import annotations

# [start:ranges]
# Fastener pose at reset: position and yaw from work_cell.FASTENER_SPAWN_*; it lies on its side
# with a random spin about its own axis (the hex head settles on a flat).
FASTENER_SPIN = (-3.1416, 3.1416)          # rad, about the bolt axis
# Robot joints at reset: uniform offset around the home pose (all joints 0).
ROBOT_JOINT_OFFSET = (-0.05, 0.05)         # rad
# Fastener friction (steel on the jaws and the table). Nominal 0.5 / 0.4 (assets/fasteners.py).
FASTENER_STATIC_FRICTION = (0.45, 0.60)
FASTENER_DYNAMIC_FRICTION = (0.35, 0.50)
FRICTION_BUCKETS = 16
# Fastener mass scale (nominal 6.1 g for M6x20).
FASTENER_MASS_SCALE = (0.9, 1.1)
# Dome light, sampled at every reset. Nominal 2000 / (0.75, 0.75, 0.75) as in soarm_common.scene.
LIGHT_INTENSITY = (1500.0, 2500.0)
LIGHT_COLOR_VARIATION = 0.05               # per channel, brightness kept
# Later issue (not wired yet):
CAMERA_MOUNT_JITTER = None                 # target: (0.003 m, 3 deg)
ACTUATOR_BIAS = None                       # target: encoder offset +-0.02 rad, gain scale 0.9-1.1
# [end:ranges]
