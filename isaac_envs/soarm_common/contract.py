"""SO-ARM101 robot / camera / control contract (single source of truth).

Plain Python, stdlib only: importable by the Isaac Sim envs, the data collector,
``interactive_inference.py``, the ROS 2 bridge (ROS Humble container), the OpenPi training
config and the mock policy server. Isaac Lab config builders live next to it in ``robot.py``,
``cameras.py`` and ``scene.py``.

Print it as JSON with ``python3 isaac_envs/soarm_common/contract.py``.

Conventions
-----------
* Lengths in metres, angles in radians, quaternions ``(x, y, z, w)`` (Isaac Lab 3.0 order).
* Camera rotations use Isaac Lab's ``"world"`` convention: the camera looks along its +x
  axis with +z up.

Data flow (one name per camera, end to end)
--------------------------------------------
=====================  ===========================  ==============================
stage                  wrist camera                 overhead camera
=====================  ===========================  ==============================
USD prim               ``<robot>/<EE link>/wrist_cam``  ``<env>/Cameras/overhead_cam``
env observation        ``obs["wrist_rgb"]``         ``obs["overhead_rgb"]``
LeRobot dataset        ``observation.images.wrist`` ``observation.images.overhead``
ROS 2 topic            ``/camera/wrist/image_raw``  ``/camera/overhead/image_raw``
policy server request  ``images["cam_wrist"]``      ``images["cam_overhead"]``
=====================  ===========================  ==============================
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass

CONTRACT_VERSION = 1

# ═══════════════════════════════════════════════════════════════════════════
# Robot
# ═══════════════════════════════════════════════════════════════════════════

ROBOT_TYPE = "so_arm101"

# Joint order of the state and action vectors everywhere (sim, dataset, ROS, policy).
JOINT_NAMES: tuple[str, ...] = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)
NUM_JOINTS = len(JOINT_NAMES)  # 6
NUM_ARM_JOINTS = 5
ARM_JOINT_NAMES: tuple[str, ...] = JOINT_NAMES[:NUM_ARM_JOINTS]
GRIPPER_JOINT_NAME = "gripper"

# Joint position limits (rad), as authored in robot_description/urdf/soarm101.urdf and the USD
# made from it. robot.check_joint_limits() compares them with the spawned articulation.
JOINT_LIMITS: dict[str, tuple[float, float]] = {
    "shoulder_pan": (-1.91986, 1.91986),
    "shoulder_lift": (-1.74533, 1.74533),
    "elbow_flex": (-1.69, 1.69),
    "wrist_flex": (-1.65806, 1.65806),
    "wrist_roll": (-2.74385, 2.84121),
    "gripper": (-0.174533, 1.74533),  # ~0 = jaws closed, 1.745 = fully open
}

# End effector: a fixed frame between the jaw tips. +z points out along the fingers, the fixed
# jaw is on its +x side and the moving jaw on its -x side.
EE_BODY_NAME = "gripper_frame_link"
# Robot USD made by URDF importer 3.0 (docker/isaac-sim/convert_urdf.py). The default prim is the
# robot and articulation root; links are nested along the kinematic chain under Geometry/.
ROBOT_USD_PATH = "/robot_description/usd/soarm101/soarm101.usda"  # path inside the Isaac container
EE_LINK_REL_PATH = (
    "Geometry/base_link/shoulder_link/upper_arm_link/lower_arm_link/wrist_link/gripper_link/gripper_frame_link"
)
# Robot prim in the Isaac Lab envs (one per env) and in interactive_inference.py.
ENV_ROBOT_PRIM_PATH = "/World/envs/env_.*/Robot"
ENV0_PATH = "/World/envs/env_0"
INTERACTIVE_ROBOT_PRIM_PATH = "/World/Robot"
INTERACTIVE_CAMERA_ROOT = "/World/Cameras"

# Real hardware: every joint is a Feetech STS3215 servo. 3.35 N m is the stall torque used by the
# SysID physics variant of Isaac Lab's SO-101 asset (drive maxForce = 3.35 on all joints).
REAL_SERVO_MODEL = "Feetech STS3215"
REAL_SERVO_MAX_TORQUE_NM = 3.35
REAL_GRIPPER_EFFORT_LIMIT_NM = REAL_SERVO_MAX_TORQUE_NM

# Simulated joint drives (implicit PD), tuned for the sim, not identified on hardware.
# The gripper effort is capped at the real servo's torque: with 3.35 N m (was 10) the scripted
# pick still placed 20/20 (GitHub issue #4). The arm drives keep 30 N m; capping them at the real
# torque is untested and would likely need re-tuned gains (Isaac Lab's SysID asset pairs the
# 3.35 N m limit with its own identified stiffness/damping).
SIM_ARM_DRIVE = {"stiffness": 80.0, "damping": 4.0, "effort_limit": 30.0, "velocity_limit": 2.0}
SIM_GRIPPER_DRIVE = {
    "stiffness": 80.0, "damping": 4.0, "effort_limit": REAL_GRIPPER_EFFORT_LIMIT_NM, "velocity_limit": 2.0,
}

# ═══════════════════════════════════════════════════════════════════════════
# Timing
# ═══════════════════════════════════════════════════════════════════════════

PHYSICS_DT = 1.0 / 120.0
DECIMATION = 4                       # physics steps per control step
CONTROL_HZ = 30.0                    # = 1 / (PHYSICS_DT * DECIMATION)
CONTROL_DT = 1.0 / CONTROL_HZ
RENDER_INTERVAL = 2                  # physics steps per render (60 Hz); cameras sample at CONTROL_HZ
DATASET_FPS = int(CONTROL_HZ)        # one dataset frame per control step
assert abs(PHYSICS_DT * DECIMATION - CONTROL_DT) < 1e-12

# ═══════════════════════════════════════════════════════════════════════════
# Actions
# ═══════════════════════════════════════════════════════════════════════════

# Policy actions, dataset ``action`` and ROS /joint_commands are all the same thing: the absolute
# joint-position target (rad, JOINT_NAMES order, gripper included) sent to the joint PD drives
# for the next control step (1 / CONTROL_HZ s), held until the next action.
#
# The Isaac Lab env (SoarmReachEnv / SoarmPickEnv, also used for RL) instead takes a normalised
# delta a in [-1, 1]^6 and applies  target = q + clip(a, -1, 1) * ENV_DELTA_ACTION_SCALE.
# The collector records that applied target (``env.joint_pos_target``), so recorded actions are
# absolute targets that the robot can track at 30 Hz, never the raw delta.
ACTION_TYPE = "absolute_joint_position"
ACTION_DIM = NUM_JOINTS
ENV_DELTA_ACTION_SCALE = 0.05        # rad per control step for a = 1

# ═══════════════════════════════════════════════════════════════════════════
# Cameras
# ═══════════════════════════════════════════════════════════════════════════

IMAGE_HEIGHT = 224
IMAGE_WIDTH = 224
IMAGE_SIZE = IMAGE_HEIGHT  # square images
# Pinhole intrinsics shared by both cameras: f = 1.93 mm, horizontal aperture 2.65 mm
# (horizontal FOV 2 * atan(2.65 / 2 / 1.93) = 69 deg). Near clip 1 cm.
FOCAL_LENGTH = 1.93
HORIZONTAL_APERTURE = 2.65
CLIPPING_RANGE = (0.01, 1.0e6)


def _normalize(v):
    n = math.sqrt(sum(c * c for c in v))
    return tuple(c / n for c in v)


def _cross(a, b):
    return (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0])


def quat_from_matrix(m) -> tuple[float, float, float, float]:
    """(x, y, z, w) quaternion of a 3x3 rotation matrix given as rows."""
    tr = m[0][0] + m[1][1] + m[2][2]
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        w, x, y, z = 0.25 * s, (m[2][1] - m[1][2]) / s, (m[0][2] - m[2][0]) / s, (m[1][0] - m[0][1]) / s
    elif m[0][0] > m[1][1] and m[0][0] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[0][0] - m[1][1] - m[2][2])
        w, x, y, z = (m[2][1] - m[1][2]) / s, 0.25 * s, (m[0][1] + m[1][0]) / s, (m[0][2] + m[2][0]) / s
    elif m[1][1] > m[2][2]:
        s = 2.0 * math.sqrt(1.0 + m[1][1] - m[0][0] - m[2][2])
        w, x, y, z = (m[0][2] - m[2][0]) / s, (m[0][1] + m[1][0]) / s, 0.25 * s, (m[1][2] + m[2][1]) / s
    else:
        s = 2.0 * math.sqrt(1.0 + m[2][2] - m[0][0] - m[1][1])
        w, x, y, z = (m[1][0] - m[0][1]) / s, (m[0][2] + m[2][0]) / s, (m[1][2] + m[2][1]) / s, 0.25 * s
    if w < 0:
        x, y, z, w = -x, -y, -z, -w
    return (x, y, z, w)


def look_at_quat(eye, target, up=(0.0, 0.0, 1.0)) -> tuple[float, float, float, float]:
    """(x, y, z, w) rotation, in the "world" camera convention, of a camera at ``eye`` looking at
    ``target`` with image-up as close to ``up`` as possible (all in the parent frame)."""
    f = _normalize(tuple(t - e for t, e in zip(target, eye)))
    left = _normalize(_cross(up, f))
    u = _cross(f, left)
    # columns: camera x = forward, y = left, z = up
    return quat_from_matrix([[f[i], left[i], u[i]] for i in range(3)])


@dataclass(frozen=True)
class CameraSpec:
    """One camera, from USD prim to policy input.

    ``parent`` is ``"ee"`` (rigidly attached to EE_BODY_NAME; pose in that link's frame) or
    ``"world"`` (fixed; pose in the env frame, which is the world frame for env_0 and for
    interactive_inference.py).
    """

    name: str
    parent: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]  # (x, y, z, w), "world" convention
    prim_name: str
    obs_key: str        # Isaac Lab env observation dict
    dataset_key: str    # LeRobot feature / video name
    policy_key: str     # images[...] key in policy-server requests
    ros_topic: str      # sensor_msgs/Image topic published by interactive_inference.py
    height: int = IMAGE_HEIGHT
    width: int = IMAGE_WIDTH
    focal_length: float = FOCAL_LENGTH
    horizontal_aperture: float = HORIZONTAL_APERTURE
    clipping_range: tuple[float, float] = CLIPPING_RANGE


# Wrist camera, beside the gripper on its +y side, 7 cm back from the jaw tips, looking at a
# point just past the tips between the jaws, image up = gripper +x (fixed-jaw side). It sees both
# jaws and the grasped object. The earlier mount sat 5 cm out on the +x side (towards the base in
# the top-down grasp); when the gripper worked close to the base the forearm folded down over it
# and the frames went black (camera inside the arm mesh, GitHub issue #4). On the side, the
# camera is out of the arm's plane: no dark frames over 10 scripted picks incl. near-base ones.
_WRIST_POS = (0.020, 0.040, -0.070)
_WRIST_AIM = (-0.010, 0.0, 0.010)
WRIST_CAMERA = CameraSpec(
    name="wrist",
    parent="ee",
    pos=_WRIST_POS,
    rot=look_at_quat(_WRIST_POS, _WRIST_AIM, up=(1.0, 0.0, 0.0)),
    prim_name="wrist_cam",
    obs_key="wrist_rgb",
    dataset_key="observation.images.wrist",
    policy_key="cam_wrist",
    ros_topic="/camera/wrist/image_raw",
)

# Overhead camera: elevated three-quarter view of the table area in front of the robot, close
# enough that a 3 cm object is visible at 224 px. (Named "overhead" for the ROS topic and the
# cam_overhead policy key, which the bridge and dashboards already use.)
_OVERHEAD_EYE = (0.45, 0.40, 0.30)
_OVERHEAD_TARGET = (0.19, 0.0, 0.03)
OVERHEAD_CAMERA = CameraSpec(
    name="overhead",
    parent="world",
    pos=_OVERHEAD_EYE,
    rot=look_at_quat(_OVERHEAD_EYE, _OVERHEAD_TARGET),
    prim_name="overhead_cam",
    obs_key="overhead_rgb",
    dataset_key="observation.images.overhead",
    policy_key="cam_overhead",
    ros_topic="/camera/overhead/image_raw",
)

CAMERAS: tuple[CameraSpec, ...] = (WRIST_CAMERA, OVERHEAD_CAMERA)

# ═══════════════════════════════════════════════════════════════════════════
# Dataset (LeRobot v3.0) and policy-server keys
# ═══════════════════════════════════════════════════════════════════════════

DATASET_STATE_KEY = "observation.state"      # float32[6], joint positions (rad)
DATASET_ACTION_KEY = "action"                 # float32[6], absolute joint targets (rad)
DATASET_TASK_KEY = "language_instruction"

POLICY_STATE_KEY = "state"
POLICY_IMAGES_KEY = "images"
POLICY_PROMPT_KEY = "prompt"
POLICY_ACTIONS_KEY = "actions"                # (horizon, 6) absolute joint targets

# ═══════════════════════════════════════════════════════════════════════════
# ROS 2 topics
# ═══════════════════════════════════════════════════════════════════════════

JOINT_STATES_TOPIC = "/joint_states"
JOINT_COMMANDS_TOPIC = "/joint_commands"      # sensor_msgs/JointState, position = absolute targets
PROMPT_TOPIC = "/vla/prompt"
ENABLED_TOPIC = "/vla/enabled"


def as_dict() -> dict:
    """The contract as a JSON-serialisable dict."""
    return {
        "contract_version": CONTRACT_VERSION,
        "robot_type": ROBOT_TYPE,
        "joint_names": list(JOINT_NAMES),
        "joint_limits": {k: list(v) for k, v in JOINT_LIMITS.items()},
        "ee_body_name": EE_BODY_NAME,
        "robot_usd_path": ROBOT_USD_PATH,
        "sim_arm_drive": SIM_ARM_DRIVE,
        "sim_gripper_drive": SIM_GRIPPER_DRIVE,
        "real_servo": {"model": REAL_SERVO_MODEL, "max_torque_nm": REAL_SERVO_MAX_TORQUE_NM},
        "physics_dt": PHYSICS_DT,
        "decimation": DECIMATION,
        "control_hz": CONTROL_HZ,
        "dataset_fps": DATASET_FPS,
        "action_type": ACTION_TYPE,
        "action_dim": ACTION_DIM,
        "env_delta_action_scale": ENV_DELTA_ACTION_SCALE,
        "cameras": [asdict(c) for c in CAMERAS],
        "dataset": {"state": DATASET_STATE_KEY, "action": DATASET_ACTION_KEY, "task": DATASET_TASK_KEY},
        "policy": {"state": POLICY_STATE_KEY, "images": POLICY_IMAGES_KEY, "prompt": POLICY_PROMPT_KEY,
                   "actions": POLICY_ACTIONS_KEY},
        "ros_topics": {"joint_states": JOINT_STATES_TOPIC, "joint_commands": JOINT_COMMANDS_TOPIC,
                       "prompt": PROMPT_TOPIC, "enabled": ENABLED_TOPIC,
                       **{c.name: c.ros_topic for c in CAMERAS}},
    }


if __name__ == "__main__":
    print(json.dumps(as_dict(), indent=2))
