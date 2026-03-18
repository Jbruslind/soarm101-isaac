"""Interactive VLA inference test for the SO-ARM101 in Isaac Sim.

Launches Isaac Sim with the SOARM 101 robot arm, wrist + overhead cameras,
OmniGraph ROS2 bridge for sensor/actuator topics, and an omni.ui control
panel for sending VLA prompts, spawning objects, and monitoring telemetry.
The full GUI is streamed via WebRTC (LIVESTREAM=2).

Usage (inside Isaac Sim container):
    /isaac-sim/python.sh /isaac_envs/interactive_inference.py

From host with Docker:
    ./scripts/interactive_test.sh
"""

from __future__ import annotations

import argparse
import os
import time as _time

# ---------------------------------------------------------------------------
# AppLauncher must be created before any other Omniverse / Isaac imports.
# ---------------------------------------------------------------------------
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Interactive VLA inference test")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()

app_launcher = AppLauncher(args)
simulation_app = app_launcher.app

# Set allowResize for WebRTC as early as possible so the stream can negotiate
# resolution when the client connects (must be before viewport/livestream init).
if os.environ.get("LIVESTREAM") in ("1", "2"):
    try:
        import carb
        carb.settings.get_settings().set_bool("/app/livestream/allowResize", True)
    except Exception:
        pass

# ---------------------------------------------------------------------------
# Imports available only after AppLauncher
# ---------------------------------------------------------------------------
import math
import torch
import numpy as np

import omni.usd
import omni.ui as ui
import omni.graph.core as og
import omni.kit.app
from pxr import Usd, UsdGeom, Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, Articulation
from isaaclab.sensors import CameraCfg, Camera
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationCfg, SimulationContext

# ---------------------------------------------------------------------------
# Constants (matching soarm_reach_env.py)
# ---------------------------------------------------------------------------
SOARM_JOINT_NAMES = [
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
]
NUM_JOINTS = 6
EE_BODY_NAME = "gripper_frame_link"
ROBOT_USD_PATH = "/robot_description/usd/soarm101.usd"
ROBOT_PRIM_PATH = "/World/Robot/so101_new_calib"

# Execution states
STATE_IDLE = "IDLE"
STATE_EXECUTING = "EXECUTING"
STATE_STOPPED = "STOPPED"

DEFAULT_TIMEOUT = 30  # seconds


def _euler_xyz_to_quat_wxyz(rx_deg: float, ry_deg: float, rz_deg: float) -> tuple[float, float, float, float]:
    """Convert Euler angles (X, Y, Z) in degrees to quaternion (w, x, y, z).

    Uses intrinsic XYZ order: rotate about X, then Y, then Z. Result is normalized.
    Isaac Lab OffsetCfg expects (w, x, y, z); world convention uses +X forward, +Z up.
    Implemented via quaternion multiplication (q = qz * qy * qx) to avoid sign/order bugs.
    """
    rx = math.radians(rx_deg)
    ry = math.radians(ry_deg)
    rz = math.radians(rz_deg)

    def qx(a: float) -> tuple[float, float, float, float]:
        ha = a / 2
        return (math.cos(ha), math.sin(ha), 0.0, 0.0)

    def qy(a: float) -> tuple[float, float, float, float]:
        ha = a / 2
        return (math.cos(ha), 0.0, math.sin(ha), 0.0)

    def qz(a: float) -> tuple[float, float, float, float]:
        ha = a / 2
        return (math.cos(ha), 0.0, 0.0, math.sin(ha))

    def mul(p: tuple[float, float, float, float], q: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
        pw, px, py, pz = p
        qw, qx_, qy_, qz_ = q
        return (
            pw * qw - px * qx_ - py * qy_ - pz * qz_,
            pw * qx_ + px * qw + py * qz_ - pz * qy_,
            pw * qy_ - px * qz_ + py * qw + pz * qx_,
            pw * qz_ + px * qy_ - py * qx_ + pz * qw,
        )

    q = mul(mul(qz(rz), qy(ry)), qx(rx))
    n = math.sqrt(q[0] * q[0] + q[1] * q[1] + q[2] * q[2] + q[3] * q[3])
    if n < 1e-10:
        return (1.0, 0.0, 0.0, 0.0)
    return (q[0] / n, q[1] / n, q[2] / n, q[3] / n)


# ═══════════════════════════════════════════════════════════════════════════
# Scene Setup
# ═══════════════════════════════════════════════════════════════════════════

def _setup_scene(sim: SimulationContext):
    """Spawn robot, cameras, ground, and lighting onto the USD stage.

    The robot USD (from URDF import) can contain two articulation roots
    (so101_new_calib and so101_new_calib_01). We remove the duplicate so
    only one SO-ARM101 is visible. The wrist camera is parented to
    gripper_frame_link and moves with the end-effector.
    """
    stage = omni.usd.get_context().get_stage()

    # -- Robot USD --
    robot_cfg = ArticulationCfg(
        prim_path=ROBOT_PRIM_PATH,
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            joint_pos={name: 0.0 for name in SOARM_JOINT_NAMES},
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=SOARM_JOINT_NAMES[:5],
                velocity_limit_sim=2.0,
                effort_limit_sim=30.0,
                stiffness=80.0,
                damping=4.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                velocity_limit_sim=2.0,
                effort_limit_sim=10.0,
                stiffness=80.0,
                damping=4.0,
            ),
        },
    )
    robot_cfg.spawn.func("/World/Robot", robot_cfg.spawn)

    # The URDF→USD importer often creates two articulation roots: so101_new_calib and
    # so101_new_calib_01. They come from a reference, so RemovePrim() does not stick.
    # Deactivate the duplicate so only one robot is visible and controlled.
    robot_prim = stage.GetPrimAtPath("/World/Robot")
    if robot_prim.IsValid():
        deactivated = 0
        for child in robot_prim.GetChildren():
            if child.GetName() != "so101_new_calib":
                child.SetActive(False)
                deactivated += 1
        if deactivated:
            print(f"[INFO] Deactivated {deactivated} duplicate robot prim(s) (kept so101_new_calib).", flush=True)

    saved_spawn = robot_cfg.spawn
    robot_cfg.spawn = None
    robot = Articulation(robot_cfg)
    robot_cfg.spawn = saved_spawn

    # -- Ground plane --
    ground_cfg = sim_utils.CuboidCfg(
        size=(100.0, 100.0, 0.02),
        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        collision_props=sim_utils.CollisionPropertiesCfg(),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.3, 0.3, 0.3)),
    )
    ground_cfg.func("/World/ground", ground_cfg, translation=(0.0, 0.0, -0.01))

    # -- Lighting --
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/light", light_cfg)

    # -- Container for user-spawned objects --
    stage.DefinePrim("/World/SpawnedObjects", "Xform")

    # -- Cameras --
    stage.DefinePrim("/World/Cameras", "Xform")

    # 1) Wrist (gripper) camera: under the robot so it moves with the end-effector.
    #    Position (-0.09, 0, -0.075) m, rotation (-180, -32, -180) deg (x,y,z) → quat (w,x,y,z).
    #    In the Stage panel it appears under World > Robot > so101_new_calib > gripper_frame_link > wrist_cam.
    wrist_cam = Camera(CameraCfg(
        prim_path=f"{ROBOT_PRIM_PATH}/{EE_BODY_NAME}/wrist_cam",
        update_period=0.0333,
        height=224, width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=1.93, horizontal_aperture=2.65),
        offset=CameraCfg.OffsetCfg(
            pos=(-0.09, 0.0, -0.075),
            rot=(-0.275637, 0.0, 0.961262, 0.0),  # Euler (-180, -32, -180) deg (x,y,z)
            convention="world",
        ),
    ))

    # 2) Overhead camera: at (0, 0, 0.8474) m, rotation Euler (rx, ry, rz) = (0, -20, 0) deg.
    # Quat computed explicitly via _euler_xyz_to_quat_wxyz to avoid hand-calculation errors.
    # World convention: +X forward, +Z up. If angle still looks wrong in Sim, try convention="opengl".
    _overhead_euler_deg = (0.0, -20.0, 0.0)
    _overhead_quat = _euler_xyz_to_quat_wxyz(*_overhead_euler_deg)
    _norm = math.sqrt(sum(x * x for x in _overhead_quat))
    print(
        f"[INFO] Overhead cam: euler_deg={_overhead_euler_deg} -> quat(w,x,y,z)={_overhead_quat} (norm={_norm:.6f})",
        flush=True,
    )
    overhead_cam = Camera(CameraCfg(
        prim_path="/World/Cameras/overhead_cam",
        update_period=0.0333,
        height=224, width=224,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=1.93, horizontal_aperture=2.65),
        offset=CameraCfg.OffsetCfg(
            pos=(0.0, 0.0, 0.8474),
            rot=_overhead_quat,
            convention="world",
        ),
    ))

    return robot, wrist_cam, overhead_cam


# ═══════════════════════════════════════════════════════════════════════════
# OmniGraph ROS2 Bridge
# ═══════════════════════════════════════════════════════════════════════════

def _setup_ros2_bridge():
    """Create OmniGraph action graph for ROS2 pub/sub.

    Publishes:
      /joint_states  (sensor_msgs/JointState)
      /camera/wrist/image_raw  (sensor_msgs/Image)
      /camera/overhead/image_raw  (sensor_msgs/Image)

    Subscribes:
      /joint_commands  (trajectory_msgs/JointTrajectory)

    Also creates helper graphs for /vla/prompt and /vla/enabled publishing
    which are driven by the omni.ui callbacks via global state.
    """
    # #region agent log f1832b
    import json as _json_dbg, time as _time_dbg, glob as _glob_dbg, os as _os_dbg
    _DEBUG_LOG = "/.cursor/debug-f1832b.log"

    def _dbg(msg, data, hyp, run_id="run1"):
        entry = {
            "sessionId": "f1832b",
            "id": f"log_{int(_time_dbg.time()*1000)}",
            "timestamp": int(_time_dbg.time()*1000),
            "location": "interactive_inference.py:_setup_ros2_bridge",
            "message": msg, "data": data, "runId": run_id, "hypothesisId": hyp,
        }
        try:
            with open(_DEBUG_LOG, "a") as _fh:
                _fh.write(_json_dbg.dumps(entry) + "\n")
        except Exception:
            pass

    # Hyp A/D: check if ROS2 is installed at all
    ros_humble_exists = _os_dbg.path.isdir("/opt/ros/humble")
    ros_env_vars = {k: v for k, v in _os_dbg.environ.items() if "ROS" in k or "RMW" in k or "AMENT" in k}
    _dbg("ROS2 environment check", {"ros_humble_exists": ros_humble_exists, "ros_env_vars": ros_env_vars}, "A_D")

    # Hyp B/C: check extension presence in extscache for both naming conventions
    old_ext_dirs = _glob_dbg.glob("/isaac-sim/extscache/omni.isaac.ros2_bridge*")
    new_ext_dirs = _glob_dbg.glob("/isaac-sim/extscache/isaacsim.ros2.bridge*")
    _dbg("Extension directory probe", {
        "omni_isaac_ros2_bridge_dirs": old_ext_dirs,
        "isaacsim_ros2_bridge_dirs": new_ext_dirs,
    }, "B_C")

    # Hyp B: try the new Isaac Sim 5.x extension name directly
    new_name_error = None
    try:
        import isaacsim.ros2.bridge  # noqa: F401
        _dbg("isaacsim.ros2.bridge import SUCCESS", {}, "B")
    except Exception as new_name_exc:
        new_name_error = str(new_name_exc)
        _dbg("isaacsim.ros2.bridge import FAILED", {"error": new_name_error}, "B")

    # Hyp C: check extensions manager for currently loaded/registered extensions
    try:
        import omni.kit.app as _kit_app
        ext_mgr = _kit_app.get_app().get_extension_manager()
        ros2_exts = [e for e in ext_mgr.get_extensions() if "ros2" in e.get("id", "").lower()]
        _dbg("Kit extension manager ros2 entries", {"ros2_extensions": ros2_exts}, "C")
    except Exception as ext_mgr_exc:
        _dbg("Kit extension manager query failed", {"error": str(ext_mgr_exc)}, "C")

    # Original import attempt (Hyp B: old name)
    try:
        import omni.isaac.ros2_bridge  # noqa: F401 – triggers extension load
    except (ImportError, ModuleNotFoundError) as old_name_exc:
        _dbg("omni.isaac.ros2_bridge import FAILED (old name)", {"error": str(old_name_exc)}, "B")
        print("[WARN] omni.isaac.ros2_bridge not available – ROS2 topics disabled.", flush=True)
        return
    # #endregion

    keys = og.Controller.Keys

    # -- Main action graph --
    (graph, nodes, _, _) = og.Controller.edit(
        {"graph_path": "/World/ROS2Bridge", "evaluator_name": "execution"},
        {
            keys.CREATE_NODES: [
                ("tick", "omni.graph.action.OnPlaybackTick"),
                ("sim_time", "omni.isaac.core_nodes.IsaacReadSimulationTime"),
                # Joint state publisher
                ("read_joint", "omni.isaac.core_nodes.IsaacArticulationController"),
                ("pub_joint", "omni.isaac.ros2_bridge.ROS2PublishJointState"),
                # Joint command subscriber
                ("sub_joint_cmd", "omni.isaac.ros2_bridge.ROS2SubscribeJointState"),
                ("art_ctrl", "omni.isaac.core_nodes.IsaacArticulationController"),
                # Camera publishers
                ("cam_wrist_helper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
                ("cam_overhead_helper", "omni.isaac.ros2_bridge.ROS2CameraHelper"),
            ],
            keys.SET_VALUES: [
                ("pub_joint.inputs:topicName", "/joint_states"),
                ("pub_joint.inputs:targetPrim", ROBOT_PRIM_PATH),
                ("sub_joint_cmd.inputs:topicName", "/joint_commands"),
                ("art_ctrl.inputs:targetPrim", ROBOT_PRIM_PATH),
                ("art_ctrl.inputs:robotPath", ROBOT_PRIM_PATH),
                ("cam_wrist_helper.inputs:topicName", "/camera/wrist/image_raw"),
                ("cam_wrist_helper.inputs:renderProductPath",
                 f"{ROBOT_PRIM_PATH}/{EE_BODY_NAME}/wrist_cam"),
                ("cam_wrist_helper.inputs:type", "rgb"),
                ("cam_overhead_helper.inputs:topicName", "/camera/overhead/image_raw"),
                ("cam_overhead_helper.inputs:renderProductPath",
                 "/World/Cameras/overhead_cam"),
                ("cam_overhead_helper.inputs:type", "rgb"),
            ],
            keys.CONNECT: [
                ("tick.outputs:tick", "sim_time.inputs:execIn"),
                ("tick.outputs:tick", "pub_joint.inputs:execIn"),
                ("tick.outputs:tick", "sub_joint_cmd.inputs:execIn"),
                ("tick.outputs:tick", "cam_wrist_helper.inputs:execIn"),
                ("tick.outputs:tick", "cam_overhead_helper.inputs:execIn"),
                ("sim_time.outputs:simulationTime", "pub_joint.inputs:timeStamp"),
                ("sub_joint_cmd.outputs:jointNames", "art_ctrl.inputs:jointNames"),
                ("sub_joint_cmd.outputs:positionCommand", "art_ctrl.inputs:positionCommand"),
                ("sub_joint_cmd.outputs:execOut", "art_ctrl.inputs:execIn"),
            ],
        },
    )
    print("[INFO] OmniGraph ROS2 bridge created.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Object Spawning Helpers
# ═══════════════════════════════════════════════════════════════════════════

_spawn_counter = 0


def spawn_object(obj_type: str, size: float, pos: tuple, color: tuple):
    """Spawn a physics-enabled primitive under /World/SpawnedObjects/."""
    global _spawn_counter

    _spawn_counter += 1
    prim_path = f"/World/SpawnedObjects/obj_{_spawn_counter:03d}"

    common_rigid = sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0)
    common_mass = sim_utils.MassPropertiesCfg(mass=0.05)
    common_collision = sim_utils.CollisionPropertiesCfg()
    common_visual = sim_utils.PreviewSurfaceCfg(diffuse_color=color)

    if obj_type == "Cube":
        cfg = sim_utils.CuboidCfg(
            size=(size, size, size),
            rigid_props=common_rigid,
            mass_props=common_mass,
            collision_props=common_collision,
            visual_material=common_visual,
        )
    elif obj_type == "Sphere":
        cfg = sim_utils.SphereCfg(
            radius=size / 2.0,
            rigid_props=common_rigid,
            mass_props=common_mass,
            collision_props=common_collision,
            visual_material=common_visual,
        )
    elif obj_type == "Cylinder":
        cfg = sim_utils.CylinderCfg(
            radius=size / 2.0,
            height=size,
            rigid_props=common_rigid,
            mass_props=common_mass,
            collision_props=common_collision,
            visual_material=common_visual,
        )
    else:
        print(f"[WARN] Unknown object type: {obj_type}", flush=True)
        return

    cfg.func(prim_path, cfg, translation=pos)
    print(f"[INFO] Spawned {obj_type} at {pos} → {prim_path}", flush=True)


def clear_spawned_objects():
    """Remove all prims under /World/SpawnedObjects/."""
    global _spawn_counter
    stage = omni.usd.get_context().get_stage()
    parent = stage.GetPrimAtPath("/World/SpawnedObjects")
    if parent.IsValid():
        for child in parent.GetChildren():
            stage.RemovePrim(child.GetPath())
    _spawn_counter = 0
    print("[INFO] Cleared all spawned objects.", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# ROS2 Prompt / Enable Publishing via OmniGraph Script Node
# ═══════════════════════════════════════════════════════════════════════════
# Rather than building a full OmniGraph for String/Bool publishing, we use a
# lightweight approach: write the prompt and enable state to files in /tmp
# that the sim loop periodically reads and publishes through a tiny helper.
# If omni.isaac.ros2_bridge exposes a Python-level publisher we use that;
# otherwise we fall back to writing a flag file that the VLA bridge reads.
#
# For the interactive test this is robust: all containers share host network
# and we can use the OmniGraph pipeline for sensor data while handling the
# low-rate prompt/enable signals through a simpler channel.

_ros2_pub_prompt = None
_ros2_pub_enabled = None


def _init_ros2_publishers():
    """Try to create lightweight ROS2 publishers for prompt and enabled topics.

    Uses rclpy if available inside the Isaac Sim Python environment.
    Falls back to file-based signaling if not.
    """
    global _ros2_pub_prompt, _ros2_pub_enabled

    # #region agent log f1832b
    import json as _json_dbg2, time as _time_dbg2, os as _os_dbg2, glob as _glob_dbg2, sys as _sys_dbg2
    _DEBUG_LOG2 = "/.cursor/debug-f1832b.log"

    def _dbg2(msg, data, hyp, run_id="run1"):
        entry = {
            "sessionId": "f1832b",
            "id": f"log_{int(_time_dbg2.time()*1000)}",
            "timestamp": int(_time_dbg2.time()*1000),
            "location": "interactive_inference.py:_init_ros2_publishers",
            "message": msg, "data": data, "runId": run_id, "hypothesisId": hyp,
        }
        try:
            with open(_DEBUG_LOG2, "a") as _fh2:
                _fh2.write(_json_dbg2.dumps(entry) + "\n")
        except Exception:
            pass

    # Hyp A: confirm whether /opt/ros/humble exists and rclpy is importable
    ros_humble_exists = _os_dbg2.path.isdir("/opt/ros/humble")
    rclpy_in_sys_path = any("rclpy" in p for p in _sys_dbg2.path)
    rclpy_search_paths = [p for p in _sys_dbg2.path if "ros" in p.lower() or "ament" in p.lower()]
    _dbg2("rclpy sys.path probe", {
        "ros_humble_exists": ros_humble_exists,
        "rclpy_in_sys_path": rclpy_in_sys_path,
        "ros_related_paths": rclpy_search_paths,
        "sys_path_count": len(_sys_dbg2.path),
    }, "A")

    # Hyp E: check if any ros2 library paths exist in LD_LIBRARY_PATH
    ld_lib_path = _os_dbg2.environ.get("LD_LIBRARY_PATH", "")
    ros_in_ld = [p for p in ld_lib_path.split(":") if "ros" in p.lower()]
    _dbg2("LD_LIBRARY_PATH ros2 entries", {"ros_in_ld_library_path": ros_in_ld}, "E")
    # #endregion

    try:
        import rclpy
        from std_msgs.msg import Bool as BoolMsg, String as StringMsg

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("isaac_sim_ui_publisher")
        _ros2_pub_prompt = node.create_publisher(StringMsg, "/vla/prompt", 10)
        _ros2_pub_enabled = node.create_publisher(BoolMsg, "/vla/enabled", 10)
        print("[INFO] ROS2 publishers for /vla/prompt and /vla/enabled created via rclpy.", flush=True)
        return node
    except Exception as e:
        # #region agent log f1832b
        _dbg2("rclpy import/init FAILED", {"error": str(e), "error_type": type(e).__name__}, "A")
        # #endregion
        print(f"[WARN] rclpy not available ({e}). Using file-based signaling for prompt/enabled.", flush=True)
        return None

_ros2_node = None


def publish_prompt(prompt: str):
    """Publish a prompt string to /vla/prompt."""
    global _ros2_pub_prompt
    if _ros2_pub_prompt is not None:
        from std_msgs.msg import String as StringMsg
        msg = StringMsg()
        msg.data = prompt
        _ros2_pub_prompt.publish(msg)
        print(f"[OpenVLA debug] Published /vla/prompt: {prompt!r}", flush=True)
    else:
        os.makedirs("/tmp/vla_signals", exist_ok=True)
        with open("/tmp/vla_signals/prompt.txt", "w") as f:
            f.write(prompt)
        print(f"[OpenVLA debug] Wrote prompt to /tmp/vla_signals/prompt.txt (no ROS2)", flush=True)


def publish_enabled(enabled: bool):
    """Publish an enabled flag to /vla/enabled."""
    global _ros2_pub_enabled
    if _ros2_pub_enabled is not None:
        from std_msgs.msg import Bool as BoolMsg
        msg = BoolMsg()
        msg.data = enabled
        _ros2_pub_enabled.publish(msg)
        print(f"[OpenVLA debug] Published /vla/enabled: {enabled} (Execute/Stop signal)", flush=True)
    else:
        os.makedirs("/tmp/vla_signals", exist_ok=True)
        with open("/tmp/vla_signals/enabled.txt", "w") as f:
            f.write("1" if enabled else "0")
        print(f"[OpenVLA debug] Wrote enabled={enabled} to /tmp/vla_signals/enabled.txt (no ROS2)", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# omni.ui Control Panel
# ═══════════════════════════════════════════════════════════════════════════

class InteractiveControlPanel:
    """omni.ui window providing Execute/Stop controls, telemetry, and object
    spawning for the interactive VLA inference test."""

    def __init__(self, robot: Articulation):
        self._robot = robot
        self._state = STATE_IDLE
        self._execute_start_time = 0.0
        self._timeout_sec = DEFAULT_TIMEOUT
        self._current_prompt = ""

        # Spawn defaults
        self._spawn_type_idx = 0
        self._spawn_size = 0.03
        self._spawn_pos = [0.15, 0.0, 0.02]
        self._spawn_color = [1.0, 0.2, 0.2]

        self._build_window()

    # -- State machine helpers --

    @property
    def state(self) -> str:
        return self._state

    def _on_execute(self):
        if not self._current_prompt.strip():
            self._status_label.text = "Status: Enter a command first"
            return

        publish_prompt(self._current_prompt)
        publish_enabled(True)

        self._state = STATE_EXECUTING
        self._execute_start_time = _time.time()
        self._update_button_states()
        self._status_label.text = f"Status: EXECUTING ({self._timeout_sec}s remaining)"

    def _on_stop(self):
        publish_enabled(False)
        self._state = STATE_STOPPED

        # Hold current joint positions
        if self._robot is not None and self._robot.is_initialized:
            try:
                current = self._robot.data.joint_pos
                self._robot.set_joint_position_target(current)
            except Exception:
                pass

        self._update_button_states()
        self._status_label.text = "Status: STOPPED"

    def _on_estop(self):
        publish_enabled(False)
        self._state = STATE_STOPPED

        if self._robot is not None and self._robot.is_initialized:
            try:
                current = self._robot.data.joint_pos
                self._robot.set_joint_position_target(current)
                zeros = torch.zeros_like(self._robot.data.joint_vel)
                self._robot.write_joint_state_to_sim(current, zeros)
            except Exception:
                pass

        self._update_button_states()
        self._status_label.text = "Status: E-STOPPED"

    def _on_reset_robot(self):
        publish_enabled(False)
        self._state = STATE_IDLE

        if self._robot is not None and self._robot.is_initialized:
            try:
                home = torch.zeros(1, NUM_JOINTS, device=self._robot.device)
                zeros = torch.zeros(1, NUM_JOINTS, device=self._robot.device)
                self._robot.write_joint_state_to_sim(home, zeros)
                self._robot.set_joint_position_target(home)
            except Exception:
                pass

        self._update_button_states()
        self._status_label.text = "Status: IDLE"

    def _on_reset_scene(self):
        clear_spawned_objects()
        self._on_reset_robot()

    def _update_button_states(self):
        executing = self._state == STATE_EXECUTING
        self._execute_btn.enabled = not executing
        self._stop_btn.enabled = True

    # -- Periodic update (called from sim loop) --

    def update(self, sim_time: float):
        """Called each sim step to refresh telemetry and check auto-stop."""

        # Auto-stop timeout
        if self._state == STATE_EXECUTING:
            elapsed = _time.time() - self._execute_start_time
            remaining = max(0, self._timeout_sec - elapsed)
            self._status_label.text = f"Status: EXECUTING ({int(remaining)}s remaining)"
            if remaining <= 0:
                self._on_stop()
                self._status_label.text = "Status: STOPPED (auto-timeout)"

        # Telemetry
        if self._robot is not None and self._robot.is_initialized:
            try:
                jpos = self._robot.data.joint_pos[0].cpu().numpy()
                for i, name in enumerate(SOARM_JOINT_NAMES):
                    if i < len(self._joint_labels):
                        self._joint_labels[i].text = f"  {name}: {jpos[i]:+.3f} rad"

                ee_idx = self._robot.find_bodies(EE_BODY_NAME)[0][0]
                ee_pos = self._robot.data.body_pos_w[0, ee_idx].cpu().numpy()
                self._ee_label.text = f"  EE: ({ee_pos[0]:.3f}, {ee_pos[1]:.3f}, {ee_pos[2]:.3f}) m"
            except Exception:
                pass

    # -- Build the window --

    def _build_window(self):
        self._window = ui.Window("VLA Interactive Test", width=380, height=700)

        with self._window.frame:
            with ui.ScrollingFrame():
                with ui.VStack(spacing=6):
                    ui.Spacer(height=4)

                    # ── Status ──
                    self._status_label = ui.Label("Status: IDLE", style={"font_size": 16})

                    ui.Spacer(height=4)
                    ui.Line(height=1)

                    # ── Command Section ──
                    ui.Label("Command:", style={"font_size": 14})
                    self._prompt_field = ui.StringField(height=28)
                    self._prompt_field.model.set_value("pick up the red cube")
                    self._current_prompt = "pick up the red cube"

                    def _on_prompt_changed(model):
                        self._current_prompt = model.get_value_as_string()

                    self._prompt_field.model.add_value_changed_fn(_on_prompt_changed)

                    with ui.HStack(spacing=8, height=36):
                        self._execute_btn = ui.Button(
                            "Execute",
                            clicked_fn=self._on_execute,
                            style={"background_color": 0xFF2D8B2D, "font_size": 15},
                        )
                        self._stop_btn = ui.Button(
                            "STOP",
                            clicked_fn=self._on_stop,
                            style={"background_color": 0xFF1A1ACD, "font_size": 15},
                        )

                    with ui.HStack(height=36):
                        self._estop_btn = ui.Button(
                            "E-STOP",
                            clicked_fn=self._on_estop,
                            style={"background_color": 0xFF0000DD, "font_size": 15},
                        )

                    with ui.HStack(spacing=4, height=24):
                        ui.Label("Auto-stop (sec):", width=110)
                        timeout_field = ui.IntField(width=60, height=22)
                        timeout_field.model.set_value(DEFAULT_TIMEOUT)

                        def _on_timeout(model):
                            val = model.get_value_as_int()
                            self._timeout_sec = max(5, min(300, val))

                        timeout_field.model.add_value_changed_fn(_on_timeout)

                    ui.Spacer(height=4)
                    ui.Line(height=1)

                    # ── Telemetry Section ──
                    with ui.CollapsableFrame("Telemetry", collapsed=False):
                        with ui.VStack(spacing=2):
                            self._joint_labels = []
                            for name in SOARM_JOINT_NAMES:
                                lbl = ui.Label(f"  {name}: 0.000 rad", style={"font_size": 12})
                                self._joint_labels.append(lbl)
                            self._ee_label = ui.Label(
                                "  EE: (0.000, 0.000, 0.000) m", style={"font_size": 12}
                            )

                    ui.Spacer(height=2)

                    # ── Spawn Objects Section ──
                    with ui.CollapsableFrame("Spawn Objects", collapsed=True):
                        with ui.VStack(spacing=4):
                            obj_types = ["Cube", "Sphere", "Cylinder"]

                            with ui.HStack(spacing=4, height=24):
                                ui.Label("Type:", width=50)
                                combo = ui.ComboBox(0, *obj_types, width=100)

                                def _type_changed(model, item):
                                    idx = model.get_item_value_model().get_value_as_int()
                                    self._spawn_type_idx = idx

                                combo.model.add_item_changed_fn(_type_changed)

                            with ui.HStack(spacing=4, height=24):
                                ui.Label("Size (m):", width=70)
                                size_f = ui.FloatField(width=70)
                                size_f.model.set_value(0.03)

                                def _size_changed(model):
                                    self._spawn_size = model.get_value_as_float()

                                size_f.model.add_value_changed_fn(_size_changed)

                            ui.Label("Position:", style={"font_size": 12})
                            with ui.HStack(spacing=4, height=24):
                                for axis_i, (axis_name, default_val) in enumerate(
                                    [("X", 0.15), ("Y", 0.0), ("Z", 0.02)]
                                ):
                                    ui.Label(axis_name, width=14)
                                    fld = ui.FloatField(width=60)
                                    fld.model.set_value(default_val)

                                    def _pos_cb(model, idx=axis_i):
                                        self._spawn_pos[idx] = model.get_value_as_float()

                                    fld.model.add_value_changed_fn(_pos_cb)

                            ui.Label("Color (RGB 0-1):", style={"font_size": 12})
                            with ui.HStack(spacing=4, height=24):
                                for c_i, (c_name, c_def) in enumerate(
                                    [("R", 1.0), ("G", 0.2), ("B", 0.2)]
                                ):
                                    ui.Label(c_name, width=14)
                                    cf = ui.FloatField(width=50)
                                    cf.model.set_value(c_def)

                                    def _col_cb(model, idx=c_i):
                                        self._spawn_color[idx] = max(
                                            0.0, min(1.0, model.get_value_as_float())
                                        )

                                    cf.model.add_value_changed_fn(_col_cb)

                            with ui.HStack(spacing=8, height=30):
                                ui.Button(
                                    "Spawn",
                                    clicked_fn=lambda: spawn_object(
                                        ["Cube", "Sphere", "Cylinder"][self._spawn_type_idx],
                                        self._spawn_size,
                                        tuple(self._spawn_pos),
                                        tuple(self._spawn_color),
                                    ),
                                )
                                ui.Button("Clear All", clicked_fn=clear_spawned_objects)

                    ui.Spacer(height=2)

                    # ── Scene Section ──
                    with ui.CollapsableFrame("Scene", collapsed=True):
                        with ui.VStack(spacing=4):
                            with ui.HStack(spacing=8, height=30):
                                ui.Button("Reset Robot", clicked_fn=self._on_reset_robot)
                                ui.Button("Reset Scene", clicked_fn=self._on_reset_scene)

                            presets = ["Empty", "Table with Cube", "Cluttered Table"]
                            with ui.HStack(spacing=4, height=24):
                                ui.Label("Preset:", width=50)
                                preset_combo = ui.ComboBox(0, *presets, width=140)

                                def _preset_changed(model, item):
                                    idx = model.get_item_value_model().get_value_as_int()
                                    self._apply_preset(presets[idx])

                                preset_combo.model.add_item_changed_fn(_preset_changed)

                    ui.Spacer(height=8)

    def _apply_preset(self, name: str):
        clear_spawned_objects()
        if name == "Table with Cube":
            spawn_object("Cube", 0.03, (0.15, 0.0, 0.015), (1.0, 0.2, 0.2))
        elif name == "Cluttered Table":
            spawn_object("Cube", 0.03, (0.12, -0.05, 0.015), (1.0, 0.2, 0.2))
            spawn_object("Cube", 0.025, (0.18, 0.06, 0.0125), (0.2, 0.2, 1.0))
            spawn_object("Sphere", 0.03, (0.10, 0.08, 0.015), (0.2, 1.0, 0.2))
            spawn_object("Cylinder", 0.025, (0.20, -0.04, 0.0125), (1.0, 1.0, 0.2))
        # "Empty" needs no objects


# ═══════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════

def main():
    global _ros2_node

    livestream = os.environ.get("LIVESTREAM") in ("1", "2")

    # -- Simulation context --
    sim_cfg = SimulationCfg(dt=1 / 120.0, render_interval=2)
    sim = SimulationContext(sim_cfg)

    # -- Build the scene --
    print("[INFO] Setting up scene (robot, cameras, lighting)...", flush=True)
    robot, wrist_cam, overhead_cam = _setup_scene(sim)

    # -- ROS2 OmniGraph bridge --
    print("[INFO] Creating OmniGraph ROS2 bridge...", flush=True)
    _setup_ros2_bridge()

    # -- ROS2 publishers for prompt/enabled --
    _ros2_node = _init_ros2_publishers()

    # -- Reset simulation --
    sim.reset()
    robot.reset()

    # -- Set viewport camera so WebRTC stream shows the robot (eye position, look-at position in meters) --
    sim.set_camera_view([0.8, 0.8, 0.5], [0.0, 0.0, 0.2])

    # -- Build the control panel --
    print("[INFO] Building omni.ui control panel...", flush=True)
    panel = InteractiveControlPanel(robot)

    print("[INFO] Interactive inference environment ready.", flush=True)
    if livestream:
        public_ip = os.environ.get("PUBLIC_IP", "127.0.0.1")
        print(
            f"\nWebRTC streaming is active.\n"
            f"  1. Open the Isaac Sim WebRTC Streaming Client.\n"
            f"  2. Enter server address: {public_ip}\n"
            f"  3. Click Connect.\n"
            f"  4. Use the 'VLA Interactive Test' panel on the right to send commands.\n"
            f"  5. To view through a camera: click the video/camera icon at the top of the\n"
            f"     viewport and select 'overhead_cam' or 'wrist_cam' (under Robot > gripper_frame_link).\n",
            flush=True,
        )

    # -- Main simulation loop --
    step_count = 0
    while simulation_app.is_running():
        sim.step()
        step_count += 1

        # Update robot data
        robot.update(sim.cfg.dt)

        # Update telemetry + auto-stop every 6 steps (~20 Hz at 120 Hz sim)
        if step_count % 6 == 0:
            panel.update(sim.get_physics_dt() * step_count)

        # Spin ROS2 node if we have one (non-blocking)
        if _ros2_node is not None:
            try:
                import rclpy
                rclpy.spin_once(_ros2_node, timeout_sec=0)
            except Exception:
                pass

    # -- Cleanup --
    if _ros2_node is not None:
        try:
            _ros2_node.destroy_node()
            import rclpy
            rclpy.shutdown()
        except Exception:
            pass

    sim.stop()
    simulation_app.close()


if __name__ == "__main__":
    main()
