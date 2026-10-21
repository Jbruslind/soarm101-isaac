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
parser.add_argument(
    "--snapshot", metavar="DIR", default=None,
    help="Test mode: hold --snapshot-joint-pos, save one wrist and one overhead camera image "
         "(<DIR>/<camera>.png and .npy) and exit, without ROS or UI. Used to check that inference "
         "renders the same images as data collection.",
)
parser.add_argument(
    "--snapshot-joint-pos", default="0,0,0,0,0,0",
    help="Joint positions (rad, contract joint order) for --snapshot: 6 comma-separated values "
         "per pose, poses separated by ';'.",
)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
# The wrist/overhead cameras always render (ROS image topics). Isaac Lab 3.0 picks the Kit
# experience from this flag only (the ENABLE_CAMERAS env var is no longer read); without it a
# headless run gets isaaclab.python.headless.kit, whose renderer reads USD instead of Fabric, and
# the published images showed the robot frozen at its USD pose while the joints moved.
args.enable_cameras = True
# Isaac Lab 3.0 only opens a window for a Kit visualizer (livestream adds one itself).
if os.environ.get("HEADLESS") == "0" and not args.visualizer:
    args.visualizer = "kit"

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
import omni.timeline
from pxr import Usd, UsdGeom, Gf, Sdf

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.sensors import Camera
from isaaclab.sim import SimulationCfg, SimulationContext
from isaaclab_physx.sim.schemas import PhysxCollisionPropertiesCfg, PhysxRigidBodyPropertiesCfg

# ---------------------------------------------------------------------------
# Robot, cameras, timing and topics: the shared contract (same as data collection)
# ---------------------------------------------------------------------------
from soarm_common import contract as C
from soarm_common.cameras import camera_prim_path, make_camera_cfg
from soarm_common.robot import check_joint_limits, make_physx_cfg, make_robot_cfg
from soarm_common.scene import spawn_ground_and_light

SOARM_JOINT_NAMES = list(C.JOINT_NAMES)
NUM_JOINTS = C.NUM_JOINTS
ROBOT_PRIM_PATH = C.INTERACTIVE_ROBOT_PRIM_PATH
# PhysX/ROS2 nodes often need the *actual* articulation-root prim (can be a descendant of ROBOT_PRIM_PATH).
ROBOT_ARTICULATION_PRIM_PATH = ROBOT_PRIM_PATH
# Camera/link attachments should stay on the robot model hierarchy (not articulation-root fallback).
ROBOT_MODEL_PRIM_PATH = ROBOT_PRIM_PATH
# Fixed cameras go under /World/Cameras (the env uses /World/envs/env_0/Cameras; both at the origin).
CAMERA_ENV_PATH = C.INTERACTIVE_CAMERA_ROOT.rsplit("/", 1)[0]

# Execution states
STATE_IDLE = "IDLE"
STATE_EXECUTING = "EXECUTING"
STATE_STOPPED = "STOPPED"

# Auto-stop after Execute: must exceed worst-case first OpenPi RTT (often 60–120s+ on remote/Jetson).
# Override: VLA_AUTO_STOP_SEC=300 ./scripts/interactive_test.sh  (or panel "Auto-stop (sec)")
_AUTO_STOP_ENV = "VLA_AUTO_STOP_SEC"
AUTO_STOP_MIN_SEC = 5
AUTO_STOP_MAX_SEC = 900  # UI clamp; remote inference can be very slow


def _default_auto_stop_sec() -> int:
    raw = os.environ.get(_AUTO_STOP_ENV, "120")
    try:
        v = int(raw)
    except ValueError:
        return 120
    return max(AUTO_STOP_MIN_SEC, min(AUTO_STOP_MAX_SEC, v))


DEFAULT_TIMEOUT = _default_auto_stop_sec()


def _create_camera_debug_frustum(
    stage: Usd.Stage,
    camera_prim_path: str,
    *,
    depth_m: float = 0.18,
    line_width: float = 0.003,
    color: tuple[float, float, float] = (1.0, 0.65, 0.1),
) -> None:
    """Create a lightweight wireframe frustum as a child of a camera prim.

    This is a viewport/debug aid for Isaac Sim 5.1 when the built-in camera
    frustum gizmo is not visible. Since it is parented under the camera prim,
    it follows the camera automatically (including wrist camera motion).
    """
    cam_prim = stage.GetPrimAtPath(camera_prim_path)
    if not cam_prim.IsValid():
        print(f"[WARN] Cannot create debug frustum; camera prim not found: {camera_prim_path}", flush=True)
        return

    # Use camera intrinsics when available so frustum shape reflects FOV.
    cam = UsdGeom.Camera(cam_prim)
    focal = cam.GetFocalLengthAttr().Get() if cam else None
    h_ap = cam.GetHorizontalApertureAttr().Get() if cam else None
    v_ap = cam.GetVerticalApertureAttr().Get() if cam else None
    if focal is None or focal <= 1e-6:
        focal = 1.93
    if h_ap is None or h_ap <= 1e-6:
        h_ap = 2.65
    if v_ap is None or v_ap <= 1e-6:
        v_ap = h_ap

    half_w = max(0.01, depth_m * (h_ap / (2.0 * focal)))
    half_h = max(0.01, depth_m * (v_ap / (2.0 * focal)))

    # USD camera looks down -Z in local space.
    p0 = Gf.Vec3f(0.0, 0.0, 0.0)
    p1 = Gf.Vec3f(+half_w, +half_h, -depth_m)
    p2 = Gf.Vec3f(-half_w, +half_h, -depth_m)
    p3 = Gf.Vec3f(-half_w, -half_h, -depth_m)
    p4 = Gf.Vec3f(+half_w, -half_h, -depth_m)

    frustum_path = f"{camera_prim_path}/debug_frustum"
    if stage.GetPrimAtPath(frustum_path).IsValid():
        stage.RemovePrim(frustum_path)

    curves = UsdGeom.BasisCurves.Define(stage, frustum_path)
    curves.CreateTypeAttr("linear")
    # 8 independent line segments:
    # 4 rays from origin to far plane corners + 4 far-plane rectangle edges.
    curves.CreateCurveVertexCountsAttr([2, 2, 2, 2, 2, 2, 2, 2])
    curves.CreatePointsAttr([
        p0, p1,
        p0, p2,
        p0, p3,
        p0, p4,
        p1, p2,
        p2, p3,
        p3, p4,
        p4, p1,
    ])
    curves.CreateWidthsAttr([line_width])
    UsdGeom.Gprim(curves.GetPrim()).CreateDisplayColorAttr([Gf.Vec3f(*color)])
    UsdGeom.Imageable(curves.GetPrim()).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)

    print(f"[INFO] Added camera debug frustum: {frustum_path}", flush=True)


def _remove_ros2_bridge_graph():
    """Delete a partially-created OmniGraph if a previous attempt left it behind."""
    stage = omni.usd.get_context().get_stage()
    graph_prim = stage.GetPrimAtPath("/World/ROS2Bridge")
    if graph_prim.IsValid():
        stage.RemovePrim(graph_prim.GetPath())


# ═══════════════════════════════════════════════════════════════════════════
# Scene Setup
# ═══════════════════════════════════════════════════════════════════════════

def _setup_scene(sim: SimulationContext):
    """Spawn robot, cameras, ground, and lighting onto the USD stage.

    The wrist camera is parented to gripper_frame_link and moves with the end-effector.
    """
    global ROBOT_ARTICULATION_PRIM_PATH, ROBOT_MODEL_PRIM_PATH
    stage = omni.usd.get_context().get_stage()

    # -- Robot USD (soarm_common.robot, same cfg as the Isaac Lab envs) --
    robot_cfg = make_robot_cfg(ROBOT_PRIM_PATH)
    robot_cfg.spawn.func(ROBOT_PRIM_PATH, robot_cfg.spawn)

    # Resolve the actual articulation-root prim path to use for PhysX/ROS2 (with the
    # converter's articulation-root fix this is ROBOT_PRIM_PATH itself).
    try:
        from pxr import UsdPhysics
        _resolved = None

        # Search for the first active prim that has ArticulationRootAPI applied.
        _search_root = stage.GetPrimAtPath(ROBOT_PRIM_PATH)
        if _search_root.IsValid():
            for _p in Usd.PrimRange(_search_root):
                if _p.IsActive() and _p.HasAPI(UsdPhysics.ArticulationRootAPI):
                    _resolved = str(_p.GetPath())
                    break
        if _resolved is None:
            # Fallback: best effort
            _resolved = ROBOT_PRIM_PATH

        # Update global and cfg so everything uses the true articulation root.
        ROBOT_ARTICULATION_PRIM_PATH = _resolved
        ROBOT_MODEL_PRIM_PATH = ROBOT_PRIM_PATH
        robot_cfg.prim_path = _resolved
    except Exception:
        pass

    saved_spawn = robot_cfg.spawn
    robot_cfg.spawn = None
    robot = Articulation(robot_cfg)
    robot_cfg.spawn = saved_spawn

    # -- Ground plane and lighting (shared with the envs) --
    spawn_ground_and_light()

    # -- Container for user-spawned objects --
    stage.DefinePrim("/World/SpawnedObjects", "Xform")

    # -- Cameras: the contract's wrist and overhead cameras, as in data collection --
    stage.DefinePrim(C.INTERACTIVE_CAMERA_ROOT, "Xform")
    # Wrist camera under gripper_frame_link (Stage panel: World > Robot > Geometry > ... >
    # gripper_frame_link > wrist_cam); overhead camera at /World/Cameras/overhead_cam.
    _wrist_cam_prim = camera_prim_path(C.WRIST_CAMERA, ROBOT_MODEL_PRIM_PATH, CAMERA_ENV_PATH)
    _overhead_cam_prim = camera_prim_path(C.OVERHEAD_CAMERA, ROBOT_MODEL_PRIM_PATH, CAMERA_ENV_PATH)
    wrist_cam = Camera(make_camera_cfg(C.WRIST_CAMERA, _wrist_cam_prim))
    overhead_cam = Camera(make_camera_cfg(C.OVERHEAD_CAMERA, _overhead_cam_prim))

    # Debug frustum line meshes (workaround for missing camera FOV gizmos in some Isaac Sim 5.1 views).
    if args.snapshot is None:
        _create_camera_debug_frustum(stage, _wrist_cam_prim)
        _create_camera_debug_frustum(stage, _overhead_cam_prim)

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
    # Programmatically enable the extension via Kit's extension manager.
    # On first run this will auto-download from NVIDIA's extension registry
    # into the persistent isaac-cache-kit volume.
    for _ext_id in ("isaacsim.ros2.bridge", "omni.isaac.ros2_bridge"):
        try:
            import omni.kit.app as _kit_app_en
            _em = _kit_app_en.get_app().get_extension_manager()
            if not _em.is_extension_enabled(_ext_id):
                _em.set_extension_enabled_immediate(_ext_id, True)
            break
        except Exception:
            pass

    try:
        import isaacsim.ros2.bridge  # noqa: F401
        _bridge_ext_loaded = True
    except Exception:
        _bridge_ext_loaded = False

    # Try old name if new name failed (fallback for older Isaac Sim builds)
    try:
        import omni.isaac.ros2_bridge  # noqa: F401 – triggers extension load
    except (ImportError, ModuleNotFoundError):
        if not _bridge_ext_loaded:
            print("[WARN] omni.isaac.ros2_bridge not available – ROS2 topics disabled.", flush=True)
            return

    # Determine which OmniGraph node type prefix to use based on what loaded
    _og_prefix = "isaacsim.ros2.bridge" if _bridge_ext_loaded else "omni.isaac.ros2_bridge"

    keys = og.Controller.Keys

    _cn_prefix = "isaacsim.core.nodes"

    # -- Main action graph --
    # Create + connect in a single edit call (avoids graph wrap issues).
    # Isaac Sim 5.x: IsaacReadSimulationTime no longer exposes inputs:swhFrameNumber (see OgnIsaacReadSimulationTime
    # in isaacsim.core.nodes docs). Omitting that wire uses default behavior (latest simulation time on the output).
    ros2_graph_mode = os.environ.get("ROS2_GRAPH_MODE", "soarm").strip().lower()
    if ros2_graph_mode not in ("soarm", "minimal"):
        ros2_graph_mode = "soarm"
    print(f"[INFO] ROS2 graph mode: {ros2_graph_mode}", flush=True)
    _remove_ros2_bridge_graph()
    try:
        if ros2_graph_mode == "minimal":
            # Minimal tutorial-style graph: joint pub/sub + articulation control only.
            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": "/World/ROS2Bridge", "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("tick", "omni.graph.action.OnPlaybackTick"),
                        ("sim_time", f"{_cn_prefix}.IsaacReadSimulationTime"),
                        ("pub_joint", f"{_og_prefix}.ROS2PublishJointState"),
                        ("sub_joint_cmd", f"{_og_prefix}.ROS2SubscribeJointState"),
                        ("art_ctrl", f"{_cn_prefix}.IsaacArticulationController"),
                    ],
                    keys.SET_VALUES: [
                        ("pub_joint.inputs:topicName", C.JOINT_STATES_TOPIC),
                        ("pub_joint.inputs:targetPrim", ROBOT_ARTICULATION_PRIM_PATH),
                        ("sub_joint_cmd.inputs:topicName", C.JOINT_COMMANDS_TOPIC),
                        ("art_ctrl.inputs:robotPath", ROBOT_ARTICULATION_PRIM_PATH),
                    ],
                    keys.CONNECT: [
                        ("tick.outputs:tick", "pub_joint.inputs:execIn"),
                        ("tick.outputs:tick", "sub_joint_cmd.inputs:execIn"),
                        ("tick.outputs:tick", "art_ctrl.inputs:execIn"),
                        ("sim_time.outputs:simulationTime", "pub_joint.inputs:timeStamp"),
                        ("sub_joint_cmd.outputs:jointNames", "art_ctrl.inputs:jointNames"),
                        ("sub_joint_cmd.outputs:positionCommand", "art_ctrl.inputs:positionCommand"),
                        ("sub_joint_cmd.outputs:velocityCommand", "art_ctrl.inputs:velocityCommand"),
                        ("sub_joint_cmd.outputs:effortCommand", "art_ctrl.inputs:effortCommand"),
                    ],
                },
            )
        else:
            (graph, nodes, _, _) = og.Controller.edit(
                {"graph_path": "/World/ROS2Bridge", "evaluator_name": "execution"},
                {
                    keys.CREATE_NODES: [
                        ("tick", "omni.graph.action.OnPlaybackTick"),
                        ("sim_time", f"{_cn_prefix}.IsaacReadSimulationTime"),
                        ("pub_joint", f"{_og_prefix}.ROS2PublishJointState"),
                        ("sub_joint_cmd", f"{_og_prefix}.ROS2SubscribeJointState"),
                        ("art_ctrl", f"{_cn_prefix}.IsaacArticulationController"),
                        ("rp_wrist", f"{_cn_prefix}.IsaacCreateRenderProduct"),
                        ("rp_overhead", f"{_cn_prefix}.IsaacCreateRenderProduct"),
                        ("cam_wrist_helper", f"{_og_prefix}.ROS2CameraHelper"),
                        ("cam_overhead_helper", f"{_og_prefix}.ROS2CameraHelper"),
                    ],
                    keys.SET_VALUES: [
                        ("pub_joint.inputs:topicName", C.JOINT_STATES_TOPIC),
                        ("pub_joint.inputs:targetPrim", ROBOT_ARTICULATION_PRIM_PATH),
                        ("sub_joint_cmd.inputs:topicName", C.JOINT_COMMANDS_TOPIC),
                        ("art_ctrl.inputs:robotPath", ROBOT_ARTICULATION_PRIM_PATH),
                        ("cam_wrist_helper.inputs:topicName", C.WRIST_CAMERA.ros_topic),
                        ("cam_wrist_helper.inputs:type", "rgb"),
                        ("cam_overhead_helper.inputs:topicName", C.OVERHEAD_CAMERA.ros_topic),
                        ("cam_overhead_helper.inputs:type", "rgb"),
                        ("rp_wrist.inputs:cameraPrim", camera_prim_path(C.WRIST_CAMERA, ROBOT_MODEL_PRIM_PATH, CAMERA_ENV_PATH)),
                        ("rp_wrist.inputs:width", C.WRIST_CAMERA.width),
                        ("rp_wrist.inputs:height", C.WRIST_CAMERA.height),
                        ("rp_overhead.inputs:cameraPrim", camera_prim_path(C.OVERHEAD_CAMERA, ROBOT_MODEL_PRIM_PATH, CAMERA_ENV_PATH)),
                        ("rp_overhead.inputs:width", C.OVERHEAD_CAMERA.width),
                        ("rp_overhead.inputs:height", C.OVERHEAD_CAMERA.height),
                    ],
                    keys.CONNECT: [
                        ("tick.outputs:tick", "pub_joint.inputs:execIn"),
                        ("tick.outputs:tick", "sub_joint_cmd.inputs:execIn"),
                        ("tick.outputs:tick", "rp_wrist.inputs:execIn"),
                        ("tick.outputs:tick", "rp_overhead.inputs:execIn"),
                        ("rp_wrist.outputs:execOut", "cam_wrist_helper.inputs:execIn"),
                        ("rp_overhead.outputs:execOut", "cam_overhead_helper.inputs:execIn"),
                        ("rp_wrist.outputs:renderProductPath", "cam_wrist_helper.inputs:renderProductPath"),
                        ("rp_overhead.outputs:renderProductPath", "cam_overhead_helper.inputs:renderProductPath"),
                        ("sim_time.outputs:simulationTime", "pub_joint.inputs:timeStamp"),
                        ("sub_joint_cmd.outputs:jointNames", "art_ctrl.inputs:jointNames"),
                        ("sub_joint_cmd.outputs:positionCommand", "art_ctrl.inputs:positionCommand"),
                        ("sub_joint_cmd.outputs:velocityCommand", "art_ctrl.inputs:velocityCommand"),
                        ("sub_joint_cmd.outputs:effortCommand", "art_ctrl.inputs:effortCommand"),
                        ("sub_joint_cmd.outputs:execOut", "art_ctrl.inputs:execIn"),
                    ],
                },
            )

        print(f"[INFO] OmniGraph ROS2 bridge created (prefix={_og_prefix}, core={_cn_prefix}).", flush=True)
    except Exception as _og_exc:
        print(f"[WARN] OmniGraph ROS2 bridge creation failed: {_og_exc}", flush=True)
        # Fallback if graph creation still fails (e.g. extension mismatch). Omits sim_time + timestamp wiring.
        try:
            _remove_ros2_bridge_graph()
            if ros2_graph_mode == "minimal":
                og.Controller.edit(
                    {"graph_path": "/World/ROS2Bridge", "evaluator_name": "execution"},
                    {
                        keys.CREATE_NODES: [
                            ("tick", "omni.graph.action.OnPlaybackTick"),
                            ("pub_joint", f"{_og_prefix}.ROS2PublishJointState"),
                            ("sub_joint_cmd", f"{_og_prefix}.ROS2SubscribeJointState"),
                            ("art_ctrl", f"{_cn_prefix}.IsaacArticulationController"),
                        ],
                        keys.SET_VALUES: [
                            ("pub_joint.inputs:topicName", C.JOINT_STATES_TOPIC),
                            ("pub_joint.inputs:targetPrim", ROBOT_ARTICULATION_PRIM_PATH),
                            ("sub_joint_cmd.inputs:topicName", C.JOINT_COMMANDS_TOPIC),
                            ("art_ctrl.inputs:robotPath", ROBOT_ARTICULATION_PRIM_PATH),
                        ],
                        keys.CONNECT: [
                            ("tick.outputs:tick", "pub_joint.inputs:execIn"),
                            ("tick.outputs:tick", "sub_joint_cmd.inputs:execIn"),
                            ("tick.outputs:tick", "art_ctrl.inputs:execIn"),
                            ("sub_joint_cmd.outputs:jointNames", "art_ctrl.inputs:jointNames"),
                            ("sub_joint_cmd.outputs:positionCommand", "art_ctrl.inputs:positionCommand"),
                            ("sub_joint_cmd.outputs:velocityCommand", "art_ctrl.inputs:velocityCommand"),
                            ("sub_joint_cmd.outputs:effortCommand", "art_ctrl.inputs:effortCommand"),
                        ],
                    },
                )
            else:
                og.Controller.edit(
                    {"graph_path": "/World/ROS2Bridge", "evaluator_name": "execution"},
                    {
                        keys.CREATE_NODES: [
                            ("tick", "omni.graph.action.OnPlaybackTick"),
                            ("pub_joint", f"{_og_prefix}.ROS2PublishJointState"),
                            ("sub_joint_cmd", f"{_og_prefix}.ROS2SubscribeJointState"),
                            ("art_ctrl", f"{_cn_prefix}.IsaacArticulationController"),
                            ("rp_wrist", f"{_cn_prefix}.IsaacCreateRenderProduct"),
                            ("rp_overhead", f"{_cn_prefix}.IsaacCreateRenderProduct"),
                            ("cam_wrist_helper", f"{_og_prefix}.ROS2CameraHelper"),
                            ("cam_overhead_helper", f"{_og_prefix}.ROS2CameraHelper"),
                        ],
                        keys.SET_VALUES: [
                            ("pub_joint.inputs:topicName", C.JOINT_STATES_TOPIC),
                            ("pub_joint.inputs:targetPrim", ROBOT_ARTICULATION_PRIM_PATH),
                            ("sub_joint_cmd.inputs:topicName", C.JOINT_COMMANDS_TOPIC),
                            ("art_ctrl.inputs:robotPath", ROBOT_ARTICULATION_PRIM_PATH),
                            ("cam_wrist_helper.inputs:topicName", C.WRIST_CAMERA.ros_topic),
                            ("cam_wrist_helper.inputs:type", "rgb"),
                            ("cam_overhead_helper.inputs:topicName", C.OVERHEAD_CAMERA.ros_topic),
                            ("cam_overhead_helper.inputs:type", "rgb"),
                            ("rp_wrist.inputs:cameraPrim", camera_prim_path(C.WRIST_CAMERA, ROBOT_MODEL_PRIM_PATH, CAMERA_ENV_PATH)),
                            ("rp_wrist.inputs:width", C.WRIST_CAMERA.width),
                            ("rp_wrist.inputs:height", C.WRIST_CAMERA.height),
                            ("rp_overhead.inputs:cameraPrim", camera_prim_path(C.OVERHEAD_CAMERA, ROBOT_MODEL_PRIM_PATH, CAMERA_ENV_PATH)),
                            ("rp_overhead.inputs:width", C.OVERHEAD_CAMERA.width),
                            ("rp_overhead.inputs:height", C.OVERHEAD_CAMERA.height),
                        ],
                        keys.CONNECT: [
                            ("tick.outputs:tick", "pub_joint.inputs:execIn"),
                            ("tick.outputs:tick", "sub_joint_cmd.inputs:execIn"),
                            ("tick.outputs:tick", "rp_wrist.inputs:execIn"),
                            ("tick.outputs:tick", "rp_overhead.inputs:execIn"),
                            ("rp_wrist.outputs:execOut", "cam_wrist_helper.inputs:execIn"),
                            ("rp_overhead.outputs:execOut", "cam_overhead_helper.inputs:execIn"),
                            ("rp_wrist.outputs:renderProductPath", "cam_wrist_helper.inputs:renderProductPath"),
                            ("rp_overhead.outputs:renderProductPath", "cam_overhead_helper.inputs:renderProductPath"),
                            ("sub_joint_cmd.outputs:jointNames", "art_ctrl.inputs:jointNames"),
                            ("sub_joint_cmd.outputs:positionCommand", "art_ctrl.inputs:positionCommand"),
                            ("sub_joint_cmd.outputs:velocityCommand", "art_ctrl.inputs:velocityCommand"),
                            ("sub_joint_cmd.outputs:effortCommand", "art_ctrl.inputs:effortCommand"),
                            ("sub_joint_cmd.outputs:execOut", "art_ctrl.inputs:execIn"),
                        ],
                    },
                )
            print(f"[INFO] OmniGraph ROS2 bridge fallback created (prefix={_og_prefix}, core={_cn_prefix}).", flush=True)
        except Exception as _og_fallback_exc:
            print(f"[WARN] OmniGraph fallback also failed: {_og_fallback_exc}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════
# Object Spawning Helpers
# ═══════════════════════════════════════════════════════════════════════════

_spawn_counter = 0


def spawn_object(obj_type: str, size: float, pos: tuple, color: tuple):
    """Spawn a physics-enabled primitive under /World/SpawnedObjects/."""
    global _spawn_counter

    _spawn_counter += 1
    prim_path = f"/World/SpawnedObjects/obj_{_spawn_counter:03d}"

    common_rigid = PhysxRigidBodyPropertiesCfg(max_depenetration_velocity=1.0)
    common_mass = sim_utils.MassPropertiesCfg(mass=0.05)
    common_collision = PhysxCollisionPropertiesCfg()
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
_ros2_pub_joint_states = None
_ros2_pub_joint_commands = None


def _init_ros2_publishers():
    """Try to create lightweight ROS2 publishers for prompt and enabled topics.

    Uses rclpy if available inside the Isaac Sim Python environment.
    Falls back to file-based signaling if not.
    """
    global _ros2_pub_prompt, _ros2_pub_enabled, _ros2_pub_joint_states, _ros2_pub_joint_commands

    try:
        import rclpy
        from sensor_msgs.msg import JointState as JointStateMsg
        from std_msgs.msg import Bool as BoolMsg, String as StringMsg

        if not rclpy.ok():
            rclpy.init()
        node = rclpy.create_node("isaac_sim_ui_publisher")
        _ros2_pub_prompt = node.create_publisher(StringMsg, C.PROMPT_TOPIC, 10)
        _ros2_pub_enabled = node.create_publisher(BoolMsg, C.ENABLED_TOPIC, 10)
        # Fallback publisher: publish joint states from Isaac Lab articulation data directly.
        _ros2_pub_joint_states = node.create_publisher(JointStateMsg, C.JOINT_STATES_TOPIC, 10)
        # Reset helper: overwrite stale command targets latched by ROS2SubscribeJointState.
        _ros2_pub_joint_commands = node.create_publisher(JointStateMsg, C.JOINT_COMMANDS_TOPIC, 10)
        print(
            "[INFO] ROS2 publishers for /vla/prompt, /vla/enabled, /joint_states, and /joint_commands created via rclpy.",
            flush=True,
        )
        return node
    except Exception as e:
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


def publish_joint_states(robot: Articulation, ros2_node):
    """Publish sensor_msgs/JointState directly from articulation data (fallback path)."""
    global _ros2_pub_joint_states
    if _ros2_pub_joint_states is None or ros2_node is None:
        return
    if robot is None or (not robot.is_initialized):
        return
    try:
        from sensor_msgs.msg import JointState as JointStateMsg
        msg = JointStateMsg()
        msg.header.stamp = ros2_node.get_clock().now().to_msg()
        msg.name = list(SOARM_JOINT_NAMES)
        msg.position = robot.data.joint_pos.torch[0, :NUM_JOINTS].cpu().tolist()
        msg.velocity = robot.data.joint_vel.torch[0, :NUM_JOINTS].cpu().tolist()
        _ros2_pub_joint_states.publish(msg)
    except Exception:
        pass


def publish_joint_command_reset(positions: list[float] | None = None):
    """Publish a neutral command to clear stale /joint_commands targets in OmniGraph."""
    global _ros2_pub_joint_commands
    if _ros2_pub_joint_commands is None:
        return
    try:
        from sensor_msgs.msg import JointState as JointStateMsg
        msg = JointStateMsg()
        msg.name = list(SOARM_JOINT_NAMES)
        if positions is None:
            positions = [0.0] * NUM_JOINTS
        msg.position = list(positions[:NUM_JOINTS])
        msg.velocity = [float("nan")] * NUM_JOINTS
        msg.effort = [float("nan")] * NUM_JOINTS
        _ros2_pub_joint_commands.publish(msg)
        print(f"[OpenVLA debug] Published reset /joint_commands: {msg.position}", flush=True)
    except Exception:
        pass


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
                current = self._robot.data.joint_pos.torch.clone()
                self._robot.actuators.target_command.set_position_index(value=current)
            except Exception:
                pass

        self._update_button_states()
        self._status_label.text = "Status: STOPPED"

    def _on_estop(self):
        publish_enabled(False)
        self._state = STATE_STOPPED

        if self._robot is not None and self._robot.is_initialized:
            try:
                current = self._robot.data.joint_pos.torch.clone()
                self._robot.actuators.target_command.set_position_index(value=current)
                zeros = torch.zeros_like(current)
                self._robot.write_joint_state_to_sim_index(position=current, velocity=zeros)
            except Exception:
                pass

        self._update_button_states()
        self._status_label.text = "Status: E-STOPPED"

    def _on_reset_robot(self):
        # Disable inference first so the bridge flushes queued actions.
        publish_enabled(False)
        self._state = STATE_IDLE

        if self._robot is not None and self._robot.is_initialized:
            try:
                home = torch.zeros(1, NUM_JOINTS, device=self._robot.device)
                zeros = torch.zeros(1, NUM_JOINTS, device=self._robot.device)
                self._robot.write_joint_state_to_sim_index(position=home, velocity=zeros)
                self._robot.actuators.target_command.set_position_index(value=home)
                # Also overwrite any stale ROS-side command target held by sub_joint_cmd.
                publish_joint_command_reset([0.0] * NUM_JOINTS)
            except Exception:
                pass

        self._update_button_states()
        self._status_label.text = "Status: IDLE (hard reset)"

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
                jpos = self._robot.data.joint_pos.torch[0].cpu().numpy()
                for i, name in enumerate(SOARM_JOINT_NAMES):
                    if i < len(self._joint_labels):
                        self._joint_labels[i].text = f"  {name}: {jpos[i]:+.3f} rad"

                ee_idx = self._robot.find_bodies(C.EE_BODY_NAME)[0][0]
                ee_pos = self._robot.data.body_pos_w.torch[0, ee_idx].cpu().numpy()
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
                            self._timeout_sec = max(AUTO_STOP_MIN_SEC, min(AUTO_STOP_MAX_SEC, val))

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

def _snapshot(sim, robot, wrist_cam, overhead_cam, out_dir: str, joint_pos_spec: str):
    """Hold each joint pose, render, save both camera images (test mode for the frame-diff check).

    ``joint_pos_spec``: poses separated by ";", each 6 comma-separated joint positions (rad).
    Pose i is saved to ``<out_dir>/pose<i>/{wrist,overhead}.{png,npy}``.
    """
    from PIL import Image
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

    for i, pose in enumerate(p for p in joint_pos_spec.split(";") if p.strip()):
        q = torch.tensor([[float(v) for v in pose.split(",")]], device=robot.device)
        zeros = torch.zeros_like(q)
        for _ in range(30):
            robot.write_joint_state_to_sim_index(position=q, velocity=zeros)
            robot.actuators.target_command.set_position_index(value=q)
            robot.write_data_to_sim()
            sim.step(render=True)
            ensure_isaac_rtx_render_update(force=True)
            robot.update(sim.cfg.dt)
            wrist_cam.update(sim.cfg.dt, force_recompute=True)
            overhead_cam.update(sim.cfg.dt, force_recompute=True)
        pose_dir = os.path.join(out_dir, f"pose{i}")
        os.makedirs(pose_dir, exist_ok=True)
        for spec, cam in ((C.WRIST_CAMERA, wrist_cam), (C.OVERHEAD_CAMERA, overhead_cam)):
            img = cam.data.output["rgb"].torch[0, :, :, :3].cpu().numpy().astype(np.uint8)
            np.save(f"{pose_dir}/{spec.name}.npy", img)
            Image.fromarray(img).save(f"{pose_dir}/{spec.name}.png")
        print(f"[INFO] Snapshot saved to {pose_dir}: joint_pos={robot.data.joint_pos.torch[0].cpu().tolist()}",
              flush=True)


def main():
    global _ros2_node

    livestream = os.environ.get("LIVESTREAM") in ("1", "2")

    # -- Simulation context --
    # Contract timing and the envs' PhysX settings (incl. solve_articulation_contact_last, so grasps
    # behave as in data collection). CPU physics for the OmniGraph articulation controller.
    sim_cfg = SimulationCfg(
        dt=C.PHYSICS_DT, render_interval=C.RENDER_INTERVAL, device="cpu", physics=make_physx_cfg()
    )
    sim = SimulationContext(sim_cfg)

    # -- Build the scene --
    print("[INFO] Setting up scene (robot, cameras, lighting)...", flush=True)
    robot, wrist_cam, overhead_cam = _setup_scene(sim)
    # The ROS camera render products (and a livestream viewport) render every frame, so PhysX must
    # write body poses to Fabric every step. Isaac Lab 3.0 decides this when the SimulationContext
    # is created (before any camera exists) and turns it off in headless runs, which froze the
    # robot in the published camera images at its USD pose while the joints moved.
    sim.set_setting("/physics/fabricUpdateTransformations", True)

    # -- Reset simulation --
    sim.reset()
    robot.reset()

    # -- ROS2 OmniGraph bridge --
    # PhysX articulations/tensors are only fully initialized after reset (and often after at least 1 step).
    # Creating the ROS2 bridge before reset can cause "Failed to find articulation" errors.

    # Step once to ensure PhysX articulation is instantiated.
    sim.step()
    robot.update(sim.cfg.dt)
    check_joint_limits(robot)

    if args.snapshot is not None:
        _snapshot(sim, robot, wrist_cam, overhead_cam, args.snapshot, args.snapshot_joint_pos)
        simulation_app.close()
        return

    # Ensure playback-based OmniGraph triggers (OnPlaybackTick) are active in standalone mode.
    try:
        omni.timeline.get_timeline_interface().play()
    except Exception:
        pass

    print("[INFO] Creating OmniGraph ROS2 bridge...", flush=True)
    _setup_ros2_bridge()

    # Standalone scripts can step physics while timeline remains stopped.
    # ROS2 OmniGraph publishers driven by OnPlaybackTick only run in Play mode.
    try:
        sim.play()
    except Exception:
        pass
    try:
        omni.timeline.get_timeline_interface().play()
    except Exception:
        pass

    # -- ROS2 publishers for prompt/enabled --
    _ros2_node = _init_ros2_publishers()

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
            f"     viewport and select 'overhead_cam' or 'wrist_cam' (under Robot > Geometry > ... > gripper_frame_link).\n",
            flush=True,
        )

    # Isaac Lab 3.0 only pumps Kit (app.update) from a Kit visualizer (GUI / livestream) or when an
    # Isaac Lab camera fetches data. The OmniGraph ROS 2 graph (joint pub/sub, articulation controller,
    # camera helpers) and its render products only run inside app.update, so headless runs would
    # publish no images and ignore /joint_commands. Pump Kit on render steps; this is a no-op when a
    # Kit visualizer already does it.
    from isaaclab_physx.renderers.isaac_rtx_renderer_utils import ensure_isaac_rtx_render_update

    # -- Main simulation loop --
    step_count = 0
    while simulation_app.is_running():
        sim.step()
        step_count += 1
        if step_count % sim.cfg.render_interval == 0:
            ensure_isaac_rtx_render_update(force=True)

        # Update robot data
        robot.update(sim.cfg.dt)

        # Update telemetry + auto-stop every 6 steps (~20 Hz at 120 Hz sim)
        if step_count % 6 == 0:
            panel.update(sim.get_physics_dt() * step_count)
            # Ensure /joint_states is always available to the VLA bridge even if
            # OmniGraph ROS2 joint publisher is discovered but not actively emitting.
            publish_joint_states(robot, _ros2_node)

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
