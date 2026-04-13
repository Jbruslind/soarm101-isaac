"""Phospho-bridge: FastAPI server bridging a web dashboard to Isaac Sim via ROS2.

Subscribes to /joint_states, /camera/wrist/image_raw, /camera/overhead/image_raw.
Publishes to /joint_commands, /vla/prompt, /vla/enabled.
Serves a web dashboard at / and provides REST + WebSocket APIs.

Logging: stderr (docker logs) always; optional file via PHOSPHO_BRIDGE_LOG_FILE=/path/to.log
Debug: GET /api/debug/status (ROS thread, last message ages, camera skips)
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
import time
import traceback
from logging.handlers import RotatingFileHandler
from typing import Any, Optional

import cv2
import numpy as np
import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy
from sensor_msgs.msg import Image, JointState
from std_msgs.msg import Bool, String

from contextlib import asynccontextmanager

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ---------------------------------------------------------------------------
# Constants
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
BRIDGE_PORT = int(os.environ.get("PHOSPHO_BRIDGE_PORT", "8080"))
# Drop camera frames if Isaac publishes faster than this (reduces CPU/GIL load when sim is live).
_CAM_MIN_INTERVAL = float(os.environ.get("PHOSPHO_CAM_MIN_INTERVAL_SEC", "0.04"))  # ~25 Hz max per camera

_log = logging.getLogger("phospho_bridge")
_dashboard_html: Optional[str] = None
_ros_spin_error: Optional[str] = None
_ros_spin_started_monotonic: float = 0.0


def _setup_logging() -> None:
    """Attach handlers only to phospho_bridge (do not clear root — keeps uvicorn access logs)."""
    if _log.handlers:
        return
    level_name = os.environ.get("PHOSPHO_BRIDGE_LOG_LEVEL", "INFO").upper()
    level = getattr(logging, level_name, logging.INFO)
    _log.setLevel(level)
    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(threadName)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    h = logging.StreamHandler(sys.stderr)
    h.setFormatter(fmt)
    _log.addHandler(h)
    log_path = os.environ.get("PHOSPHO_BRIDGE_LOG_FILE", "").strip()
    if log_path:
        try:
            fh = RotatingFileHandler(
                log_path, maxBytes=8 * 1024 * 1024, backupCount=3, encoding="utf-8"
            )
            fh.setFormatter(fmt)
            _log.addHandler(fh)
        except OSError as e:
            print(f"[phospho-bridge] WARNING: could not open PHOSPHO_BRIDGE_LOG_FILE={log_path}: {e}", file=sys.stderr)
    _log.propagate = False
    _log.info(
        "Logger ready (level=%s). Tail: docker logs -f soarm-phospho-bridge | grep phospho_bridge",
        level_name,
    )
    if log_path:
        _log.info("Also writing rotating log file: %s", log_path)

# ---------------------------------------------------------------------------
# Pydantic models for the REST API
# ---------------------------------------------------------------------------

class JointCommandRequest(BaseModel):
    positions: list[float]
    joint_names: Optional[list[str]] = None

class VLAPromptRequest(BaseModel):
    prompt: str

class VLAExecuteRequest(BaseModel):
    prompt: str
    auto_stop_sec: Optional[int] = 120


def image_msg_to_bgr(msg: Image) -> np.ndarray:
    """Decode sensor_msgs/Image to BGR uint8 without cv_bridge (avoids NumPy 1.x Boost ABI issues)."""
    if getattr(msg, "is_bigendian", False):
        raise ValueError("big-endian images are not supported")
    h, w = int(msg.height), int(msg.width)
    step = int(msg.step)
    enc = (msg.encoding or "").lower()
    buf = np.frombuffer(msg.data, dtype=np.uint8)
    need = h * step
    if buf.size < need:
        raise ValueError(f"image buffer too small: {buf.size} < {need}")
    raw = buf[:need].reshape((h, step))

    if enc == "bgr8":
        cols = w * 3
        return np.ascontiguousarray(raw[:, :cols].reshape((h, w, 3)))
    if enc == "rgb8":
        cols = w * 3
        rgb = np.ascontiguousarray(raw[:, :cols].reshape((h, w, 3)))
        return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    if enc == "rgba8":
        cols = w * 4
        rgba = np.ascontiguousarray(raw[:, :cols].reshape((h, w, 4)))
        return cv2.cvtColor(rgba, cv2.COLOR_RGBA2BGR)
    if enc == "bgra8":
        cols = w * 4
        bgra = np.ascontiguousarray(raw[:, :cols].reshape((h, w, 4)))
        return cv2.cvtColor(bgra, cv2.COLOR_BGRA2BGR)
    if enc in ("mono8", "8uc1", "uint8"):
        cols = w
        gray = np.ascontiguousarray(raw[:, :cols].reshape((h, w)))
        return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    raise ValueError(f"unsupported image encoding {msg.encoding!r}")


# ---------------------------------------------------------------------------
# ROS2 Bridge Node
# ---------------------------------------------------------------------------

class PhosphoBridgeNode(Node):
    """ROS2 node that relays between the web dashboard and Isaac Sim topics."""

    def __init__(self):
        super().__init__("phospho_bridge")

        self._lock = threading.Lock()
        self._joint_positions: list[float] = [0.0] * NUM_JOINTS
        self._joint_velocities: list[float] = [0.0] * NUM_JOINTS
        self._joint_names: list[str] = list(SOARM_JOINT_NAMES)
        self._joint_stamp: float = 0.0
        self._joint_rx_count: int = 0
        self._logged_first_joint: bool = False

        self._camera_frames: dict[str, Optional[bytes]] = {
            "wrist": None,
            "overhead": None,
        }
        self._camera_stamps: dict[str, float] = {"wrist": 0.0, "overhead": 0.0}
        self._cam_last_encode_t: dict[str, float] = {"wrist": 0.0, "overhead": 0.0}
        self._cam_skipped: dict[str, int] = {"wrist": 0, "overhead": 0}
        self._cam_encoded: dict[str, int] = {"wrist": 0, "overhead": 0}
        self._logged_first_cam: dict[str, bool] = {"wrist": False, "overhead": False}

        # Match soarm_vla_bridge: Isaac's ROS2PublishJointState uses RELIABLE; BEST_EFFORT
        # here yields discovery without sample delivery (no joint telemetry, broken UI sync).
        joint_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )
        # Cameras: same as vla_bridge (Isaac ROS2CameraHelper is typically best-effort).
        camera_qos = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            history=HistoryPolicy.KEEP_LAST,
            depth=1,
        )

        self.create_subscription(JointState, "/joint_states", self._on_joint_states, joint_qos)
        self.create_subscription(Image, "/camera/wrist/image_raw", self._on_wrist_image, camera_qos)
        self.create_subscription(Image, "/camera/overhead/image_raw", self._on_overhead_image, camera_qos)

        # Match vla_bridge create_publisher(JointState, "/joint_commands", 10)
        cmd_qos = QoSProfile(
            reliability=ReliabilityPolicy.RELIABLE,
            history=HistoryPolicy.KEEP_LAST,
            depth=10,
        )

        self.pub_joint_cmd = self.create_publisher(JointState, "/joint_commands", cmd_qos)
        self.pub_vla_prompt = self.create_publisher(String, "/vla/prompt", cmd_qos)
        self.pub_vla_enabled = self.create_publisher(Bool, "/vla/enabled", cmd_qos)

        self.get_logger().info("PhosphoBridgeNode initialized")

    # -- Callbacks --

    def _on_joint_states(self, msg: JointState):
        first = False
        n = 0
        snap_name: list[str] = []
        snap_pos_head: list[float] = []
        with self._lock:
            n = min(len(msg.position), NUM_JOINTS)
            self._joint_names = list(msg.name[:n])
            self._joint_positions = list(msg.position[:n])
            self._joint_velocities = list(msg.velocity[:n]) if msg.velocity else [0.0] * n
            self._joint_stamp = time.time()
            self._joint_rx_count += 1
            if not self._logged_first_joint:
                self._logged_first_joint = True
                first = True
                snap_name = list(msg.name[:n])
                snap_pos_head = list(msg.position[: min(3, len(msg.position))])
        if first:
            _log.info(
                "First /joint_states (%d joints); names=%s pos[0:3]=%s",
                n,
                snap_name,
                snap_pos_head,
            )

    def _on_wrist_image(self, msg: Image):
        self._encode_camera_frame("wrist", msg)

    def _on_overhead_image(self, msg: Image):
        self._encode_camera_frame("overhead", msg)

    def _encode_camera_frame(self, name: str, msg: Image):
        now = time.time()
        with self._lock:
            if now - self._cam_last_encode_t[name] < _CAM_MIN_INTERVAL:
                self._cam_skipped[name] += 1
                return
            # Reserve the next slot so parallel executor threads do not both encode.
            self._cam_last_encode_t[name] = now
        try:
            cv_img = image_msg_to_bgr(msg)
            _, jpeg = cv2.imencode(".jpg", cv_img, [cv2.IMWRITE_JPEG_QUALITY, 75])
            jpeg_bytes = jpeg.tobytes()
            with self._lock:
                self._camera_frames[name] = jpeg_bytes
                self._camera_stamps[name] = time.time()
                self._cam_encoded[name] += 1
                log_cam = not self._logged_first_cam[name]
                if log_cam:
                    self._logged_first_cam[name] = True
            if log_cam:
                _log.info(
                    "First /camera/%s/image_raw encoding=%s shape=%s jpeg_bytes=%d",
                    name,
                    getattr(msg, "encoding", "?"),
                    getattr(cv_img, "shape", None),
                    len(jpeg_bytes),
                )
        except Exception as e:
            self.get_logger().warn(f"Camera encode error ({name}, encoding={getattr(msg, 'encoding', '?')}): {e}")
            _log.warning("Camera encode error (%s): %s", name, e)

    # -- Public getters (thread-safe) --

    def get_joint_state(self) -> dict:
        with self._lock:
            return {
                "names": list(self._joint_names),
                "positions": list(self._joint_positions),
                "velocities": list(self._joint_velocities),
                "stamp": self._joint_stamp,
            }

    def get_camera_jpeg(self, name: str) -> Optional[bytes]:
        with self._lock:
            return self._camera_frames.get(name)

    # -- Public publishers --

    def send_joint_command(self, positions: list[float], names: Optional[list[str]] = None):
        msg = JointState()
        msg.name = names or list(SOARM_JOINT_NAMES)
        msg.position = [float(p) for p in positions]
        msg.velocity = [float("nan")] * len(positions)
        msg.effort = [float("nan")] * len(positions)
        self.pub_joint_cmd.publish(msg)

    def send_vla_prompt(self, prompt: str):
        msg = String()
        msg.data = prompt
        self.pub_vla_prompt.publish(msg)

    def send_vla_enabled(self, enabled: bool):
        msg = Bool()
        msg.data = enabled
        self.pub_vla_enabled.publish(msg)

    def debug_snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "joint_stamp_wall": self._joint_stamp,
                "joint_rx_count": self._joint_rx_count,
                "cam_encoded": dict(self._cam_encoded),
                "cam_skipped": dict(self._cam_skipped),
                "cam_stamps_wall": dict(self._camera_stamps),
                "cam_min_interval_sec": _CAM_MIN_INTERVAL,
            }


# ---------------------------------------------------------------------------
# ROS2 spin thread
# ---------------------------------------------------------------------------

_ros2_node: Optional[PhosphoBridgeNode] = None
_ros2_thread: Optional[threading.Thread] = None


def _spin_ros2():
    global _ros2_node, _ros_spin_error, _ros_spin_started_monotonic
    _ros_spin_error = None
    _ros_spin_started_monotonic = time.monotonic()
    node_local: Optional[PhosphoBridgeNode] = None
    try:
        rclpy.init()
        node_local = PhosphoBridgeNode()
        _ros2_node = node_local
        n_threads = int(os.environ.get("PHOSPHO_ROS_EXECUTOR_THREADS", "4"))
        n_threads = max(2, n_threads)
        _log.info("ROS2 MultiThreadedExecutor starting (threads=%d)", n_threads)
        executor = MultiThreadedExecutor(num_threads=n_threads)
        executor.add_node(node_local)
        executor.spin()
    except Exception:
        _ros_spin_error = traceback.format_exc()
        _log.exception("ROS2 executor failed")
    finally:
        _ros2_node = None
        try:
            if node_local is not None:
                node_local.destroy_node()
        except Exception as e:
            _log.warning("destroy_node: %s", e)
        try:
            if rclpy.ok():
                rclpy.shutdown()
        except Exception:
            pass
        _log.warning(
            "ROS2 background thread ended — ros2_ready will be false until restart "
            "(Isaac still running is OK; check DDS / domain / this log for errors above)."
        )


def start_ros2_async():
    """Start ROS2 spin in a daemon thread without blocking HTTP server startup.

    Binds to the network immediately so other machines can reach / and /api/health
    while rclpy initializes in the background.
    """
    global _ros2_thread
    if _ros2_thread is not None and _ros2_thread.is_alive():
        return
    _ros2_thread = threading.Thread(target=_spin_ros2, name="ros2-executor", daemon=True)
    _ros2_thread.start()
    _log.info("ROS2 thread started (name=ros2-executor)")


def get_node() -> Optional[PhosphoBridgeNode]:
    return _ros2_node


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(application: FastAPI):
    global _dashboard_html
    _setup_logging()
    app_dir = os.path.dirname(os.path.abspath(__file__))
    idx_path = os.path.join(app_dir, "static", "index.html")
    try:
        with open(idx_path, encoding="utf-8") as f:
            _dashboard_html = f.read()
        _log.info("Dashboard loaded from %s (%d bytes)", idx_path, len(_dashboard_html))
    except OSError as e:
        _log.error("Failed to read dashboard %s: %s", idx_path, e)
        _dashboard_html = (
            f"<html><body><h1>phospho-bridge</h1><p>Missing or unreadable {idx_path}: {e}</p></body></html>"
        )
    start_ros2_async()
    yield


app = FastAPI(title="SO-ARM101 Phospho Bridge", docs_url="/docs", lifespan=lifespan)


# -- Static files (dashboard) --
app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/", response_class=HTMLResponse)
async def root():
    if _dashboard_html is None:
        return HTMLResponse("Dashboard not initialized", status_code=500)
    return HTMLResponse(_dashboard_html)


@app.get("/api/health")
async def health():
    """Liveness probe; does not require ROS2 (use from another machine to verify routing/firewall)."""
    node = get_node()
    ros_thread_alive = _ros2_thread is not None and _ros2_thread.is_alive()
    return {
        "ok": True,
        "ros2_ready": node is not None,
        "ros2_thread_alive": ros_thread_alive,
        "ros_spin_error": _ros_spin_error is not None,
        "port": BRIDGE_PORT,
        "listen": "0.0.0.0",
    }


@app.get("/api/debug/status")
async def debug_status():
    """Diagnostics when the UI hangs or ROS drops (JSON for curl / browser)."""
    alive = _ros2_thread is not None and _ros2_thread.is_alive()
    uptime = None
    if alive and _ros_spin_started_monotonic > 0:
        uptime = round(time.monotonic() - _ros_spin_started_monotonic, 3)
    out: dict[str, Any] = {
        "ros2_thread_alive": alive,
        "ros2_thread_name": getattr(_ros2_thread, "name", None) if _ros2_thread else None,
        "ros2_node_present": get_node() is not None,
        "ros_spin_error_trace": _ros_spin_error,
        "ros_thread_uptime_sec": uptime,
        "bridge_port": BRIDGE_PORT,
        "cam_min_interval_sec": _CAM_MIN_INTERVAL,
    }
    node = get_node()
    if node is not None:
        out["node"] = node.debug_snapshot()
    return JSONResponse(out)


# -- Joint state API --

@app.get("/api/joints")
async def get_joints():
    node = get_node()
    if node is None:
        return JSONResponse({"detail": "ROS2 node not ready yet"}, status_code=503)
    return JSONResponse(node.get_joint_state())


@app.post("/api/joints/command")
async def post_joint_command(req: JointCommandRequest):
    node = get_node()
    if node is None:
        return JSONResponse({"detail": "ROS2 node not ready yet"}, status_code=503)
    node.send_joint_command(req.positions, req.joint_names)
    return {"status": "ok"}


@app.post("/api/joints/home")
async def post_joint_home():
    node = get_node()
    if node is None:
        return JSONResponse({"detail": "ROS2 node not ready yet"}, status_code=503)
    node.send_joint_command([0.0] * NUM_JOINTS)
    return {"status": "ok"}


# -- VLA API --

@app.post("/api/vla/execute")
async def post_vla_execute(req: VLAExecuteRequest):
    node = get_node()
    if node is None:
        return JSONResponse({"detail": "ROS2 node not ready yet"}, status_code=503)
    node.send_vla_prompt(req.prompt)
    node.send_vla_enabled(True)
    return {"status": "executing", "prompt": req.prompt}


@app.post("/api/vla/stop")
async def post_vla_stop():
    node = get_node()
    if node is None:
        return JSONResponse({"detail": "ROS2 node not ready yet"}, status_code=503)
    node.send_vla_enabled(False)
    # Hold current position
    state = node.get_joint_state()
    if state["positions"]:
        node.send_joint_command(state["positions"])
    return {"status": "stopped"}


@app.post("/api/vla/estop")
async def post_vla_estop():
    node = get_node()
    if node is None:
        return JSONResponse({"detail": "ROS2 node not ready yet"}, status_code=503)
    node.send_vla_enabled(False)
    state = node.get_joint_state()
    if state["positions"]:
        node.send_joint_command(state["positions"])
    return {"status": "e-stopped"}


# -- Camera MJPEG streaming (async iterator — avoids blocking the asyncio event loop) --

async def _mjpeg_async(camera_name: str):
    boundary = b"--frame\r\n"
    while True:
        node = get_node()
        if node is None:
            await asyncio.sleep(0.2)
            continue
        # Short lock hold in thread pool so ROS callbacks are not stalled by slow clients.
        frame = await asyncio.to_thread(node.get_camera_jpeg, camera_name)
        if frame is not None:
            yield boundary + b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
        await asyncio.sleep(0.066)  # ~15 fps cap per stream


@app.get("/api/camera/{name}")
async def get_camera_stream(name: str):
    if name not in ("wrist", "overhead"):
        return JSONResponse({"error": f"Unknown camera: {name}"}, status_code=404)
    return StreamingResponse(
        _mjpeg_async(name),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# -- WebSocket telemetry --

@app.websocket("/ws/telemetry")
async def ws_telemetry(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            node = get_node()
            if node is None:
                await websocket.send_text(json.dumps({"ros2_ready": False}))
            else:
                state = node.get_joint_state()
                await websocket.send_text(json.dumps(state))
            await asyncio.sleep(0.05)  # 20 Hz
    except WebSocketDisconnect:
        pass
    except Exception as e:
        _log.warning("WebSocket telemetry ended: %s", e)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    _setup_logging()
    _log.info(
        "Uvicorn starting on 0.0.0.0:%s — logs: stderr + PHOSPHO_BRIDGE_LOG_FILE if set; "
        "diagnostics: GET /api/debug/status",
        BRIDGE_PORT,
    )
    uvicorn.run(app, host="0.0.0.0", port=BRIDGE_PORT, log_level="info")
