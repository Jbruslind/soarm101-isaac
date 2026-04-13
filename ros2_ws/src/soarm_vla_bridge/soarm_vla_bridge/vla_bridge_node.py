"""ROS2 node that bridges robot observations to the OpenPi VLA policy server.

Subscribes to:
  /joint_states       (sensor_msgs/JointState)
  /camera/wrist/image_raw   (sensor_msgs/Image)
  /camera/overhead/image_raw (sensor_msgs/Image, optional)
  /vla/prompt         (std_msgs/String)
  /vla/enabled        (std_msgs/Bool)  -- gates inference on/off

Publishes:
  /joint_commands     (sensor_msgs/JointState)

Connects to the OpenPi policy server via WebSocket. Supports both local
(docker network) and remote (cloud GPU) endpoints via OPENPI_HOST env var.
Uses action chunking with a background inference thread to tolerate
network latency on WAN connections.
"""

from __future__ import annotations

import os
import threading
from collections import deque

import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy

from sensor_msgs.msg import JointState, Image
from std_msgs.msg import Bool, String
from builtin_interfaces.msg import Duration

from soarm_vla_bridge.observation_builder import (
    build_observation,
    SOARM_JOINT_NAMES,
    POLICY_MODE_SOARM,
    POLICY_MODE_DROID,
)

try:
    from openpi_client import websocket_client_policy
except ImportError:
    websocket_client_policy = None

try:
    import websockets.sync.client as ws_sync_client
    from openpi_client import msgpack_numpy
except Exception:
    ws_sync_client = None
    msgpack_numpy = None


class _OpenPiWebsocketClient:
    """Minimal websocket client with configurable keepalive timeouts.

    openpi-client's WebsocketClientPolicy uses websockets defaults (ping_interval
    and ping_timeout). If model inference is slow (common on Jetson/DROID
    fallback), the keepalive can trigger a 1011 and abort the request.

    This client lets us increase ping_timeout so "slow is okay" for demos.
    """

    def __init__(
        self,
        host: str,
        port: int,
        *,
        ping_interval: float = 60.0,
        ping_timeout: float = 120.0,
        api_key: str | None = None,
    ):
        if ws_sync_client is None or msgpack_numpy is None:
            raise RuntimeError("websockets or openpi_client.msgpack_numpy not available")

        self._uri = host if host.startswith("ws") else f"ws://{host}:{port}"
        self._packer = msgpack_numpy.Packer()
        self._api_key = api_key
        self._ping_interval = ping_interval
        self._ping_timeout = ping_timeout
        self._ws, self._server_metadata = self._wait_for_server()

    def get_server_metadata(self) -> dict:
        return self._server_metadata

    def _wait_for_server(self):
        import logging
        import time

        open_timeout_sec = float(os.environ.get("OPENPI_WS_OPEN_TIMEOUT_SEC", "60"))
        logging.info(f"Waiting for OpenPi websocket server at {self._uri}...")
        while True:
            try:
                headers = {"Authorization": f"Api-Key {self._api_key}"} if self._api_key else None
                conn = ws_sync_client.connect(
                    self._uri,
                    compression=None,
                    max_size=None,
                    additional_headers=headers,
                    ping_interval=self._ping_interval,
                    ping_timeout=self._ping_timeout,
                    open_timeout=open_timeout_sec,
                )
                metadata = msgpack_numpy.unpackb(conn.recv())
                return conn, metadata
            except (ConnectionRefusedError, TimeoutError, OSError) as e:
                # OpenPi may be restarting or still warming up (model load can take
                # tens of seconds). Keep retrying until the websocket handshake
                # completes and metadata can be read.
                logging.info(f"OpenPi websocket not ready yet ({type(e).__name__}): {e}; retrying...")
                time.sleep(2.0)

    def infer(self, obs: dict) -> dict:
        data = self._packer.pack(obs)
        self._ws.send(data)
        response = self._ws.recv()
        if isinstance(response, str):
            # we expect bytes for msgpack
            raise RuntimeError(f"Error in inference server:\n{response}")
        return msgpack_numpy.unpackb(response)


class VLABridgeNode(Node):
    """Bridges ROS2 sensor topics to OpenPi VLA policy server."""

    def __init__(self):
        super().__init__("vla_bridge")

        # Parameters
        self.declare_parameter("openpi_host", os.environ.get("OPENPI_HOST", "localhost"))
        self.declare_parameter("openpi_port", int(os.environ.get("OPENPI_PORT", "8000")))
        self.declare_parameter("control_rate", 10.0)
        self.declare_parameter("prompt", "move the robot arm to the target")
        self.declare_parameter("action_chunk_size", 1)

        host = self.get_parameter("openpi_host").value
        port = self.get_parameter("openpi_port").value
        rate = self.get_parameter("control_rate").value

        self.current_prompt = self.get_parameter("prompt").value
        self.chunk_size = self.get_parameter("action_chunk_size").value
        self.policy_mode = os.environ.get("OPENPI_POLICY_MODE", POLICY_MODE_SOARM).strip().lower()
        self.policy_config_name = os.environ.get("OPENPI_POLICY_CONFIG", "").strip()
        if self.policy_mode not in (POLICY_MODE_SOARM, POLICY_MODE_DROID):
            self.get_logger().warn(
                f"[OpenVLA] Invalid OPENPI_POLICY_MODE={self.policy_mode!r}; defaulting to '{POLICY_MODE_SOARM}'."
            )
            self.policy_mode = POLICY_MODE_SOARM
        if self.policy_mode == POLICY_MODE_SOARM and self.policy_config_name and not self.policy_config_name.startswith("soarm_"):
            self.get_logger().warn(
                f"[OpenVLA] OPENPI_POLICY_MODE=soarm but OPENPI_POLICY_CONFIG={self.policy_config_name!r} "
                "does not look like a SOARM config."
            )
        if self.policy_mode == POLICY_MODE_DROID and self.policy_config_name.startswith("soarm_"):
            self.get_logger().warn(
                f"[OpenVLA] OPENPI_POLICY_MODE=droid with SOARM config {self.policy_config_name!r}; "
                "this mismatch can cause incoherent trajectories."
            )

        # State
        self.latest_joint_pos: dict[str, float] = {}
        self.latest_wrist_image: np.ndarray | None = None
        self.latest_overhead_image: np.ndarray | None = None
        self.action_queue: deque = deque()
        self._lock = threading.Lock()
        self._inference_active = True  # default on for backward compat

        # OpenPi client
        self._openpi_host = host
        self._openpi_port = port
        self._debug_inference_count = 0
        self._debug_control_publish_count = 0
        self._debug_last_empty_log_time = 0.0
        self._warned_no_joint_state = False
        # True while blocked in OpenPi infer(); control loop uses this for clearer empty-queue logs.
        self._infer_in_flight = False
        self._warned_missing_wrist = False
        self._warned_missing_overhead = False
        self._warned_joint_name_mismatch = False
        self._preflight_ok = False

        self._require_wrist_image = os.environ.get("OPENPI_REQUIRE_WRIST_IMAGE", "1") == "1"
        self._require_overhead_image = os.environ.get("OPENPI_REQUIRE_OVERHEAD_IMAGE", "0") == "1"
        self._strict_joint_name_check = os.environ.get("OPENPI_STRICT_JOINT_NAMES", "1") == "1"
        self._expected_action_dim = (
            len(SOARM_JOINT_NAMES) if self.policy_mode == POLICY_MODE_SOARM else 8
        )

        # Websocket keepalive tuning: slow demos (Jetson + DROID fallback)
        # can take longer than the default ping timeout.
        self._ws_ping_interval = float(os.environ.get("OPENPI_WS_PING_INTERVAL_SEC", "60"))
        self._ws_ping_timeout = float(os.environ.get("OPENPI_WS_PING_TIMEOUT_SEC", "180"))

        # Optional inference request throttle (so you can target ~1Hz demos).
        self._infer_interval_sec = float(os.environ.get("OPENPI_INFER_INTERVAL_SEC", "0"))

        if websocket_client_policy is not None:
            self.get_logger().info(
                f"[OpenVLA] Connecting to OpenPi server at {host}:{port} "
                f"(OPENPI_HOST={os.environ.get('OPENPI_HOST', 'not set')}, "
                f"OPENPI_PORT={os.environ.get('OPENPI_PORT', 'not set')})"
            )
            # Prefer our custom client to avoid keepalive timeouts on slow inference.
            self.client = _OpenPiWebsocketClient(
                host=host,
                port=port,
                ping_interval=self._ws_ping_interval,
                ping_timeout=self._ws_ping_timeout,
            )
            # Optional: TCP reachability probe (disabled by default).
            # Note: Opening a raw TCP socket to a WebSocket server and closing immediately
            # will show up server-side as a "handshake failed" (no valid HTTP request).
            if os.environ.get("OPENPI_TCP_PROBE", "0") == "1":
                try:
                    import socket

                    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                    s.settimeout(3.0)
                    s.connect((host, port))
                    s.close()
                    self.get_logger().info(f"[OpenVLA] TCP connect to {host}:{port} succeeded (port open)")
                except Exception as e:
                    self.get_logger().warn(
                        f"[OpenVLA] TCP connect to {host}:{port} failed: {e} "
                        "(Inference will still try WebSocket; this is just a reachability hint.)"
                    )
        else:
            self.get_logger().warn(
                "openpi-client not installed. Running in dry-run mode."
            )
            self.client = None

        # Subscribers
        # Isaac Sim ROS2PublishJointState commonly uses RELIABLE QoS; using BEST_EFFORT
        # here can result in discovery without actual sample delivery.
        joint_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.RELIABLE)
        camera_qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, joint_qos
        )
        self.wrist_cam_sub = self.create_subscription(
            Image, "/camera/wrist/image_raw", self._wrist_image_cb, camera_qos
        )
        self.overhead_cam_sub = self.create_subscription(
            Image, "/camera/overhead/image_raw", self._overhead_image_cb, camera_qos
        )
        self.prompt_sub = self.create_subscription(
            String, "/vla/prompt", self._prompt_cb, 10
        )
        self.enabled_sub = self.create_subscription(
            Bool, "/vla/enabled", self._enabled_cb, 10
        )

        # Publisher: Isaac Sim's ROS2SubscribeJointState expects `sensor_msgs/JointState`
        # commands (position/velocity/effort arrays), not `trajectory_msgs/JointTrajectory`.
        self.action_pub = self.create_publisher(JointState, "/joint_commands", 10)

        # Control timer
        period = 1.0 / rate
        self.timer = self.create_timer(period, self._control_loop)

        # Background inference thread
        self._inference_thread = threading.Thread(
            target=self._inference_loop, daemon=True
        )
        self._inference_thread.start()

        # When Isaac Sim has no rclpy it writes to /tmp/vla_signals; poll so we still get Execute
        self._vla_signals_dir = os.environ.get("VLA_SIGNALS_DIR", "/tmp/vla_signals")
        self._file_signals_thread = threading.Thread(
            target=self._file_signals_loop, daemon=True
        )
        self._file_signals_thread.start()

        self.get_logger().info("VLA Bridge node started")
        self.get_logger().info(
            f"[OpenVLA] Policy mode: {self.policy_mode} "
            f"(expected action dim={self._expected_action_dim}, action_chunk_size={self.chunk_size})"
        )

    # -- Callbacks --

    def _joint_cb(self, msg: JointState):
        for name, pos in zip(msg.name, msg.position):
            self.latest_joint_pos[name] = pos

    def _wrist_image_cb(self, msg: Image):
        self.latest_wrist_image = self._image_msg_to_numpy(msg)

    def _overhead_image_cb(self, msg: Image):
        self.latest_overhead_image = self._image_msg_to_numpy(msg)

    def _prompt_cb(self, msg: String):
        self.current_prompt = msg.data
        self.get_logger().info(f"Prompt updated: {msg.data}")

    def _file_signals_loop(self):
        """Poll file-based signals when Isaac Sim has no ROS2 (shared volume /tmp/vla_signals)."""
        import time
        last_enabled: str | None = None
        last_prompt: str | None = None
        while rclpy.ok():
            time.sleep(0.2)
            d = self._vla_signals_dir
            try:
                enabled_path = os.path.join(d, "enabled.txt")
                if os.path.isfile(enabled_path):
                    with open(enabled_path) as f:
                        val = f.read().strip()
                    if val != last_enabled:
                        last_enabled = val
                        self._inference_active = val == "1"
                        if not self._inference_active:
                            with self._lock:
                                self.action_queue.clear()
                            self.get_logger().info("Inference DISABLED (file signal) – action queue flushed")
                        else:
                            self.get_logger().info(
                                f"[OpenVLA] Inference ENABLED (file signal). Will request actions from "
                                f"{self._openpi_host}:{self._openpi_port}"
                            )
                            self.get_logger().info(
                                "[OpenVLA] First OpenPi response can take 30-120s on slow/remote GPUs; "
                                "keep enabled until you see 'OpenPi returned ...'."
                            )
                prompt_path = os.path.join(d, "prompt.txt")
                if os.path.isfile(prompt_path):
                    with open(prompt_path) as f:
                        p = f.read().strip()
                    if p != last_prompt:
                        last_prompt = p
                        self.current_prompt = p
                        self.get_logger().info(f"Prompt updated (file signal): {p}")
            except (OSError, IOError):
                pass

    def _enabled_cb(self, msg: Bool):
        was_active = self._inference_active
        self._inference_active = msg.data
        if was_active and not msg.data:
            with self._lock:
                self.action_queue.clear()
            self.get_logger().info("Inference DISABLED – action queue flushed")
        elif not was_active and msg.data:
            self.get_logger().info(
                f"[OpenVLA] Inference ENABLED (Execute received). Will request actions from "
                f"{self._openpi_host}:{self._openpi_port}"
            )
            self.get_logger().info(
                "[OpenVLA] First OpenPi response can take 30-120s on slow/remote GPUs; "
                "keep Execute on until you see 'OpenPi returned ...' (not a WebSocket ping timeout)."
            )

    @staticmethod
    def _image_msg_to_numpy(msg: Image) -> np.ndarray:
        """Convert sensor_msgs/Image to numpy RGB array."""
        if msg.encoding in ("rgb8", "RGB8"):
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
        elif msg.encoding in ("bgr8", "BGR8"):
            bgr = np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, 3
            )
            return bgr[:, :, ::-1].copy()
        else:
            return np.frombuffer(msg.data, dtype=np.uint8).reshape(
                msg.height, msg.width, -1
            )[:, :, :3]

    def _preflight_check(self) -> bool:
        """Validate observation/action contracts before first inference request."""
        if not self.latest_joint_pos:
            if not self._warned_no_joint_state:
                self._warned_no_joint_state = True
                self.get_logger().warn(
                    "[OpenVLA] Preflight: no /joint_states samples yet; waiting."
                )
            return False

        if self._strict_joint_name_check:
            missing = [n for n in SOARM_JOINT_NAMES if n not in self.latest_joint_pos]
            if missing:
                if not self._warned_joint_name_mismatch:
                    self._warned_joint_name_mismatch = True
                    self.get_logger().warn(
                        f"[OpenVLA] Preflight: missing expected joint names: {missing}. "
                        "Blocking inference until joint name/order contract matches SOARM."
                    )
                return False

        if self._require_wrist_image and self.latest_wrist_image is None:
            if not self._warned_missing_wrist:
                self._warned_missing_wrist = True
                self.get_logger().warn(
                    "[OpenVLA] Preflight: waiting for /camera/wrist/image_raw."
                )
            return False

        if self._require_overhead_image and self.latest_overhead_image is None:
            if not self._warned_missing_overhead:
                self._warned_missing_overhead = True
                self.get_logger().warn(
                    "[OpenVLA] Preflight: waiting for /camera/overhead/image_raw."
                )
            return False

        if not self._preflight_ok:
            self._preflight_ok = True
            self.get_logger().info(
                "[OpenVLA] Preflight OK: joint/image contracts satisfied; starting inference."
            )
        return True

    # -- Inference --

    def _inference_loop(self):
        """Background thread: continuously query OpenPi for action chunks."""
        import time

        while rclpy.ok():
            if not self._inference_active:
                time.sleep(0.1)
                continue

            if self.client is None:
                time.sleep(0.1)
                continue

            if not self._preflight_check():
                time.sleep(0.1)
                continue

            # Only request new actions when queue is running low
            with self._lock:
                qlen = len(self.action_queue)
            if qlen > self.chunk_size // 2:
                time.sleep(0.02)
                continue

            obs = build_observation(
                joint_positions=self.latest_joint_pos,
                wrist_image=self.latest_wrist_image,
                overhead_image=self.latest_overhead_image,
                prompt=self.current_prompt,
                policy_mode=self.policy_mode,
            )

            self._debug_inference_count += 1
            self.get_logger().info(
                f"[OpenVLA] Requesting inference #{self._debug_inference_count} from "
                f"{self._openpi_host}:{self._openpi_port} (queue size before: {qlen})"
            )

            self._infer_in_flight = True
            try:
                t0 = time.perf_counter()
                result = self.client.infer(obs)
                elapsed = time.perf_counter() - t0
                actions = result.get("actions", [])
                try:
                    actions_len = len(actions)
                except TypeError:
                    actions_len = 0 if actions is None else 1

                if not self._inference_active:
                    self.get_logger().info(
                        f"[OpenVLA] OpenPi returned after {elapsed:.1f}s but inference was disabled "
                        f"while waiting — discarding {actions_len} action(s)."
                    )
                else:
                    with self._lock:
                        self.action_queue.extend(actions)
                        qsize_after = len(self.action_queue)
                    self.get_logger().info(
                        f"[OpenVLA] OpenPi returned {actions_len} action(s) in {elapsed:.1f}s; "
                        f"queue size now: {qsize_after}"
                    )
                    if actions_len == 0 and self._debug_inference_count <= 10:
                        self.get_logger().warn(
                            "[OpenVLA] OpenPi returned 0 actions (empty chunk). Robot will not move."
                        )
            except Exception as e:
                self.get_logger().warn(
                    f"[OpenVLA] Inference failed (connection or server error): {type(e).__name__}: {e}"
                )
                time.sleep(0.5)
            finally:
                self._infer_in_flight = False
                if self._infer_interval_sec > 0:
                    time.sleep(self._infer_interval_sec)

    # -- Control --

    def _control_loop(self):
        """Timer callback: pop one action from the queue and publish."""
        import time as _time

        if not self._inference_active:
            return

        with self._lock:
            if not self.action_queue:
                now = _time.time()
                if now - self._debug_last_empty_log_time >= 2.0:
                    self._debug_last_empty_log_time = now
                    if self._infer_in_flight:
                        self.get_logger().info(
                            "[OpenVLA] Control loop: queue empty — OpenPi inference still running "
                            "(blocked until the server responds; first chunk is often slow on remote/Jetson)."
                        )
                    else:
                        self.get_logger().info(
                            "[OpenVLA] Control loop: inference enabled but action queue empty "
                            "(waiting for OpenPi to return actions; check inference logs above for connection errors)"
                        )
                return
            action = self.action_queue.popleft()

        if isinstance(action, (list, np.ndarray)):
            action = np.asarray(action, dtype=np.float64)
        else:
            return

        self._debug_control_publish_count += 1
        if self._debug_control_publish_count <= 3 or self._debug_control_publish_count % 50 == 0:
            self.get_logger().info(
                f"[OpenVLA] Publishing joint command #{self._debug_control_publish_count} to /joint_commands"
            )

        msg = JointState()
        msg.name = list(SOARM_JOINT_NAMES)
        action_vec = action.tolist()
        if self.policy_mode == POLICY_MODE_SOARM:
            if len(action_vec) < len(SOARM_JOINT_NAMES):
                self.get_logger().warn(
                    f"[OpenVLA] SOARM mode expected >=6 action dims, got {len(action_vec)}; dropping action."
                )
                return
            msg.position = action_vec[: len(SOARM_JOINT_NAMES)]
        elif self.policy_mode == POLICY_MODE_DROID:
            # DROID policy outputs 8 dims: 7 arm joint targets + 1 gripper target.
            # SOARM has 5 arm joints + 1 gripper; we map:
            # - SOARM arm joints  <- DROID joint_position[0:5]
            # - SOARM gripper      <- DROID gripper_position (index 7)
            if len(action_vec) < 8:
                self.get_logger().warn(
                    f"[OpenVLA] DROID mode expected >=8 action dims, got {len(action_vec)}; dropping action."
                )
                return
            msg.position = [
                action_vec[0],
                action_vec[1],
                action_vec[2],
                action_vec[3],
                action_vec[4],
                action_vec[7],
            ]
        else:
            # Defensive fallback: should not happen because mode is validated at startup.
            msg.position = action_vec[: len(SOARM_JOINT_NAMES)]

        # Use NaN for unused modes so the Isaac node picks position control.
        nan_list = [float("nan")] * len(SOARM_JOINT_NAMES)
        msg.velocity = nan_list
        msg.effort = nan_list

        self.action_pub.publish(msg)


def main(args=None):
    rclpy.init(args=args)
    node = VLABridgeNode()

    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
