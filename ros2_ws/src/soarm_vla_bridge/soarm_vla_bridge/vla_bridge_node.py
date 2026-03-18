"""ROS2 node that bridges robot observations to the OpenPi VLA policy server.

Subscribes to:
  /joint_states       (sensor_msgs/JointState)
  /camera/wrist/image_raw   (sensor_msgs/Image)
  /camera/overhead/image_raw (sensor_msgs/Image, optional)
  /vla/prompt         (std_msgs/String)
  /vla/enabled        (std_msgs/Bool)  -- gates inference on/off

Publishes:
  /joint_commands     (trajectory_msgs/JointTrajectory)

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
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint
from std_msgs.msg import Bool, String
from builtin_interfaces.msg import Duration

from soarm_vla_bridge.observation_builder import (
    build_observation,
    SOARM_JOINT_NAMES,
)

try:
    from openpi_client import websocket_client_policy
except ImportError:
    websocket_client_policy = None


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

        if websocket_client_policy is not None:
            self.get_logger().info(
                f"[OpenVLA] Connecting to OpenPi server at {host}:{port} "
                f"(OPENPI_HOST={os.environ.get('OPENPI_HOST', 'not set')}, "
                f"OPENPI_PORT={os.environ.get('OPENPI_PORT', 'not set')})"
            )
            self.client = websocket_client_policy.WebsocketClientPolicy(
                host=host, port=port
            )
            # Optional: try a quick TCP connect to see if remote is reachable
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
                    "(OpenPi may still use WebSocket on same port; inference will try anyway)"
                )
        else:
            self.get_logger().warn(
                "openpi-client not installed. Running in dry-run mode."
            )
            self.client = None

        # Subscribers
        qos = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)

        self.joint_sub = self.create_subscription(
            JointState, "/joint_states", self._joint_cb, qos
        )
        self.wrist_cam_sub = self.create_subscription(
            Image, "/camera/wrist/image_raw", self._wrist_image_cb, qos
        )
        self.overhead_cam_sub = self.create_subscription(
            Image, "/camera/overhead/image_raw", self._overhead_image_cb, qos
        )
        self.prompt_sub = self.create_subscription(
            String, "/vla/prompt", self._prompt_cb, 10
        )
        self.enabled_sub = self.create_subscription(
            Bool, "/vla/enabled", self._enabled_cb, 10
        )

        # Publisher
        self.action_pub = self.create_publisher(JointTrajectory, "/joint_commands", 10)

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

            if not self.latest_joint_pos:
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
            )

            self._debug_inference_count += 1
            if self._debug_inference_count <= 3 or self._debug_inference_count % 20 == 0:
                self.get_logger().info(
                    f"[OpenVLA] Requesting inference #{self._debug_inference_count} from "
                    f"{self._openpi_host}:{self._openpi_port} (queue size before: {qlen})"
                )

            try:
                result = self.client.infer(obs)
                actions = result.get("actions", [])
                with self._lock:
                    self.action_queue.extend(actions)
                    qsize_after = len(self.action_queue)
                if self._debug_inference_count <= 5 or (self._debug_inference_count % 20 == 0 and actions):
                    self.get_logger().info(
                        f"[OpenVLA] OpenPi returned {len(actions)} action(s); queue size now: {qsize_after}"
                    )
                if not actions and self._debug_inference_count <= 10:
                    self.get_logger().warn(
                        "[OpenVLA] OpenPi returned 0 actions (empty chunk). Robot will not move."
                    )
            except Exception as e:
                self.get_logger().warn(
                    f"[OpenVLA] Inference failed (connection or server error): {type(e).__name__}: {e}"
                )
                time.sleep(0.5)

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

        msg = JointTrajectory()
        msg.joint_names = SOARM_JOINT_NAMES

        point = JointTrajectoryPoint()
        point.positions = action.tolist()[:len(SOARM_JOINT_NAMES)]
        point.time_from_start = Duration(sec=0, nanosec=100_000_000)  # 100ms

        msg.points = [point]
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
