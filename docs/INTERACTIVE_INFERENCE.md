# Interactive VLA Inference Test

Interactively test pi0 or other VLA models against the SO-ARM101 robot arm
in Isaac Sim. Send natural language commands, watch the robot execute them,
spawn objects into the scene, and evaluate model behavior -- all streamed to
your machine via WebRTC.

---

## Prerequisites

| Requirement | Details |
|---|---|
| GPU | NVIDIA RTX series with NVENC (RTX 3070+). A100 is **not** supported (no NVENC). |
| Docker | Docker Engine 26+ with `nvidia-container-toolkit` installed |
| Robot USD | Run `./scripts/setup_robot_usd.sh` if you haven't already |
| WebRTC Client | Download the **Isaac Sim WebRTC Streaming Client** from [NVIDIA](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) (AppImage on Linux, .exe on Windows, .dmg on macOS) |
| Trained model (optional) | A LoRA checkpoint in `models/soarm_lora/`. Without one, the base pi0-FAST model is used. |

### Ports

The following ports must be open on the machine running Isaac Sim:

| Port | Protocol | Purpose |
|---|---|---|
| 49100 | TCP | WebRTC signaling |
| 47998 | UDP | WebRTC media stream |

---

## Quick Start

```bash
# Launch everything (Isaac Sim + OpenPi server + ROS2 bridge)
./scripts/interactive_test.sh
```

Wait for the log line:

```
[INFO] Interactive inference environment ready.
```

Then open the Isaac Sim WebRTC Streaming Client, enter `127.0.0.1` (or the
server IP if remote), and click **Connect**. You will see the simulation
viewport with the SO-ARM101 robot and a **VLA Interactive Test** control panel.

---

## Usage Examples

### Example 1: Basic Reach Command

1. In the control panel's **Command** field, type:
   ```
   move the robot arm to the right
   ```
2. Click **Execute**.
3. The robot begins moving. The status shows **EXECUTING (30s remaining)**.
4. When the motion looks complete, click **STOP**.
5. The robot freezes in place.

### Example 2: Pick and Place with a Spawned Object

1. Expand the **Spawn Objects** section in the control panel.
2. Set **Type** to `Cube`, **Size** to `0.03`, position to `X=0.15  Y=0.0  Z=0.015`.
3. Click **Spawn**. A small red cube appears on the table surface.
4. In the **Command** field, type:
   ```
   pick up the red cube
   ```
5. Click **Execute** and watch the robot attempt the grasp.
6. Click **STOP** when finished (or let the auto-timeout handle it).

### Example 3: Cluttered Scene Test

1. Expand the **Scene** section in the control panel.
2. Select the **Cluttered Table** preset from the dropdown.
3. The scene populates with several objects of different shapes and colors.
4. Type a command:
   ```
   push the blue cube to the left
   ```
5. Click **Execute**.

### Example 4: Emergency Stop

At any time during execution, click the red **E-STOP** button. The robot
immediately halts and all joint velocities are zeroed. This works regardless
of the current state.

---

## Command-Line Options

```bash
./scripts/interactive_test.sh [OPTIONS]
```

| Option | Default | Description |
|---|---|---|
| `--public-ip <IP>` | Auto-detected | Public IP for WebRTC. Set this when connecting from a different machine. |
| `--model-config <config>` | `soarm_pi0_fast` | OpenPi policy config name (e.g. `soarm_pi0`, `soarm_pi0_fast`). |
| `--checkpoint <path>` | `models/soarm_lora` | Path to a LoRA checkpoint directory. Omit to use the base model. |

### Examples

```bash
# Connect from another machine on the LAN
./scripts/interactive_test.sh --public-ip 192.168.1.50

# Use the slower but more capable pi0 model
./scripts/interactive_test.sh --model-config soarm_pi0

# Test a specific fine-tuned checkpoint
./scripts/interactive_test.sh --checkpoint /data/checkpoints/run_42
```

---

## Control Panel Reference

The **VLA Interactive Test** panel is docked in the Isaac Sim GUI and visible
through the WebRTC stream. It has four sections:

### Command (always visible)

| Element | Description |
|---|---|
| **Status** | Current state: `IDLE`, `EXECUTING (Ns remaining)`, `STOPPED`, or `E-STOPPED` |
| **Command** | Text field for the natural language instruction |
| **Execute** | Start inference with the current command. Disabled while already executing. |
| **STOP** | Gracefully stop inference. The robot holds its current position. |
| **E-STOP** | Immediately halt the robot and zero all velocities. |
| **Auto-stop (sec)** | Timeout in seconds (5--300). Execution stops automatically after this duration. Default: 30. |

### Telemetry (collapsible, expanded by default)

Live readout of all six joint positions in radians and the end-effector
world position in meters, updated at ~20 Hz.

### Spawn Objects (collapsible, collapsed by default)

| Element | Description |
|---|---|
| **Type** | `Cube`, `Sphere`, or `Cylinder` |
| **Size (m)** | Edge length / diameter of the object |
| **Position X/Y/Z** | World-frame spawn coordinates. Defaults are within the robot workspace. |
| **Color R/G/B** | RGB values in the 0--1 range |
| **Spawn** | Add the object to the scene with rigid-body physics |
| **Clear All** | Remove all previously spawned objects |

### Scene (collapsible, collapsed by default)

| Element | Description |
|---|---|
| **Reset Robot** | Return all joints to the home (zero) position |
| **Reset Scene** | Clear all spawned objects and reset the robot |
| **Preset** | Load a predefined scene: `Empty`, `Table with Cube`, or `Cluttered Table` |

---

## Architecture

Three Docker containers run under the `interactive` profile:

```
User Machine                         Docker Host (GPU)
┌─────────────────┐          ┌──────────────────────────────────┐
│ WebRTC Streaming │  WebRTC  │  isaac-sim                       │
│ Client (native)  │◄────────►│  SOARM 101 + Cameras + UI Panel  │
└─────────────────┘          │  OmniGraph ROS2 Bridge            │
                             └──────────┬───────────────────────┘
                                        │ ROS2 topics
                             ┌──────────┴───────────────────────┐
                             │  ros2-bridge                      │
                             │  vla_bridge_node.py               │
                             │  openpi-client (WebSocket)        │
                             └──────────┬───────────────────────┘
                                        │ WebSocket :8000
                             ┌──────────┴───────────────────────┐
                             │  openpi-server                    │
                             │  pi0 / pi0-FAST model             │
                             └──────────────────────────────────┘
```

### ROS2 Topics

| Topic | Type | Direction | Purpose |
|---|---|---|---|
| `/joint_states` | `sensor_msgs/JointState` | Isaac Sim -> Bridge | Current joint positions and velocities |
| `/camera/wrist/image_raw` | `sensor_msgs/Image` | Isaac Sim -> Bridge | 224x224 wrist camera feed |
| `/camera/overhead/image_raw` | `sensor_msgs/Image` | Isaac Sim -> Bridge | 224x224 overhead camera feed |
| `/joint_commands` | `sensor_msgs/JointState` | Bridge -> Isaac Sim | Joint position targets from the VLA |
| `/vla/prompt` | `std_msgs/String` | Isaac Sim -> Bridge | Natural language instruction |
| `/vla/enabled` | `std_msgs/Bool` | Isaac Sim -> Bridge | `True` = run inference, `False` = stop and flush |

### Camera Configuration

Both cameras match the LeRobot / ALOHA / OpenPi standard:

| Parameter | Value |
|---|---|
| Resolution | 224 x 224 |
| Frame rate | 30 Hz |
| FOV | ~60 degrees |
| Focal length | 1.93 mm |
| Horizontal aperture | 2.65 mm |

The wrist camera is attached to `gripper_frame_link` with offset `(-0.03, 0.05, -0.09)`.
The overhead camera is authored with world transform `(0.1, 0.0, 0.8)` and Euler `(0, -20, 0)`.

---

## Changing the VLA Model

The `--model-config` flag selects which OpenPi policy is served. Available
configs are defined in `training/configs/soarm_config.py`:

| Config | Model | Speed | Notes |
|---|---|---|---|
| `soarm_pi0_fast` | pi0-FAST | ~10 Hz | Autoregressive, faster inference. Default. |
| `soarm_pi0` | pi0 | ~2 Hz | Flow-based diffusion, higher quality but slower. |

To use a model from a different framework, replace the `openpi-server`
container with your own inference server that speaks the same WebSocket
protocol (see `ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/observation_builder.py`
for the observation format).

---

## Troubleshooting

### WebRTC client shows a blank screen

Isaac Sim takes 1--3 minutes to fully load on the first run. Wait for the
`Interactive inference environment ready` log message before connecting.

### "Connection failed" in the WebRTC client

- Verify that TCP port 49100 and UDP port 47998 are open.
- If connecting remotely, pass `--public-ip` with the correct address.
- Only one client can connect at a time.

### Robot doesn't move after clicking Execute

- Check that the OpenPi server is running: `docker logs soarm-openpi-server`.
- Check that the ROS2 bridge is running: `docker logs soarm-ros2-bridge`.
- Verify the model checkpoint exists in `models/soarm_lora/` (or omit
  `--checkpoint` to use the base model).

### A100 / H100 GPU

Livestreaming requires NVENC hardware which is absent on A100/H100. Run Isaac
Sim on an RTX workstation and point the OpenPi server at a remote A100 for
inference using the `eval-remote` profile instead (see `docs/REMOTE_DEPLOYMENT.md`).

### Containers won't start

```bash
# Rebuild all images
cd docker && docker compose build

# Check GPU access
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```
