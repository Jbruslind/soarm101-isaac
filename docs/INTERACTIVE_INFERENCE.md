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
| FOV | 69 degrees horizontal |
| Focal length | 1.93 mm |
| Horizontal aperture | 2.65 mm |

Both cameras are the ones used for data collection, defined once in
`isaac_envs/soarm_common/contract.py` (`WRIST_CAMERA`, `OVERHEAD_CAMERA`): the wrist camera sits
beside the gripper under `gripper_frame_link` at `(0.02, 0.04, -0.07)` m aimed between the jaws;
the overhead camera is at world `(0.45, 0.40, 0.30)` looking at `(0.19, 0, 0.03)`. See
docs/CONFIG_ALIGNMENT.md. `--snapshot DIR --snapshot-joint-pos "q0,...,q5"` renders both cameras
at given joint poses and exits (used for the collector-vs-inference frame-diff check).

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

### Serving a fine-tuned LeRobot policy (ACT / SmolVLA)

A checkpoint from `scripts/train_lerobot.sh` (see [LEROBOT_TRAINING.md](LEROBOT_TRAINING.md))
is served by `training/scripts/lerobot_policy_server.py` on the same OpenPi websocket protocol.
The bridge, this scene and the Isaac Sim closed-loop client
(`tutorials/fasteners/part1/07_run_policy.py`, tutorial F7) use it unchanged:

```bash
./scripts/serve_lerobot.sh --checkpoint data/checkpoints/smolvla_fastener_m6_10k --detach
curl -s http://127.0.0.1:8000/healthz                      # OK once the policy is loaded
OPENPI_REQUIRE_OVERHEAD_IMAGE=1 ./scripts/interactive_test.sh --remote 127.0.0.1 --port 8000
```

- **Inputs.** Each request carries `state` (6 joint angles, rad), both camera images
  (`cam_wrist`, `cam_overhead`, HxWx3 uint8) and the `prompt`. The server maps them onto the
  checkpoint's LeRobot inputs through `soarm_common.contract`, and applies the normalization
  and camera renames saved with the checkpoint. These policies need **both** cameras, so set
  `OPENPI_REQUIRE_OVERHEAD_IMAGE=1` for the bridge. Then it waits for the overhead image
  instead of sending requests the server would reject.
- **Outputs and re-planning.** The reply is `{"actions": (n, 6)}`: absolute joint targets in
  rad. *n* (`--n-action-steps`, default 10) is how many steps the client runs before it asks
  again. F6 showed that re-planning every 10 steps tracks the expert 2.5x better than running a
  whole 100-step ACT chunk. A request can override it with an `"n_action_steps"` key.
- **Batching.** State `(N, 6)`, images `(N, H, W, 3)` and a list of prompts get `(N, n, 6)`
  back. The F7 client and the evaluation harness (#37) use this to query many envs in one call.
- **Which device.** `--device auto` (the default) uses the GPU when it has room for the
  policy next to whatever already runs there, Isaac Sim included, and otherwise the CPU.
  The policy uses about 0.9 GB (SmolVLA) or about 0.2 GB (ACT) of GPU memory once loaded, plus
  the CUDA context. `--device cpu` forces the CPU; a server on another host also works
  (`interactive_test.sh --remote HOST`).
- **Checking it.** `training/scripts/check_policy_server.py` plays a LeRobot dataset to a
  running server over the real protocol. It reports the action error against the recorded
  actions, the round-trip time, and a batched-request test:

  ```bash
  cd docker && docker compose --profile serve-lerobot run --rm -e NVIDIA_VISIBLE_DEVICES=void lerobot-server \
      python /training/scripts/check_policy_server.py --dataset /workspace/data/lerobot/fastener_m6_heldout
  ```

Reference numbers (12 GB GPU; the F6 held-out demos):

| | SmolVLA (10k steps) | ACT (10k steps) |
|---|---|---|
| Action error vs recorded, re-plan every 10 steps | 0.028–0.030 rad (same as F6) | 0.073 rad |
| Round trip per request, batch 1 (median) | 295 ms | 16 ms |
| Batched request of 4 | 387 ms | — |
| GPU memory with this interactive scene running | 1.2 GB server + 1.25 GB Isaac Sim = 2.7 GB in total | less |

**Keeping up live.** The bridge asks again once its queue is empty, and one request covers
*n* control steps (n / 30 s). If a round trip takes longer than that, the arm pauses between
chunks. SmolVLA at n = 10 is right at the limit: 295 ms per request against a 333 ms budget. On
a slower GPU or the CPU, raise `--n-action-steps` (25 gives 833 ms), or use `--flow-steps 5`
(SmolVLA only: about half the inference time). The F7 client is not affected: the simulation
waits for the server, so its results do not depend on speed.

### Testing without a model

`scripts/mock_policy_server.py` is a stand-in policy server that speaks the OpenPi
websocket protocol and returns a slow sinusoid of joint targets (`(10, 6)` chunks), so the
whole loop (cameras, ROS 2 graph, bridge, `/joint_commands`, arm motion) can be checked
without a GPU model. It logs each observation it receives, including per-camera image
mean/std (a mean near 0 or a std near 0 means a blank or featureless frame).

```bash
docker run --rm --network host -v "$PWD/scripts:/scripts:ro" \
    -v "$PWD/isaac_envs/soarm_common:/opt/soarm/soarm_common:ro" --entrypoint python3 \
    soarm-ros2 /scripts/mock_policy_server.py --port 8000
./scripts/interactive_test.sh --remote 127.0.0.1 --port 8000
```

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
