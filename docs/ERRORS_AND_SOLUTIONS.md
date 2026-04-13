# Errors and Fixes (SO-ARM101 Isaac Sim + ROS2 + OpenPi)

This document records the major failures encountered while building the SO-ARM101 demo (Isaac Sim → ROS2 → OpenPi/VLA → joint commands) and the corresponding fixes applied.

---

## 1) OmniGraph port connection failed (`swhFrameNumber`)

### Symptom / Log
`OmniGraphError: Failed to connect /World/ROS2Bridge/tick.outputs:frame -> /World/ROS2Bridge/sim_time.inputs:swhFrameNumber`

### Root cause
In Isaac Sim 5.x, `isaacsim.core.nodes.IsaacReadSimulationTime` no longer exposes `inputs:swhFrameNumber` (the DROID/“fabric frame” style input was removed from the node schema).

### Fix
Updated graph wiring in [`isaac_envs/interactive_inference.py`](/home/jbruslind/Documents/isaac-sim_soarm101/isaac_envs/interactive_inference.py) to:
- remove the invalid connection `tick.outputs:frame -> sim_time.inputs:swhFrameNumber`
- keep the valid connection `sim_time.outputs:simulationTime -> pub_joint.inputs:timeStamp`

---

## 2) PhysX tensor device mismatch (`expected device 0, received device -1`)

### Symptom / Log
```
[omni.physx.tensors.plugin] Incompatible device of DOF position tensor in function getDofAttribute:
expected device 0, received device -1
[Ros2JointStateMessage] Failed to get dof positions
... velocities ...
... efforts ...
```

### Root cause
An incompatibility between the Isaac Lab articulation tensor pipeline and the ROS2 OmniGraph joint-state publisher path that reads DOF tensors via PhysX/legacy tensor access.

In practice, the ROS2 path was effectively trying to read GPU device tensors while the active tensor backend for the articulation was CPU/unowned (`device -1`).

### Fix
Forced Isaac Lab to run with CPU tensors in the interactive script:
- `SimulationCfg(..., device="cpu")`

Applied in [`isaac_envs/interactive_inference.py`](/home/jbruslind/Documents/isaac-sim_soarm101/isaac_envs/interactive_inference.py) at `main()`:
- `SimulationCfg(dt=..., render_interval=2)` → `SimulationCfg(dt=..., render_interval=2, device="cpu")`

### Notes
This is the recommended workaround for the interactive demo (GPU-free tensor path for ROS2 joint publishing).

---

## 3) ROS2 bridge shows “inference enabled but action queue empty”

### Symptom / Log (bridge)
`[OpenVLA] Control loop: inference enabled but action queue empty (waiting for OpenPi to return actions...)`

### Root cause (Phase A: OmniGraph event trigger)
A switch to physics-step triggers produced Isaac warnings like:
`Physics OnSimulationStep node detected in a non on-demand Graph. Node will only trigger events if the parent Graph is set to compute on-demand.`

When the trigger doesn’t fire, OmniGraph ROS2 publishers never publish `/joint_states` or camera frames, leaving the bridge with no observations.

### Fix (Phase A)
Restored playback-tick triggering and explicitly started playback in standalone mode:
- reverted trigger nodes to `omni.graph.action.OnPlaybackTick`
- added timeline start so playback-driven nodes actually pulse:
  - `omni.timeline.get_timeline_interface().play()`

Applied in [`isaac_envs/interactive_inference.py`](/home/jbruslind/Documents/isaac-sim_soarm101/isaac_envs/interactive_inference.py).

### Root cause (Phase B: `/joint_states` still not producing samples)
Even when `/joint_states` endpoints existed, samples were not arriving reliably at the bridge (publisher count present but no messages).

### Fix (Phase B: deterministic fallback)
Added a direct rclpy-based `/joint_states` publisher fallback that publishes joint state from the Isaac Lab `Articulation` buffers.

Applied in [`isaac_envs/interactive_inference.py`](/home/jbruslind/Documents/isaac-sim_soarm101/isaac_envs/interactive_inference.py):
- publish `/joint_states` from `robot.data.joint_pos` / `robot.data.joint_vel`
- publish during the sim loop (telemetry cadence)

This ensures the bridge always has `latest_joint_pos` to request inference.

---

## 4) ROS2 bridge had QoS mismatches (Reliable publisher vs BestEffort subscriber)

### Symptom / Evidence
- `/joint_states` publisher reported **RELIABLE**
- `vla_bridge` subscribed with **BEST_EFFORT**

Even if discovery succeeded, this configuration frequently led to “no samples delivered” behavior in this mixed container setup.

### Fix
Updated subscriber QoS:
- changed `/joint_states` subscription to `RELIABLE`
- kept camera subscriptions as `BEST_EFFORT`

Applied in [`ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/vla_bridge_node.py`](/home/jbruslind/Documents/isaac-sim_soarm101/ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/vla_bridge_node.py).

---

## 5) OpenPi server crashed for DROID fallback: missing observation key

### Symptom / Log (Jetson/OpenPi)
`KeyError: 'observation/gripper_position'`

### Root cause
Jetson OpenPi was serving DROID policy logic (`droid_policy.py`), which expects LeRobot-style DROID observation keys:
- `observation/joint_position`
- `observation/gripper_position`
- `observation/wrist_image_left`
- `observation/exterior_image_1_left`
- `prompt`

The bridge initially sent SOARM-style keys (`state`, `images`, `prompt`), so the policy transform failed.

### Fix
Added a DROID-compatible observation adapter:
- updated [`ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/observation_builder.py`](/home/jbruslind/Documents/isaac-sim_soarm101/ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/observation_builder.py)
- mapped SOARM inputs into the required DROID keys (including padding SOARM’s 5 arm joints → DROID’s 7 joint_position dims)
- populated the required image keys from wrist/overhead captures (or zeros if missing)

Also updated action mapping for DROID outputs:
- DROID policy outputs 8 dims (7 arm + 1 gripper)
- mapped those into SOARM’s 6 joint trajectory positions (5 arm + gripper)

Applied in [`ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/vla_bridge_node.py`](/home/jbruslind/Documents/isaac-sim_soarm101/ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/vla_bridge_node.py).

---

## 6) WebSocket failures during inference (`1011 keepalive ping timeout`)

### Symptom / Log (bridge)
`ConnectionClosedError: sent 1011 (internal error) keepalive ping timeout; no close frame received`

### Root cause
When DROID fallback was slow and/or the server-side handler took longer than websocket keepalive timeouts, connections were being aborted mid-request.

### Fix
Implemented a custom websocket client inside the bridge with longer ping/pong thresholds.

Applied in [`ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/vla_bridge_node.py`](/home/jbruslind/Documents/isaac-sim_soarm101/ros2_ws/src/soarm_vla_bridge/soarm_vla_bridge/vla_bridge_node.py):
- replaced default `WebsocketClientPolicy` with `_OpenPiWebsocketClient`
- set `ping_interval` / `ping_timeout` via env vars:
  - `OPENPI_WS_PING_INTERVAL_SEC` (default 60)
  - `OPENPI_WS_PING_TIMEOUT_SEC` (default 180)

Optionally supports slower demo inference via:
- `OPENPI_INFER_INTERVAL_SEC`

---

## 7) Jetson OpenPi config name invalid (`soarm_pi0_fast` not found)

### Symptom / Log (Jetson/OpenPi)
`ValueError: Config 'soarm_pi0_fast' not found. Did you mean 'pi0_fast_droid'?`

### Root cause
OpenPi’s built-in config registry doesn’t contain `soarm_pi0_fast`. When SOARM checkpoints aren’t available, Jetson must use an OpenPi built-in config name that exists.

### Fix
Updated Jetson defaults and deployment-generated `.env` to use a valid built-in DROID config name:
- `OPENPI_POLICY_CONFIG=${OPENPI_POLICY_CONFIG:-pi0_fast_droid}`

Applied in:
- [`docker/docker-compose.jetson.yml`](/home/jbruslind/Documents/isaac-sim_soarm101/docker/docker-compose.jetson.yml)
- [`scripts/deploy_jetson.sh`](/home/jbruslind/Documents/isaac-sim_soarm101/scripts/deploy_jetson.sh)

---

## 8) Jetson checkpoint directory missing (`models/soarm_lora` not present)

### Symptom / Evidence
- `ls -la ~/soarm/models/soarm_lora` returned: “No such file or directory”
- Jetson OpenPi then fell back to DROID default behavior.

### Root cause
There was no local SOARM checkpoint directory to deploy to Jetson.

### Fix (deployment)
Updated [`scripts/deploy_jetson.sh`](/home/jbruslind/Documents/isaac-sim_soarm101/scripts/deploy_jetson.sh) to sync `models/soarm_lora` when present.

### Practical note
If `models/soarm_lora` is not available locally, Jetson cannot serve a SOARM checkpoint, and DROID fallback is the demo path.

---

## 9) Cleanup: removed previous Cursor debug instrumentation

### What was removed
Removed `#region agent log f1832b` blocks and associated `/.cursor/debug-f1832b.log` NDJSON writers from the interactive sim script.

This kept production behavior intact while preserving the functional fixes.

---

## Appendix: Where to look for log signals

### Bridge-side
- `docker compose --profile interactive logs -f ros2-bridge`

You want to see transitions like:
- `Requesting inference #...`
- `OpenPi returned ... action(s)`
- `Publishing joint command #...`

### Jetson/OpenPi-side
- `docker logs -f soarm-openpi-jetson`

You want to see:
- it successfully serving a policy checkpoint (no config-not-found)
- it keeps websocket connections stable during inference

