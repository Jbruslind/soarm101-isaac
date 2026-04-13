# Project Summary: VLA Robot Full Stack for SO-ARM101

A detailed summary of the project, changes, challenges, and solutions implemented to date.

---

## 1. Project Overview

### 1.1 Purpose

This repository provides a **Docker-based full stack** for training and deploying **Vision-Language-Action (VLA)** policies on the **SO-ARM101** 6-axis robot arm. The workflow spans:

- **Simulation** (NVIDIA Isaac Sim + Isaac Lab) for data collection and evaluation  
- **Training** (OpenPi / pi0-FAST) for fine-tuning VLA models on collected episodes  
- **Deployment** (ROS2 bridge + OpenPi server) for closed-loop control in simulation or on real hardware  

Data is stored in **LeRobot v3.0** format so it is compatible with HuggingFace and OpenPi tooling.

### 1.2 Key Technologies

| Component        | Technology |
|-----------------|------------|
| Simulation       | NVIDIA Isaac Sim 5.1, Isaac Lab 2.x |
| Robot model     | SO-ARM101 (TheRobotStudio/SO-ARM100), 6 DOF (5 arm + gripper) |
| VLA / policy    | OpenPi (pi0, pi0-FAST) |
| Dataset format  | LeRobot v3.0 (Parquet + MP4 + JSON) |
| Robot I/O       | ROS2 Humble (joint states, images, joint commands) |
| Orchestration   | Docker Compose with profiles |

### 1.3 High-Level Workflow

```text
1. Setup robot USD (URDF → USD) in Isaac Sim
2. Collect episodes in simulation (scripted or learned policy) → data/episodes/
3. Train VLA (LoRA/QLoRA) on episodes → models/
4. Evaluate in simulation: Isaac Sim + ROS2 bridge + OpenPi server
5. (Optional) Deploy to real SO-ARM101 via same ROS2 interface
```

---

## 2. Changes Implemented

### 2.1 WebRTC Live Viewing During Data Collection and Evaluation

**Goal:** Allow users to connect to the Isaac Sim container and watch the robot live via the Isaac Sim WebRTC Streaming Client while collecting data or evaluating a policy.

**Changes:**

- **`scripts/collect_sim_data.sh`**
  - Added `--watch` flag: sets `LIVESTREAM=2`, implies `--wait-for-key`, prints connection instructions.
  - Added `--public-ip <IP>` for remote clients; auto-detects host IP when `--watch` is used and IP is not provided.
  - Passes `LIVESTREAM=2` and `PUBLIC_IP` into the container via `EXTRA_ENV`.

- **`isaac_envs/sim_data_collector.py`**
  - When `LIVESTREAM` is set: uses `AppLauncher` with `enable_cameras=True`, creates the environment with `render_mode="human"` so the viewport renders the scene.
  - After `AppLauncher` init: sets `carb.settings.get_settings().set_bool("/app/livestream/allowResize", True)` so the WebRTC client can resize the stream (fixes “Stream can't be resized” / blank stream).
  - Wait message includes `PUBLIC_IP` for the WebRTC client.

- **`scripts/eval_sim.sh`**
  - Added `--watch` and `--public-ip`; sets `LIVESTREAM=2` and `PUBLIC_IP` when `--watch` is used; prints connection instructions.

- **`docs/QUICKSTART.md`**
  - Documented `--watch` and `--public-ip` for collection and evaluation.

**Result:** Users can run `./scripts/collect_sim_data.sh --watch` (or `eval_sim.sh --watch`), connect the WebRTC client to the shown IP, and see the robot during collection or evaluation. NVENC-capable GPU (e.g. RTX) required; A100 does not support NVENC.

---

### 2.2 Data Collection Pipeline Overhaul (IK Policy, Actions, Cameras, Language)

**Goal:** Replace random policy with goal-directed demonstrations, record actions and metadata in the format the VLA expects, and fix training/eval consistency.

**Changes:**

#### 2.2.1 Policy: Random → IK-Based Scripted Demonstrations

- **Before:** `action = torch.randn(1, cfg.action_space, device=env.device) * 0.3` — robot executed pure noise; every episode timed out; training learned to jerk randomly.
- **After:** Isaac Lab `DifferentialIKController` (damped least-squares, position command) drives the end-effector toward the randomly sampled target. Each step:
  - Reads current EE pose and target position (world frame).
  - Transforms to robot base frame via `subtract_frame_transforms`.
  - Calls `ik_controller.set_command(target_b, ee_quat=ee_quat_b)` and `ik_controller.compute(...)` to get arm joint targets.
  - Converts IK joint targets to delta actions for the env (env still uses delta commands internally).
  - Records **absolute** joint-position targets as the action in the dataset (see below).

#### 2.2.2 Action Format: Record Absolute Joint Positions

- **Before:** Delta actions were recorded as-is; at eval the VLA bridge sends **absolute** joint positions to `/joint_commands`, so train and eval formats did not match.
- **After:** The collector records **absolute target joint positions** (the IK solution) as the action for each frame. The VLA is trained to predict absolute positions, matching what the bridge publishes at inference time.

#### 2.2.3 Cameras Default On; Image Keys and Metadata

- **Before:** Cameras were opt-in via `--camera`; often unused, so no images were saved; training could not use vision.
- **After:**
  - Cameras are **on by default**; `--no-camera` disables them.
  - Both wrist and third-person cameras are used when cameras are enabled.
  - `LeRobotWriter.save()` adds **video features** to `info.json` for `observation.images.wrist` and `observation.images.third_person` (dtype, shape, video_info with fps/codec/pix_fmt), so the training loader knows about the videos.

#### 2.2.4 Language Instruction per Frame

- **Before:** No text conditioning in the episode data.
- **After:** Each frame includes a `language_instruction` field (e.g. `"move the robot arm to the green target"` for reach, `"pick up the red cube and place it at the green target"` for pick). Stored in Parquet and in `info.json` features.

#### 2.2.5 Environment: Expose Robot and Target for IK

- **`isaac_envs/soarm_reach_env.py`**
  - Added `robot` property (returns the articulation) and `target_pos` property (returns the current target position tensor).
  - Added constants `ARM_JOINT_NAMES` and `EE_BODY_NAME` for clarity.
  - Renamed deprecated actuator parameters: `effort_limit` → `effort_limit_sim`, `velocity_limit` → `velocity_limit_sim`.

#### 2.2.6 Collect Script and CLI

- **`scripts/collect_sim_data.sh`**
  - Cameras on by default; `--camera` replaced with `--no-camera` to opt out.
  - ENABLE_CAMERAS=1 set unless `--no-camera` is passed.

- **`isaac_envs/sim_data_collector.py`**
  - CLI: `--use-camera` replaced with `--no-camera`; cameras default on in `collect_episodes`.
  - Removed the broken `scripted_reach_policy()` that mapped Cartesian deltas to the first three joints.
  - AppLauncher is always used (needed for cameras and livestream); when livestream is off and cameras are off, cameras are still configured for consistency.
  - Per-episode log now reports whether the robot REACHED the target or final distance (e.g. `Episode 1/50: 85 steps (REACHED)`).
  - Stats: minimum std clamped to 1e-6 to avoid division-by-zero in normalization.

#### 2.2.7 OpenPi Training Config Alignment

- **`training/configs/soarm_config.py`**
  - **Image key:** `cam_overhead` now maps from `observation.images.third_person` (collector writes `third_person`, not `overhead`).
  - **Actions:** `use_delta_joint_actions` remains `False`; actions are absolute joint positions.
  - **Prompt:** `default_prompt` set to `"move the robot arm to the green target"` to match the language in the data.
  - Docstring updated to state that actions are absolute joint position targets.

#### 2.2.8 Documentation

- **`docs/QUICKSTART.md`**
  - Step 2 rewritten: data collection described as IK-based scripted demonstrations; cameras on by default; note to delete old random-policy data before re-collecting; example output shows “REACHED” or distance.

---

## 3. Challenges and Solutions

### 3.1 Blank WebRTC Stream

**Symptom:** After “Environment ready” and “Press Enter to start collection…”, the WebRTC client connected but the stream was blank.

**Causes:**

1. **`LIVESTREAM=2` not passed to the container** — Without it, AppLauncher loads a headless kit without WebRTC extensions.
2. **Wrong experience file** — Headless + no livestream → `isaaclab.python.headless.kit` (or headless.rendering.kit), which has no streaming. With `LIVESTREAM=2`, the correct kit (with `omni.kit.livestream.webrtc`) is loaded.
3. **Viewport not rendering** — Environment was created without `render_mode="human"`, so nothing was drawn for the stream.
4. **Stream resize rejected** — AppLauncher hardcodes `--/app/livestream/allowResize=false` for LIVESTREAM=2, so the client’s resize request failed and the stream stayed blank.

**Solutions:**

- In the collect (and eval) scripts: when user passes `--watch`, set `LIVESTREAM=2` and `PUBLIC_IP` and pass them into the container; imply `--wait-for-key` and print WebRTC connection steps.
- In the data collector: when `LIVESTREAM` is set, use AppLauncher with `enable_cameras=True`, create the env with `render_mode="human"`, and after AppLauncher init call `carb.settings.get_settings().set_bool("/app/livestream/allowResize", True)`.

---

### 3.2 Robot “Jerking” / No Coherent Motion After Training

**Symptom:** Robot moved randomly during data collection and after many training iterations still had no coherent motion.

**Root cause:** The data collection policy was **pure Gaussian noise** (`torch.randn(...) * 0.3`). Collected data had action mean ≈ 0 and std ≈ 0.3; no camera images; every episode timed out at 150 steps. Training a VLA on this data taught the policy to output similar noise.

**Solutions:**

- Replace the random policy with an **IK-based scripted policy** (DifferentialIKController) so demonstrations are smooth and goal-directed.
- Record **absolute** joint-position targets as actions so the VLA learns the same representation used at eval (absolute positions on `/joint_commands`).
- Enable cameras by default and fix metadata so training can load wrist and third-person videos and language instructions.

---

### 3.3 ArticulationData API: No `body_pose_w` / `root_pose_w`

**Symptom:** `AttributeError: 'ArticulationData' object has no attribute 'body_pose_w'. Did you mean: 'body_pos_w'?`

**Cause:** The Isaac Lab version bundled with Isaac Sim 5.1 does not expose combined `body_pose_w` or `root_pose_w`. The tutorial sometimes shows these, but in this stack pose is available as separate position and quaternion or via the state buffers.

**Solution:** Use the state buffers and slice them:

- **End-effector pose:** `robot.data.body_state_w[:, ee_body_id]` → position `[:, 0:3]`, quaternion `[:, 3:7]`.
- **Root pose:** `robot.data.root_state_w` → position `[:, 0:3]`, quaternion `[:, 3:7]`.

Pass these into `subtract_frame_transforms` and into the IK controller. Use `body_state_w[:, ee_body_id, 0:3]` for the final EE position when computing end-of-episode distance.

---

### 3.4 DifferentialIKController: `ee_quat` Required for Position-Only Command

**Symptom:** `ValueError: End-effector orientation can not be None for 'position_*' command type!` when calling `ik_controller.set_command(target_b)`.

**Cause:** For `command_type="position"`, the controller still expects the current end-effector orientation so it can store a full desired pose internally (position from command, orientation from current EE).

**Solution:** Pass the current EE orientation when setting the command:  
`ik_controller.set_command(target_b, ee_quat=ee_quat_b)`. The IK computation still uses only the position part of the Jacobian; the quaternion is used for internal bookkeeping/display.

---

### 3.5 WebRTC “Stream can't be resized” Warning

**Symptom:** Log message: `[carb.livestream-rtc.plugin] onClientResizeRequested: Stream can't be resized, please make sure that /app/livestream/allowResize is set to true.`

**Cause:** Default for livestream is `allowResize=false` when using AppLauncher with LIVESTREAM=2.

**Solution:** Set `allowResize` to true immediately after creating the app:  
`carb.settings.get_settings().set_bool("/app/livestream/allowResize", True)`. If this is done before the client connects, the stream can resize and display correctly.

---

### 3.6 Actuator Deprecation Warnings

**Symptom:** Warnings about `effort_limit` and `velocity_limit` being removed in favor of `effort_limit_sim` and `velocity_limit_sim`.

**Solution:** In `isaac_envs/soarm_reach_env.py`, in the robot’s `ImplicitActuatorCfg` entries, use `effort_limit_sim` and `velocity_limit_sim` instead of the deprecated names. SoarmPickEnv inherits from SoarmReachEnv, so it picks up the same config.

---

### 3.7 Image Key Mismatch (Training vs Data)

**Symptom:** Training config expected `observation.images.overhead` for the second camera, but the collector wrote `observation.images.third_person`.

**Solution:** In `training/configs/soarm_config.py`, RepackTransform maps `cam_overhead` from `observation.images.third_person` so the loader reads the same key the collector writes. Naming kept as `third_person` in the collector to reflect the camera’s role.

---

## 4. Current State and File Map

### 4.1 Data Collection

- **Entrypoint:** `./scripts/collect_sim_data.sh [--env reach|pick] [--episodes N] [--no-camera] [--watch] [--public-ip IP]`
- **Python:** `isaac_envs/sim_data_collector.py` — IK-based policy, LeRobotWriter with video + language_instruction, absolute actions, body_state_w/root_state_w for IK.
- **Env:** `isaac_envs/soarm_reach_env.py` (robot/target_pos exposed, actuator params updated), `isaac_envs/soarm_pick_env.py` (unchanged, inherits reach).

### 4.2 Training

- **Config:** `training/configs/soarm_config.py` — third_person image key, absolute actions, default_prompt aligned with data.
- **Entrypoint:** `./scripts/train.sh` (see docs/OPENPI_TRAINING.md).

### 4.3 Evaluation

- **Entrypoint:** `./scripts/eval_sim.sh [--watch] [--public-ip IP] [--remote HOST]`
- Isaac Sim + ROS2 bridge + OpenPi server; bridge sends absolute joint positions to `/joint_commands`, matching trained action space.

### 4.4 Documentation Touched

- `docs/QUICKSTART.md` — Collection and eval steps updated for IK policy, cameras default, `--watch`, and data cleanup.
- This file: `docs/PROJECT_SUMMARY.md` — Project summary, changes, challenges, and solutions.

---

## 5. Recommendations

1. **Before each new collection run:** Delete existing data in `data/episodes/` if it was produced with the old random policy (`rm -rf data/episodes/*`).
2. **Episode count:** Start with 50–100 high-quality IK demonstrations; quality matters more than quantity for VLA fine-tuning.
3. **WebRTC:** Use an NVENC-capable GPU (e.g. RTX); A100 does not support NVENC. Only one WebRTC client per Isaac Sim instance.
4. **Remote client:** Use `--public-ip` with the host’s LAN IP when connecting from another machine; ensure firewall allows TCP 8011, TCP 49100, UDP 47998 (or the ports used by the Isaac Sim streaming stack).

---

## 6. References

- [QUICKSTART.md](QUICKSTART.md) — Step-by-step setup and first run  
- [DATA_PIPELINE.md](DATA_PIPELINE.md) — LeRobot v3.0 layout and schema  
- [ARCHITECTURE.md](ARCHITECTURE.md) — Containers, profiles, and data flow  
- [TROUBLESHOOTING.md](TROUBLESHOOTING.md) — Common errors and fixes  
- [Isaac Lab Differential IK tutorial](https://isaac-sim.github.io/IsaacLab/main/source/tutorials/05_controllers/run_diff_ik.html) — Controller usage and frames  
- [Isaac Sim WebRTC Streaming Client](https://docs.isaacsim.omniverse.nvidia.com/latest/installation/download.html) — Standalone client for live view  
