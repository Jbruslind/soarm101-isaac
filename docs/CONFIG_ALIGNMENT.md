# SOARM101 OpenPI Configuration Alignment

This guide prevents schema mismatches between Isaac Sim, ROS2 bridge, and OpenPI
policies. Most "robot moves back-and-forth" failures are configuration mismatches,
not raw motion-control issues.

## 0) One contract: `isaac_envs/soarm_common`

Robot, cameras, control rate, action semantics and every key between them are defined once,
in `isaac_envs/soarm_common/contract.py` (plain Python, stdlib only; print it with
`python3 isaac_envs/soarm_common/contract.py`). Isaac Lab configs are built from it by
`soarm_common/robot.py` (articulation + PhysX settings), `soarm_common/cameras.py` (camera
cfgs) and `soarm_common/scene.py` (ground + light).

| Consumer | How it gets the contract |
|---|---|
| `soarm_reach_env.py`, `soarm_pick_env.py`, `sim_data_collector.py`, `interactive_inference.py` | `from soarm_common import ...` (`/isaac_envs` is the script dir) |
| ROS 2 bridge (`soarm_vla_bridge`) | `soarm_vla_bridge/soarm_contract.py` imports it from `/opt/soarm` (compose mounts `../isaac_envs/soarm_common` there) or the repo checkout |
| phospho-bridge dashboard | same `/opt/soarm` mount |
| `training/configs/soarm_config.py` (openpi-server, training) | `/opt/soarm` mount locally; the deploy scripts rsync `soarm_common/` next to the configs on remote hosts |
| `scripts/mock_policy_server.py` | repo checkout, or `-v "$PWD/isaac_envs/soarm_common:/opt/soarm/soarm_common:ro"` |

The ROS package needs no build change: it is a pure-Python lookup at import time, and the
bridge fails with a clear `ImportError` if the mount is missing.

### Camera names and keys (one name per camera, end to end)

| Stage | Wrist camera | Overhead camera |
|---|---|---|
| USD prim | `<robot>/.../gripper_frame_link/wrist_cam` | `<env>/Cameras/overhead_cam` (`/World/Cameras/overhead_cam` in inference) |
| Env observation | `obs["wrist_rgb"]` | `obs["overhead_rgb"]` |
| LeRobot dataset | `observation.images.wrist` | `observation.images.overhead` (was `third_person`) |
| ROS 2 topic | `/camera/wrist/image_raw` | `/camera/overhead/image_raw` |
| Policy request | `images["cam_wrist"]` | `images["cam_overhead"]` |

Both: 224x224 RGB, focal length 1.93 mm, horizontal aperture 2.65 mm (69 deg HFOV), near clip
1 cm, sampled at 30 Hz.

- **Wrist mount:** under `gripper_frame_link` at (0.02, 0.04, -0.07) m, i.e. beside the gripper
  on its +y side, 7 cm back from the tips, aimed at (-0.01, 0, 0.01) (between the jaws, just past
  the tips), image up = gripper +x. It sees both jaws, the object in the jaws and the place
  target.
- **Overhead:** in front of the robot at (0.35, 0.035, 0.26), 60 deg above the table and 0.30 m
  from (0.20, 0.035, 0), looking back at the robot (image up = towards the base). It covers the
  fastener work cell of `soarm_tasks` (spawn area and bin) and the cube area of `SoarmPickEnv`.
  An M6x20 bolt is 8.5-13 px long (mean 11) at the pick pose; from the earlier three-quarter view
  at (0.45, 0.40, 0.30) it was 3.5-6 px (issue #32, `soarm_tasks/scripts/camera_check.py`).
  `CONTRACT_VERSION` went to 2 with this change: datasets record it in `meta/info.json`
  (`soarm_contract_version`), and overhead images recorded before it come from the old view. The
  collector, the envs and `interactive_inference.py` read the pose from the contract; the ROS
  bridge and policy server only see the images. The name "overhead" is kept for the ROS topic
  and the `cam_overhead` policy key.

### Timing and actions

- Physics 120 Hz, decimation 4: **control at 30 Hz** everywhere. The dataset is recorded at
  30 fps, the bridge `control_rate` defaults to `CONTROL_HZ` (was 10 Hz), and the mock policy
  server spaces its actions by 1/30 s.
- **Actions are absolute joint-position targets** (rad, joint order below, gripper included),
  one per control step, held until the next one: the dataset `action`, the policy-server
  `actions` chunk and ROS `/joint_commands` are all this.
- The Isaac Lab env (also usable for RL/PPO) keeps its normalised delta action:
  `target = q + clip(a, -1, 1) * 0.05 rad`. `env.joint_pos_target` exposes the absolute target
  it applied, and the collector records that (previously it recorded the raw IK goal, which the
  env only approached at 0.05 rad/step).
- A dataset frame is (state and images *before* the step, action applied in the step). The
  collector previously paired the action with the observation after the step (and, on the last
  frame, with the auto-reset state).

### Robot

Joint names/limits, the USD path, the EE body and the sim drives live in the contract.
`robot.check_joint_limits()` warns if the spawned USD disagrees. The real servo (Feetech
STS3215) stall torque, 3.35 N m (Isaac Lab SysID asset), is `REAL_SERVO_MAX_TORQUE_NM`; the sim
gripper effort limit now uses it (was 10 N m): the scripted pick still placed 20/20 with it.
The arm drives keep 30 N m.

### What was out of line before (GitHub issue #4)

| Item | Collector / env | Inference (`interactive_inference.py`) / bridge |
|---|---|---|
| Wrist camera | (0.05, 0, -0.07), 25 deg tilt | (-0.03, 0.05, -0.09), 32 deg tilt |
| Second camera | `third_person` at (0.45, 0.40, 0.30) (pick) / (0.8, 0.8, 0.5) (reach) | `overhead` at (0.1, 0, 0.8) looking down |
| Dataset key | `observation.images.third_person` | `cam_overhead` |
| Control rate | 30 Hz | bridge 10 Hz, mock 10 Hz |
| PhysX | `solve_articulation_contact_last` etc. | defaults |
| Actions | delta (env), IK goal (recorded) | absolute |

**Black wrist frames.** The old wrist mount sat 5 cm out on the gripper's +x side, which faces
the robot base in the top-down grasp. When the gripper worked within ~12 cm of the base, the
forearm folded down onto that spot and the camera ended up inside the arm mesh (fully black
frames during carry/lower; 350 of 1184 frames in 5 near-base picks). Moving it beside the
gripper (+y) takes it out of the arm's plane: 0 dark frames in 10 picks (5 near the base) and
in the 20-episode collection below.

**Frozen inference images.** `interactive_inference.py` relied on the `ENABLE_CAMERAS` env var,
which Isaac Lab 3.0 no longer reads, so headless runs got the non-rendering Kit experience
(`useFabricSceneDelegate=false`). The renderer then read USD, which PhysX doesn't update, and
the ROS camera topics showed the robot frozen at its USD pose while the joints moved. The
script now always sets `enable_cameras`.

### Verification

- Frame diff, collector env vs `interactive_inference.py --snapshot` at the same joint pose
  (3 collision-free poses, 30 render steps each): joint positions agree to 1e-6 rad; mean
  absolute pixel difference 1.0-1.8 (wrist) and 1.0-1.3 (overhead) out of 255, PSNR 35-39 dB.
  The residual is RTX denoiser noise and edge aliasing. Tolerance: mean |diff| < 3, PSNR > 30 dB.
  A pose pressing the gripper into the floor diverges (CPU vs GPU contact resolution), so the
  check uses free-space poses.
- `--env pick`, 20 episodes with cameras (gripper effort 3.35 N m): 20/20 placed, 0 dark wrist/overhead frames out of 8084. With 10 N m: 20/20, 0 dark of 7884.
- Reach: smoke test passes; 10 scripted episodes reach 9/10.
- Headless `interactive_inference.py` + bridge + mock server: both camera topics publish
  (~24 Hz each), `/joint_commands` at 30.0 Hz, camera images change as the arm moves.

### No duplicated constants

```bash
git grep -n -E '"shoulder_pan"|gripper_frame_link"|soarm101\.usda"|"cam_(wrist|overhead)"|"observation\.images\.|"/camera/(wrist|overhead)/image_raw"|"/joint_(states|commands)"|"/vla/(prompt|enabled)"|focal_length=|horizontal_aperture=|\b224\b|control_rate", *(10|30)|1 ?/ ?120|joint_effort_limit=[0-9]' -- '*.py' '*.yml' '*.sh' ':!data' ':!isaac_envs/soarm_common/contract.py'
```

Remaining hits are comments/docstrings and the USD file name in the asset build scripts.

## 1) Choose One Explicit Policy Mode

Do not mix modes or rely on implicit fallback.

| Mode | `OPENPI_POLICY_MODE` | Policy config | Checkpoint requirement | Observation contract | Action contract |
|---|---|---|---|---|---|
| SOARM-native | `soarm` | `soarm_*` (e.g. `soarm_pi0_fast`) | Required, local SOARM checkpoint dir | `state` (6D), `images.cam_wrist`, `images.cam_overhead` | 6D absolute SOARM joint targets |
| DROID-compat | `droid` | OpenPI DROID config (e.g. `pi0_fast_droid`) or `--env DROID` | Optional | `observation/joint_position` (7D padded), `observation/gripper_position` (1D), wrist/exterior images | 8D DROID action remapped to SOARM 6D |

### Fail-fast behavior now enforced

- `docker/openpi-server/entrypoint.sh` rejects:
  - `OPENPI_POLICY_MODE=soarm` with non-`soarm_*` config
  - `OPENPI_POLICY_MODE=soarm` without a valid checkpoint directory
  - `OPENPI_POLICY_MODE=droid` with `soarm_*` config
- There is no silent fallback from SOARM to DROID.

## 2) Launch Examples

### SOARM-native (recommended)

```bash
./scripts/interactive_test.sh \
  --policy-mode soarm \
  --model-config soarm_pi0_fast \
  --checkpoint /path/to/soarm_lora
```

### DROID-compat (fallback/demo)

```bash
./scripts/interactive_test.sh \
  --policy-mode droid \
  --model-config pi0_fast_droid
```

## 3) Preflight Checklist (Block Before Execute)

The bridge now validates the following before first inference:

1. `/joint_states` has samples.
2. Expected SOARM joint names exist.
3. Wrist image is present (`OPENPI_REQUIRE_WRIST_IMAGE=1` by default).
4. Overhead image present if required (`OPENPI_REQUIRE_OVERHEAD_IMAGE=1`).
5. Policy mode is valid (`soarm` or `droid`).

When checks pass, logs show:

`[OpenVLA] Preflight OK: joint/image contracts satisfied; starting inference.`

## 4) Camera and Joint Contract

Canonical SOARM joint order (bridge + observations), `soarm_common.contract.JOINT_NAMES`:

`shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`

Camera topics consumed by bridge (see section 0 for the full key table):

- `/camera/wrist/image_raw` (required by default)
- `/camera/overhead/image_raw` (optional by default)

## 5) Hardware and Deployment Recommendations

### Stable deployment hierarchy

1. **Best stability:** Isaac Sim + ROS2 bridge on RTX workstation, OpenPI on remote GPU.
2. **Single-machine dev:** RTX 12-16 GB VRAM class minimum for smooth interactive work.
3. **Jetson edge:** Use DROID mode unless you have validated SOARM checkpoint/config support.

### Practical sizing

- Inference-only practical floor: **8-12 GB VRAM**.
- Fine-tuning (LoRA/full): substantially larger VRAM (see OpenPI requirements).
- Isaac livestream host needs **NVENC** (use RTX host for visualization).

## 6) External References

- OpenPI repository and model docs: <https://github.com/Physical-Intelligence/openpi>
- Physical Intelligence research updates: <https://www.pi.website/>
- SOARM Isaac Lab project reference: <https://github.com/MuammerBay/isaac_so_arm101>
