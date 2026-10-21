# Isaac Sim / Isaac Lab versions

## Current stack

| Component | Version | Set by |
|-----------|---------|--------|
| Isaac Sim | **6.2.0.1** (`nvcr.io/nvidia/isaac-sim`, Ubuntu 24.04, Python 3.12) | `ISAACSIM_VERSION` in `docker/.env` |
| Isaac Lab | **3.0** (`release/3.0.0` branch, pinned commit) | `ISAACLAB_REF` in `docker/.env` |
| ROS 2 | Jazzy (bundled in `isaacsim.ros2.core`; system debs for native libs) | `docker/isaac-sim/Dockerfile` |

`docker/isaac-sim/Dockerfile` builds Isaac Lab from `ISAACLAB_REF` on top of the Isaac Sim
image, the same way Isaac Lab's own `docker/Dockerfile.base` does. The image is tagged
`soarm-isaac-sim:<ISAACSIM_VERSION>`.

Why not the prebuilt `nvcr.io/nvidia/isaac-lab:3.0.0-rc1` image: it bundles an Isaac Sim
**6.1** release candidate, which aborts on shutdown (`Destroying busy TaskGroup!`), so every
run exits non-zero. Isaac Lab's `release/3.0.0` branch now targets Isaac Sim 6.2.

### Known image quirk: pip breaks during the Isaac Lab install

The 6.2 image replaces 148 files in Kit's Python (124 of them pip's vendored packages)
with symlinks into `extscache/omni.kit.pip_archive-*/pip_prebundle/`. Isaac Lab's install
upgrades `urllib3`; pip uninstalls the prebundled 2.7.0 copy (first on `sys.path`), which
deletes the symlink targets, and every later `pip install` fails with
`No module named 'pip._vendor.urllib3.util.retry'` (surfacing as an `isaacteleop` install
error about an hour into the build). The Dockerfile replaces those symlinks with real files
before installing Isaac Lab. Keep that step when bumping the base image.

### Bumping

1. Pick an Isaac Sim tag and an Isaac Lab commit whose `docker/.env.base` names that
   Isaac Sim version.
2. Update `ISAACSIM_VERSION` / `ISAACLAB_REF` in `docker/.env` and `docker/.env.example`
   (and the defaults in `docker/docker-compose.yml` / the Dockerfile).
3. `cd docker && docker compose build isaac-sim`, then run `./scripts/play_reach.sh --gui`
   and a short `./scripts/collect_sim_data.sh` as smoke tests.

## NVIDIA driver

| Isaac Sim | Driver 580.x | Driver 595 / 610 / 615 |
|-----------|--------------|------------------------|
| 5.1 (Isaac Lab 2.x) | works (580.65.06 validated) | **crashes** in `librtx.scenedb.plugin.so` on the first rendered frame, GUI and headless |
| 6.2 | supported (>= 580.95.05) | works (tested on 615.71.09, RTX A2000 12 GB) |

See isaac-sim/IsaacSim#687 and #650 for the 5.1 crash reports.

## Isaac Lab 3.0 conventions used in this repo

Official guide: `docs/source/isaaclab/migrating.rst` in the Isaac Lab repo.

- **Quaternions are `(x, y, z, w)`**; identity is `(0, 0, 0, 1)`. USD (`Gf.Quatd`) is still
  `(w, x, y, z)`; `interactive_inference.py` converts with `_wxyz_to_xyzw()` where it hands
  a quaternion to Isaac Lab. Check new code with Isaac Lab's
  `scripts/tools/find_quaternions.py --path isaac_envs`, and at runtime with
  `WARN_ON_TORCH_QUATF_ACCESS=1`.
- **Asset and sensor data are `ProxyArray`**: use `.torch` (`robot.data.joint_pos.torch`,
  `camera.data.output["rgb"].torch`).
- **Writes are explicit**: `write_joint_state_to_sim_index(position=..., velocity=...)`,
  `write_root_pose_to_sim_index(root_pose=...)`; joint targets go through
  `robot.actuators.target_command.set_position_index(value=...)`.
- **Actuator limits**: `joint_velocity_limit` / `joint_effort_limit` (were
  `velocity_limit_sim` / `effort_limit_sim`).
- **PhysX schema cfgs** come from `isaaclab_physx.sim.schemas`
  (`PhysxRigidBodyPropertiesCfg`, `PhysxArticulationRootPropertiesCfg`,
  `PhysxCollisionPropertiesCfg`).
- **Viewport**: `ViewerCfg` is gone; the reach env sets
  `sim.default_visualizer_cfg = KitVisualizerCfg(origin_type="asset", origin_track_path="robot", ...)`.
  A window opens only when a Kit visualizer is selected (`--viz kit`). This repo's
  scripts select it automatically when `HEADLESS=0` (`scripts/isaac.sh --gui`).
- **Launch**: `add_launcher_args(parser)` + `with launch_simulation(cfg, args):`
  (`play_reach.py`, `convert_urdf.py`). `AppLauncher` still works but is deprecated.
- **RL**: `isaaclab.sh train|play --rl_library rsl_rl --task <Task>` (no `-v0` suffix).
- **URDF import**: importer 3.0 via `isaaclab.sim.converters.UrdfConverter`
  (`docker/isaac-sim/convert_urdf.py`). Its output prim layout differs from the 2.x
  importer that produced the committed `robot_description/usd/soarm101.usd`.

## Behavior fixes that came with the port

Testing on 6.2 surfaced three bugs that were already on `main` (Isaac Lab 2.x):

### Actions were ignored

On Isaac Lab 2.x, `SoarmReachEnv._apply_action` rebuilt the joint target from
`self.actions`, which the base class only ever sets to zeros. Every physics substep
therefore reset the target to the current joint positions, so the reach and pick envs
(and the IK data collector, which drives them through `env.step`) **ignored their
actions**. `_pre_physics_step` now computes the target once per policy step
(`action_scale` rad/step) and `_apply_action` holds it. Episodes collected with the old
code should be regenerated.

### Reward/success measured the moving jaw, not the gripper frame

The envs used `body_pos_w[:, -2]` as the end-effector. In the imported USD the bodies are
`[..., moving_jaw_so101_v1_link, gripper_frame_link]`, so index -2 is the moving jaw,
about 8 cm from `gripper_frame_link`, and the 2 cm success check never fired. The env now
resolves `gripper_frame_link` by name (`SoarmReachEnv.ee_pos_w`). The data collector also
measured its final distance after the env had auto-reset the robot; it now uses the last
pre-step distance. With both fixes the IK collector reaches every target (5/5 episodes,
31-63 steps).

### The third-person camera looked away from the robot

It sat at (0.8, 0.8, 0.5) with an identity rotation, which in the `world` convention looks
along +X, at an empty horizon. It now looks at the workspace point (0, 0, 0.2).

### The IsaacLab window showed "World" as the viewer follow mode

With `KitVisualizerCfg(origin_type="asset", origin_track_path="robot")` as the env's
`default_visualizer_cfg`, the camera did track the robot (`origin_track_path` is the scene
asset key, optionally `"robot/<body>"`), but the IsaacLab window's Viewer Settings > Follow Mode
dropdown said "World". Isaac Lab 3.0's `BaseEnvWindow` always builds that dropdown on its first
entry and never reads the visualizer config. The reach env now sets
`ui_window_class_type = SoarmEnvWindow`, which selects the configured entry ("Robot").
