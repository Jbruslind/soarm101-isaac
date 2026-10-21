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
  (`docker/isaac-sim/convert_urdf.py`, run by `scripts/setup_robot_usd.sh`). See
  "Robot USD" below.

## Robot USD (URDF importer 3.0)

The robot asset is `robot_description/usd/soarm101/soarm101.usda` plus `payloads/`. It
replaced `robot_description/usd/soarm101.usd`, a 2.x-importer (Isaac Sim 5.x) stage that
referenced `robot_description/urdf/soarm101_isaacsim/` (both removed). Layout changes:

| | Old (2.x importer) | New (importer 3.0) |
|---|---|---|
| Files | stage wrapper + 4 `configuration/*.usd` layers, 23.3 MB | root `.usda` + `payloads/` (7 `.usda`, `geometries.usd`), 5.8 MB |
| Default prim | `/World` (with a light and physics scene) holding `so101_new_calib` **and a duplicate `so101_new_calib_01`** | the robot, `so101_new_calib`; nothing else |
| Robot prim in the env | `/World/envs/env_0/Robot/so101_new_calib` | `/World/envs/env_0/Robot` |
| Links | flat: `so101_new_calib/<link>` | nested along the chain: `Robot/Geometry/base_link/shoulder_link/.../gripper_link/gripper_frame_link` |
| Joints | `so101_new_calib/joints/<joint>`, `root_joint` at the top | `Robot/Physics/<joint>` (incl. `root_joint`) |
| Articulation root | `root_joint` (fixed joint to world) | importer: `base_link` (floating base, see below); after `convert_urdf.py`: the robot prim |
| Physics layers | variants `Physics=PhysX`, `Sensor`, `Robot` | variant `Physics` = `physics` (neutral UsdPhysics, default) / `physx` / `mujoco` / `none` |
| Drive gains in the asset | stiffness 0.04-4.5, damping ~1e-4, max velocity 573 deg/s | none authored (the URDF has none); max force 10 |

Unchanged: body names and order, joint names and order, joint limits, masses, centres of
mass, inertia (principal axes ordered differently), 17 convex-hull colliders, and forward
kinematics (EE and jaw positions match to 0.1 mm at three test poses). The envs set the
drive gains, limits, self-collision and solver iterations themselves, so the PhysX
articulation they build reports the same values with either asset.

**Floating base fix.** As shipped, importer 3.0 puts `PhysicsArticulationRootAPI` on
`base_link` and fixes the base with a world joint (`Physics/root_joint`). PhysX then builds a
floating-base articulation held by an external constraint: Isaac Lab reports
`is_fixed_base == False`, the base moves about 2.4 mm, and the arm sags more under zero
action. `convert_urdf.py` moves the articulation root to the robot prim
(`--fix-articulation-root`, on by default), which is the same fix Isaac Lab applies when it
creates a root joint itself. With that fix the reach smoke test and the IK collector behave
like the old asset: 22/25 targets reached in 45.7 steps on average, against 23/25 in 44.1
steps with the old asset (random, unseeded targets; both miss a few by 2-9 cm).

`interactive_inference.py` loads the new asset and reaches "environment ready", but Kit logs
about 30 Hydra `_MarkInstancerDirty` coding-error warnings at startup that the old asset
did not trigger (the reach env and collector log none). They look cosmetic and have not been
investigated further.

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
