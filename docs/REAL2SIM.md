# Real-to-sim plan

This repo is part of a plan to use what was learned scanning with an Intel RealSense D455
in Isaac Sim. The full plan is in
[rgbd-sensor-kit docs/ARCHITECTURE.md](https://github.com/Jbruslind/rgbd-sensor-kit/blob/main/docs/ARCHITECTURE.md).

## What is coming

- **Scanned scenes as Isaac environments.** Rooms and objects reconstructed by
  [scan-twin](https://github.com/Jbruslind/scan-twin) load as scenes, with the SO-ARM101 placed on the scene's table.
- **Calibrated cameras.** A fixed overhead D455 (RGB-D, 848x480) whose depth noise and holes match the
  measured real camera, and a wrist RGB-only MIPI camera. A stereo camera may be added later.
- **Isaac Sim 6.1 migration.** The stack moves to Isaac Sim 6.1 and Isaac Lab 3.0.

## Role of this repo

This repo holds the Isaac glue (scene loader, camera factory, recording, ROS 2 publishing) and the
tutorials. Camera models, frame conventions, TUM formats, the noise model, metrics and the scene
contract live in the shared package `rgbd-sensor-kit` (import name `rgbdkit`).

## Status of the shared package

`rgbd-sensor-kit` is currently private, so links to it work for collaborators only. The public build
of this repo will not require it until it is opened; features that use it are behind a flag
(see I-5). It will be opened once licensing is settled.

## Work items

- [I-1: Isaac 6.1 prerequisites](https://github.com/Jbruslind/soarm101-isaac/issues/1)
- [I-2: Migrate to Isaac Sim 6.1 + Isaac Lab 3.0](https://github.com/Jbruslind/soarm101-isaac/issues/2)
- [I-3: Repo hygiene and CI](https://github.com/Jbruslind/soarm101-isaac/issues/3)
- [I-4: One robot and camera config module](https://github.com/Jbruslind/soarm101-isaac/issues/4)
- [I-5: Install rgbd-sensor-kit in the Isaac image](https://github.com/Jbruslind/soarm101-isaac/issues/5)
- [I-6: Scene loader](https://github.com/Jbruslind/soarm101-isaac/issues/6)
- [I-7: Place the robot in a scanned scene](https://github.com/Jbruslind/soarm101-isaac/issues/7)
- [I-8: Camera factory: overhead D455 + wrist RGB](https://github.com/Jbruslind/soarm101-isaac/issues/8)
- [I-9: Noisy depth in observations](https://github.com/Jbruslind/soarm101-isaac/issues/9)
- [I-10: Record depth and ground truth](https://github.com/Jbruslind/soarm101-isaac/issues/10)
- [I-11: ROS 2 depth, camera_info, TF and /clock](https://github.com/Jbruslind/soarm101-isaac/issues/11)
- [I-12: Tutorials](https://github.com/Jbruslind/soarm101-isaac/issues/12)
- [I-13: Close the scene loop](https://github.com/Jbruslind/soarm101-isaac/issues/13)
- [I-14: Wrist camera choice and profile (needs the user)](https://github.com/Jbruslind/soarm101-isaac/issues/14)
- [I-15: License audit before going public](https://github.com/Jbruslind/soarm101-isaac/issues/15)

The tutorial series is listed in [tutorials/README.md](tutorials/README.md). Progress is tracked in
[rgbd-sensor-kit#10](https://github.com/Jbruslind/rgbd-sensor-kit/issues/10).
