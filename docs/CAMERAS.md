# Cameras: real hardware and its Isaac Lab twin

How the SO-ARM101's two cameras are set up on the real robot and in Isaac Sim 6.1 / Isaac Lab 3.0,
so that a policy trained on simulated images sees the same kind of images on the real arm.

| Camera | Hardware | Mount | Streams | Status |
|---|---|---|---|---|
| `overhead` | Intel RealSense D455 | Fixed, ~1 m above the workspace, looking down | RGB + depth, 848x480 @ 30 | Intrinsics known; mount pose to measure (#17) |
| `wrist` | Logitech USB webcam (model TBC) | Child of `gripper_frame_link` | RGB, native mode (e.g. 640x480 @ 30, MJPG) | Hardware chosen (#14), not yet calibrated |

The robot computer is an NVIDIA **AGX Xavier**. A MIPI/CSI wrist camera was ruled out because the
AGX's 120-pin camera connector needs a $170+ adapter plus sensor bring-up work. A USB webcam
avoids both.

Work items: #8 (camera factory), #9 (noisy depth), #10 (recording), #14 (wrist camera),
#17 (calibration and overlay check), #18 (real-robot camera pipeline).

## The rule everything follows

**One camera profile, used in three places.** Each camera has one profile file in
[rgbd-sensor-kit](https://github.com/Jbruslind/rgbd-sensor-kit) (`rgbdkit`). The profile holds:
- the name and LeRobot key
- resolution and fps
- the calibrated K and distortion
- the undistorted `new_K`
- the mount: parent link plus a 4x4 transform in OpenCV axes
- the measured latency
- the domain-randomisation ranges

Three consumers read it:
1. **Isaac glue** (`isaac_envs/cameras.py`) builds the `CameraCfg` from it. The env code holds no
   camera numbers.
2. **Real capture** on the Xavier reads the same profile, locks the camera controls, and applies
   the same `process_image()`.
3. **Recording** (sim collector and real robot) writes the same feature keys,
   `observation.images.wrist` and `observation.images.overhead`, at the same size and fps,
   through the same LeRobot video encoder.

A test compares the sim and real feature schemas and fails if they differ.

## 1. Fix first: Isaac Lab 3.0 quaternion order

Isaac Lab 3.0 changed quaternions from `(w, x, y, z)` to `(x, y, z, w)`, and the old values fail
silently.
- `isaac_envs/soarm_reach_env.py` uses `rot=(1.0, 0.0, 0.0, 0.0)` as "identity" for its cameras.
  Under 3.0 that is a **180-degree rotation about X**.
- The overhead camera rotation in `isaac_envs/interactive_inference.py` is also in wxyz order.

Run Isaac Lab's `scripts/tools/find_quaternions.py` over the repo as part of the 6.1 migration (#2).
All frame and quaternion conversion goes through `rgbdkit.conventions`, which has round-trip tests.

## 2. Declaring cameras in Isaac Lab 3.0

- **Use `CameraCfg` only.** `TiledCamera` was folded into `Camera` in 3.0, and `TiledCameraCfg` is
  a deprecated alias. Cloned env cameras are batched into tiled renders automatically. NVIDIA's
  SO-101 workshop and LeIsaac still use `TiledCameraCfg` (written for 2.x), so port their code
  rather than copying it.
- **Declare cameras in the scene config, not by hand in `_setup_scene`.** Use `{ENV_REGEX_NS}` prim
  paths under the robot link. Isaac Lab then spawns one camera per env, handles timestamps, and
  auto-enables rendering. That replaces the current `_timestamp` / `_is_outdated` monkey-patches.
  The reference is the `Isaac-Stack-Cube-Franka-IK-Rel-Visuomotor` task, now under
  `isaaclab_tasks/contrib/stack/`.

  ```python
  wrist: CameraCfg = CameraCfg(
      prim_path="{ENV_REGEX_NS}/Robot/so101_new_calib/gripper_frame_link/wrist_cam",
      update_period=1 / 30,
      width=640, height=480,                      # the real capture mode, not 224x224
      data_types=["rgb"],
      spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
          intrinsic_matrix=profile.new_K.flatten().tolist(), width=640, height=480),
      offset=CameraCfg.OffsetCfg(pos=mount.pos, rot=mount.quat_xyzw, convention="ros"),
  )
  ```

  `convention="ros"` (x right, y down, z forward) is the OpenCV optical frame. A hand-eye
  calibration result can therefore be used as-is.
- **Exact intrinsics.**
  - `PinholeCameraCfg.from_intrinsic_matrix(...)` sets focal length, apertures and offsets.
    Runtime readback assumes square pixels and a centred principal point.
  - To keep fx != fy and an off-centre cx/cy exactly, use
    `distortion=OpenCvPinholeDistortionCfg(fx, fy, cx, cy, image_size, ...)` with
    `apply_lens_distortion=False`.
  - Intrinsics are fixed per run. In tiled renders all envs share the first camera's projection,
    so intrinsics can't be randomised per env.
- **Render at the real sensor's resolution and aspect ratio.** Do not render a square 224x224. The
  current `focal_length=1.93, horizontal_aperture=2.65` at 224x224 gives a square ~68-degree view.
  That matches neither a 4:3 webcam crop nor the D455.
- **Data types.** Use `["rgb"]` for the wrist and `["rgb", "distance_to_image_plane"]` for the D455.
  RealSense depth is z-depth, so use `distance_to_image_plane`, never `distance_to_camera`. Depth
  noise is added afterwards by `rgbdkit.noise` (#9).
- **Stale frames after reset.** Set `num_rerenders_on_reset` > 0 (the visuomotor task uses 3).
  Otherwise the first image after a reset is stale.
- **Enable cameras.** Pass `--enable_cameras` explicitly until the cameras live in the scene config.
- **Renderer.** Keep `IsaacRtxRendererCfg`, the Isaac Sim default. OVRTX and Newton Warp exist, but
  it is unconfirmed whether they work in our Isaac Sim + PhysX stack.
- **Reading images.**
  - For data collection, read `env.scene["wrist"].data.output["rgb"].torch` (uint8, `(N,H,W,3)`)
    directly.
  - If images go through the observation manager, use `mdp.image_rgb(normalize=False)`. Its default
    `normalize=True` subtracts the per-image mean, which a VLA should not get.
- **VRAM on a 10 GB RTX 3080.** Use `num_envs=1` for scripted collection, with exactly two cameras
  at native resolution. Drop the debug third-person camera from collection. Raise to 4-16 envs
  only after watching memory.

## 3. Lens distortion: undistort the real images

The wrist webcam has real lens distortion; the sim camera is a pinhole. Undistort the real
frames to the sim's pinhole model (`cv2.initUndistortRectifyMap` with
`getOptimalNewCameraMatrix(alpha=0)`) and give the sim **the new camera matrix `new_K`**, not the
raw K. That is the default because both sides then share one simple model.
- Undistortion is one precomputed remap per frame on the Xavier.
- It runs before video encoding, in collection and at inference alike.

Isaac Lab 3.0 can also render distortion natively (`OpenCvPinholeDistortionCfg`, added mid-2026).
Use it only for the overlay check in section 6. It is new, and its tests cover the OVRTX and Newton
renderers but not Isaac RTX. Before trusting it, confirm with an exaggerated `k1` that our renderer
honours it.

## 4. Matching the image pipelines

| Stage | Real (Xavier) | Sim (Isaac Lab) |
|---|---|---|
| Capture | Locked controls (section 5). Wrist: MJPG at native mode. D455: 848x480 | `CameraCfg` at the same mode, intrinsics = `new_K` |
| Geometry | Undistort to `new_K` | Already pinhole |
| Resize | **None.** Store native aspect | Same |
| Encoding | LeRobot video writer, same codec and CRF | Same writer, same settings |
| Policy input | OpenPi `resize_with_pad(224, 224)`, letterbox, inside the model transforms | Same |
| Timing | ~60-150 ms glass-to-glass (measure it, #18). Timestamp at capture | Zero latency. Add a 2-4 frame randomised observation delay |

- **Store native-aspect frames and let the model do the 224 step.** OpenPi letterboxes rather than
  squashes. If anything must resize earlier, it is one `to_policy_image()` in `rgbdkit`, called on
  both sides.
- **Encode both sides with the same writer.** Sim frames then get the same compression artefacts.
  Real frames also carry the webcam's own MJPG compression.
- **Rolling shutter and motion blur aren't simulated.** Use a short fixed exposure on the real
  webcam (with enough light), and keep wrist motion moderate in scripted collection.

## 5. The real wrist webcam (Logitech, USB)

**Which model.**
- **Best:** a fixed-focus model (C270 / C310 / C505 / Brio 100). Its optics can't move, so one
  calibration holds.
- **Autofocus models** (C920 / C922 / C930e / StreamCam): set `focus_automatic_continuous=0`, then
  a fixed `focus_absolute` tuned for 15-30 cm, and calibrate at that setting. Their voice-coil
  lens can still sag with wrist orientation, so check calibration in a few wrist poses.
- **Weight:**
  - A whole C920 weighs 162 g, far too heavy for the STS3215 wrist servo.
  - A whole C270-class camera is ~75 g.
  - The SO-100 community usually strips the housing and mounts the bare board. Weigh the result,
    and check that close objects (5-20 cm) are sharp enough at 224 px.
- **FOV:** don't trust spec sheets for the C270 class. It is sold as 55-60 degrees diagonal, but a
  published calibration implies ~51 degrees at 640x480. Calibrate the actual unit (#17).

**Lock the camera's settings.**
- LeRobot's `OpenCVCameraConfig` only sets size, fps and fourcc, so exposure, white balance and
  focus stay on auto unless something else sets them.
- Lock them from a systemd oneshot or a udev `RUN+=` rule.
- Re-apply after the stream opens, because some Logitech models reset controls on `STREAMON`.

```bash
DEV=/dev/v4l/by-id/usb-046d_<model>_<serial>-video-index0      # stable name, survives replug
v4l2-ctl -d $DEV --list-ctrls-menus                             # use the names this prints
v4l2-ctl -d $DEV -c auto_exposure=1,exposure_dynamic_framerate=0,exposure_time_absolute=80 \
                 -c white_balance_automatic=0,white_balance_temperature=4600 \
                 -c power_line_frequency=2,backlight_compensation=0     # 2 = 60 Hz mains
# autofocus models only:
v4l2-ctl -d $DEV -c focus_automatic_continuous=0 && v4l2-ctl -d $DEV -c focus_absolute=<N>
```

- `exposure_time_absolute` is in units of 100 µs.
- To avoid flicker, use multiples of 8.33 ms (60 Hz mains) or 10 ms (50 Hz). A shorter exposure
  reduces motion blur.
- Older kernels name the controls `exposure_auto`, `focus_auto`, and so on, so trust
  `--list-ctrls`.

**USB and the cable.**
- Use MJPG, since 720p YUYV doesn't fit USB 2.0 at 30 fps.
- Put the webcam and the D455 on different Xavier ports, with no hub, and check with `lsusb -t`.
- Webcam cables are stiff PVC and not rated for flexing. Use a thin, flexible USB 2.0 lead with
  strain relief at the camera, and service loops (> 15 mm bend radius) at wrist-flex and
  wrist-roll.
- Run the full joint range while streaming, and watch `dmesg` for USB resets.

**LeRobot config:**
`OpenCVCameraConfig(index_or_path=DEV, width=640, height=480, fps=30, fourcc="MJPG")`. Check that
your LeRobot version has `fourcc`.

## 6. Calibrate, then prove the twin matches (#17)

1. **Intrinsics.** Calibrate with ChArUco or a checkerboard, at the exact capture mode with focus
   and exposure locked. Use 20-40 views and aim for a reprojection error < 0.5 px. Save K,
   distortion, `new_K`, size, date and serial as a profile.
2. **Joint zeros.** Check that the sim URDF's joint zeros match the real arm's calibration. Hand-eye
   calibration is only as good as forward kinematics.
3. **Wrist mount.** Move the arm through 15+ varied poses with a board in view. Solve
   `cv2.calibrateHandEye` for camera to `gripper_frame_link`, in OpenCV axes. For an SO-101
   starting point see https://github.com/fireloop-ai/camera-calibration (Isaac Sim 5.x).
4. **Overhead D455.** Use the factory intrinsics (fx = fy = 427.789, cx = 425.743, cy = 238.633 at
   848x480) and measure its pose relative to the arm base.
5. **Overlay check.** Put the real arm in 3-5 configurations. Render the sim at the same joints and
   blend the two images. Landmarks (jaw tips, table edges) should line up within a few pixels. This
   one check validates intrinsics, mount, FK and conventions together.

## 7. Domain randomisation, in priority order

The published evidence is thin and mostly not wrist-specific, so this order is a best guess. Start
here, then measure.

1. **Lighting.** Dome-light intensity, colour temperature (~2500-9500 K) and HDRI.
   - This is the biggest lever in the wrist-camera studies we found.
   - In Isaac Lab's stack task, the dome-light randomiser only runs when `eval_mode` is set, so
     adapt it for training.
2. **Camera mount pose.** Jitter it: a few mm and ±1-2 degrees for the wrist, ±2 cm and ±3 degrees
   for the overhead. NVIDIA's SO-101 workshop has `randomize_camera_pose`, which needs porting to
   xyzw and to per-env sampling.
3. **Field of view.** About ±5%. Intrinsics can't vary per env in a tiled render, so vary them per
   run or per reset.
4. **Surfaces.** Table and object textures and distractor objects. The texture randomiser needs
   `replicate_physics=False`, which is fine at `num_envs=1`.
5. **Image and timing.** Noise, blur, the encoder round-trip, and the 2-4 frame observation delay.

**With a scanned room as the scene (scan-twin).**
- Treat the scan as geometry and layout. Keep randomising lighting and surfaces on top.
- Vertex colours carry baked-in capture lighting. They also look soft at wrist-camera range
  (10-30 cm).
- Gaussian-splat (NuRec) backgrounds are possible in Isaac Sim 6.x for photoreal RGB. They have no
  collision, and frame generation causes artefacts with them, so they're a later experiment.

## 8. Checklist

1. Fix quaternion order: `(1,0,0,0)` → `(0,0,0,1)`, and reorder every wxyz value (#2).
2. Use `CameraCfg` in the scene config with `{ENV_REGEX_NS}` paths. Remove the hand-built cameras
   and the monkey-patches (#4, #8).
3. Render at the real resolution and aspect, with calibrated `new_K`. No 224x224 renders (#8).
4. Set `num_rerenders_on_reset` > 0, and pass `--enable_cameras`.
5. Use `distance_to_image_plane` for depth, then apply `rgbdkit.noise` (#9).
6. Lock webcam focus, exposure and white balance in a startup service, re-applied after the stream
   opens (#18).
7. Undistort real wrist frames to `new_K`. Use rendered distortion only for the overlay check.
8. Calibrate intrinsics and hand-eye, then pass the overlay check at 3-5 poses (#17).
9. Keep camera names, sizes, fps and encoder settings identical, enforced by the schema test (#10,
   #18).
10. Leave the 224 resize to OpenPi's `resize_with_pad`. Use uint8 images with `normalize=False`.
11. Randomise lighting first, then mount pose, then FOV, then surfaces.
12. Add a 2-4 frame randomised observation delay matching the measured USB latency.
13. Use `num_envs=1` for scripted collection on the 3080, with two cameras only.

## Sources

- **Isaac Lab:** `release/3.0.0` and `develop` source and docs.
  - `camera_cfg.py`, `sensors_cfg.py`, `docs/source/concepts/sensors/camera.rst`.
  - Migration guide `migrating_to_isaaclab_3-0.rst`.
  - PRs #6608 (OpenCV lens distortion), #6851 and #7916.
  - The visuomotor stack task: https://github.com/isaac-sim/IsaacLab
- **NVIDIA Sim-to-Real SO-101 course and workshop:**
  https://docs.nvidia.com/learning/physical-ai/sim-to-real-so-101/latest/index.html and
  https://github.com/isaac-sim/Sim-to-Real-SO-101-Workshop
- **LeIsaac:** https://github.com/LightwheelAI/leisaac
- **OpenPi image transforms** (`resize_with_pad`):
  https://github.com/Physical-Intelligence/openpi
- **LeRobot camera and video configs:** https://github.com/huggingface/lerobot
- **Logitech technical specifications** (support.logi.com) and the Linux V4L2 camera-control docs.
- **Randomisation evidence:**
  - https://arxiv.org/abs/2307.15320
  - https://arxiv.org/html/2511.09932
  - https://arxiv.org/abs/2409.10161
