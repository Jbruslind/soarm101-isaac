# Troubleshooting

Common issues and their solutions.

---

## Docker / GPU

### "docker: Error response from daemon: unknown or invalid runtime name: nvidia"

Docker does not have the NVIDIA container runtime configured. Install and configure the NVIDIA Container Toolkit, then restart Docker:

```bash
# Add repo (Ubuntu/Debian; adjust for your distro: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### "docker: Error response from daemon: could not select device driver"

NVIDIA Container Toolkit is not installed or configured. See the "unknown or invalid runtime name: nvidia" section above for full install steps, or if the toolkit is already installed:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
docker run --rm --gpus all nvidia/cuda:12.4.0-base-ubuntu22.04 nvidia-smi
```

### "unauthorized: authentication required" when pulling Isaac Sim

You need NGC credentials:

```bash
docker login nvcr.io
# Username: $oauthtoken
# Password: your NGC API key from https://ngc.nvidia.com/setup/api-key
```

### Out of GPU memory (OOM)

The 3080 Ti has 12 GB.  You cannot run multiple GPU-heavy containers
simultaneously.  Use docker-compose profiles:

```bash
# WRONG: starts everything (OOM)
docker compose up

# RIGHT: start only what you need
docker compose --profile collect up      # data collection only
docker compose --profile train up        # training only
docker compose --profile eval-local up   # evaluation only
```

Or offload inference/training to a remote GPU (Mode B).

### Isaac Sim is extremely slow on first run

Isaac Sim compiles shaders on first launch.  This can take 10-20 minutes.
The compiled shaders are cached in Docker volumes (`isaac-cache-*`), so
subsequent launches are much faster.

---

## Isaac Sim

### "USD file not found" error

The URDF-to-USD conversion hasn't been run yet:

```bash
./scripts/setup_robot_usd.sh
```

If automated conversion fails, convert manually in the Isaac Sim GUI.

### "Could not create directory ... for asset" when importing URDF

The URDF importer always writes to a path derived from the URDF: for `soarm101_isaacsim.urdf` it creates `robot_description/urdf/soarm101_isaacsim/` and writes `soarm101_isaacsim.usd` there. You cannot change this destination in the GUI.

Do this:

1. **Create that directory on the host** (and use a read-write mount):
   ```bash
   ./scripts/prepare_urdf_import_dir.sh
   ```
   Start the container with a read-write mount (no `:ro`):
   ```bash
   -v "$(pwd)/robot_description:/robot_description"
   ```

2. **Import the URDF** in Isaac Sim (File > Import > soarm101_isaacsim.urdf). The file will be written to `robot_description/urdf/soarm101_isaacsim/soarm101_isaacsim.usd`.

3. **Copy to the path the project expects:**
   ```bash
   ./scripts/copy_urdf_import_to_usd.sh
   ```
   This copies the file to `robot_description/usd/soarm101.usd`.

   If you still see **"Insufficient permissions to write to destination directory"**, either:

   - **Run the container as your user** (recommended): use `./scripts/run_isaac_sim_gui.sh`, which starts the container with `--user $(id -u):$(id -g)` so files written in the container match your host user and no `chmod` is needed.
   - Or re-run `./scripts/prepare_urdf_import_dir.sh` (makes the importer dir world-writable), or run `chmod -R a+rwX robot_description`, then try the import again.

### collect_sim_data only shows Isaac Sim startup, then seems to stop

After "app ready" and RTX/scenedb messages, the script loads the SO-ARM101 environment (USD + physics); that can take **1–2 minutes** on the first run. You should then see:

- `SimulationApp started. Loading environment (this can take 1–2 min on first run)...`
- `Environment ready. Collecting episodes...`
- `Episode 1/50: ...` (and so on)

If nothing after "app ready" appears for several minutes:

1. **Ensure the USD exists**: the collector needs `robot_description/usd/soarm101.usd`. Run `./scripts/setup_robot_usd.sh` (or the manual URDF import + `copy_urdf_import_to_usd.sh`) first.
2. **Unbuffered output**: the script sets `PYTHONUNBUFFERED=1` so episode lines appear as they run. If you run the Python script by hand inside the container, use `PYTHONUNBUFFERED=1` as well.

### "chmod: Operation not permitted" on .sh files in the container

When you run the container as your user (`./scripts/run_isaac_sim_gui.sh`), files inside the image (e.g. `/isaac-sim/runheadless.sh`) are owned by root, so you cannot change their permissions. Run the scripts with `bash` instead of `./` so the execute bit is not required:

```bash
bash runheadless.sh -v
bash python.sh -c "print('hello')"
```

### Robot flies off or explodes in simulation

Check actuator stiffness/damping values in `soarm_reach_env.py`.  The
defaults (stiffness=80, damping=4) work for the STS3215 servos.  If using
a different robot, adjust these values.

Also ensure `max_depenetration_velocity` is set (default: 5.0) to prevent
objects from being ejected by collision resolution.

### Camera images are black

1. Ensure the camera is attached to the correct link in the URDF/USD
2. Check that lighting exists in the scene (`DomeLightCfg`)
3. Verify `use_camera=True` is set in the environment config

### Viewing camera feeds in the interactive test (GUI)

To see what each camera sees so you can adjust their position/orientation:

1. Run the interactive test and connect with the WebRTC Streaming Client:
   ```bash
   ./scripts/interactive_test.sh
   ```
2. In the **viewport** (the 3D view), find the **video/camera icon** at the **top of the viewport** (toolbar).
3. Click it and choose a camera from the menu:
   - **overhead_cam** — under `World > Cameras > overhead_cam`
   - **wrist_cam** — under `World > Robot > so101_new_calib > gripper_frame_link > wrist_cam`
4. The viewport will then show that camera’s view. Move the robot or scene and adjust the camera’s **Transform** (Translate/Orient) in the **Property** panel (select the camera prim in the Stage) to refine placement.
5. To view another camera, click the video icon again and pick a different one. You can also open **Window > Viewport > Viewport 2** to have a second viewport and assign a different camera to each.

---

## OpenPi

### "Connection refused" when connecting to OpenPi server

1. Check that the OpenPi server container is running:
   ```bash
   docker ps | grep openpi
   ```

2. Check the server logs:
   ```bash
   docker logs soarm-openpi-server
   ```

3. Verify the port is correct:
   ```bash
   curl http://localhost:8000
   ```

4. If using remote inference, check `OPENPI_HOST` and `OPENPI_PORT` in
   your `.env` file.

### "Model checkpoint not found"

The training hasn't been run yet, or the checkpoint path is wrong:

```bash
ls models/soarm_lora/
# Should contain checkpoint files

# Check the environment variable
echo $OPENPI_CHECKPOINT_DIR
```

### OpenPi server crashes with OOM

The base model doesn't fit in VRAM.  Solutions:
- Use pi0-FAST instead of pi0 (smaller)
- Ensure no other GPU containers are running
- Use remote inference (Mode B) on a larger GPU

---

## Training

### Training is extremely slow on 3080 Ti

Expected behavior with QLoRA + CPU offloading.  The 3080 Ti is a
development/prototyping GPU, not a training GPU.  For faster training:

1. Reduce dataset size for debugging (`--episodes 10`)
2. Reduce `max_steps` to 1000 for smoke tests
3. Use remote training on a cloud A100

### "bitsandbytes" import error

The CUDA version inside the container might not match.  Rebuild:

```bash
cd docker
docker compose build training
```

### Loss is NaN

1. Check that normalization stats exist:
   ```bash
   cat data/episodes/meta/stats.json
   ```

2. Ensure `std` values are not zero (add epsilon if needed)

3. Reduce learning rate

---

## ROS2

### "ROS2 topics not visible between containers"

Both containers must share the same ROS2 domain:

```bash
# Both must have the same ROS_DOMAIN_ID
echo $ROS_DOMAIN_ID  # should be 42 (set in .env)
```

All containers use `network_mode: host`, which avoids Docker network
isolation issues for ROS2 DDS.

### "colcon build failed"

Check that all ROS2 dependencies are installed in the container:

```bash
# Inside the ros2-bridge container
apt list --installed | grep ros-humble
pip list | grep openpi
```

### Joint names don't match

The URDF joint names must match the names expected by the VLA bridge.
Expected order:

```
shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper
```

Verify with:
```bash
grep 'type="revolute"' robot_description/urdf/soarm101.urdf
```

---

## Remote Deployment

### No robot movement with remote Jetson (or remote OpenPi)

When you click **Execute** in the interactive test but the robot does not move, the problem is somewhere on: **Isaac Sim → ROS2 bridge → network → Jetson OpenPi → back to bridge → /joint_commands → Sim**.

**Step 1 – Run debug on the Jetson**

SSH into the Jetson. If you have the full repo there (e.g. you cloned it), run:

```bash
cd ~/soarm   # or path to your repo
./scripts/debug_jetson.sh
./scripts/debug_jetson.sh --logs 80   # more OpenPi log lines
```

If you deployed with `deploy_jetson.sh` only, the Jetson has `~/soarm/docker/` but usually not `scripts/`. Run these manually instead:

```bash
cd ~/soarm/docker
docker compose -f docker-compose.jetson.yml ps -a
docker compose -f docker-compose.jetson.yml logs --tail=50 openpi-server
docker compose -f docker-compose.jetson.yml logs --tail=15 caddy
# Ports (optional):
ss -tln | grep -E '8000|8443'
```

Check:

- Containers `soarm-openpi-jetson` and `soarm-caddy` are running.
- Ports 8000 and 8443 are listening.
- OpenPi logs show no repeated errors or OOM; you should see the server listening (e.g. port 8000).
- Caddy logs show no proxy errors.

If the OpenPi container keeps restarting or shows "checkpoint not found", fix the model path and config (see "Model checkpoint not found" above). On Jetson the container name is `soarm-openpi-jetson`:

```bash
docker logs soarm-openpi-jetson --tail 100
```

**Step 2 – Check connectivity from the client**

On the machine where you run `interactive_test.sh` (your PC or Sim host):

```bash
./scripts/debug_jetson.sh --host <JETSON_IP>
# If you use port 8000 instead of 8443:
./scripts/debug_jetson.sh --host <JETSON_IP> --port 8000
```

This checks that the Jetson’s port (8443 or 8000) is reachable. If TCP or HTTPS fails, fix firewall or network (e.g. allow 8443 from client to Jetson).

**Step 3 – Watch the ROS2 bridge logs**

Start the interactive test as usual:

```bash
./scripts/interactive_test.sh --remote <JETSON_IP>
# Or with direct OpenPi port: --remote <JETSON_IP> --port 8000
```

In **another terminal** on the same machine:

```bash
cd docker
docker compose --profile interactive logs -f ros2-bridge
```

Then click **Execute** in the Isaac Sim UI and watch the bridge output. You should see (in order):

1. `[OpenVLA] Inference ENABLED (Execute received)...`
2. `[OpenVLA] Requesting inference #1 from <host>:<port>...`
3. Either:
   - `[OpenVLA] OpenPi returned N action(s)...` and `[OpenVLA] Publishing joint command #1` → inference and commands are working; if the robot still does not move, the issue is between the bridge and Sim (e.g. ROS2 domain or `/joint_commands` not applied).
   - `[OpenVLA] Inference failed ... (connection or server error)` → connection or Jetson OpenPi problem; check Jetson logs and Step 1–2.
   - `[OpenVLA] Control loop: inference enabled but action queue empty` repeating → bridge never gets actions; usually connection or OpenPi not responding.

**"omni.isaac.ros2_bridge not available" or "ROS2 topics disabled" in Sim**

The interactive script enables the ROS2 bridge via `--enable isaacsim.ros2.bridge --enable omni.isaac.ros2.bridge` when starting Isaac Sim. If you still see these warnings:

- The Sim image may not include the ROS2 bridge extension. Use an Isaac Lab image that does (e.g. a variant built with `Dockerfile.ros2` if your project provides one), or ensure the NVIDIA Isaac Sim/Isaac Lab image you use lists the ROS2 bridge extension.
- Until the bridge is available in Sim, prompt/enabled use file-based signaling (`/tmp/vla_signals`), but `/joint_states` and `/joint_commands` will not be available, so the robot in Sim will not move from OpenVLA commands.

**Step 4 – Optional: confirm /joint_commands on the client**

If you have ROS2 on the host (e.g. in a container or install):

```bash
source /opt/ros/humble/setup.bash
ros2 topic echo /joint_commands --once
```

Run the interactive test, click Execute, and see if messages appear. Use the same `ROS_DOMAIN_ID` as the bridge (default 42 in docker `.env`).

**No requests reaching the Jetson (server logs show no incoming connections)**

If the OpenPi server on the Jetson is up (`server listening on 0.0.0.0:8000`) but you never see WebSocket or inference activity when you click Execute:

1. **Start the interactive test with the Jetson as remote** (on the machine where you run Isaac Sim):
   ```bash
   ./scripts/interactive_test.sh --remote <JETSON_IP>
   ```
   Use the Jetson’s real IP or hostname (e.g. `192.168.1.50` or `jbruslind-xavier` only if it resolves from the client). If you use **port 8000** (direct to OpenPi) instead of 8443 (Caddy):
   ```bash
   ./scripts/interactive_test.sh --remote <JETSON_IP> --port 8000
   ```
2. **Watch the ROS2 bridge logs** when you click Execute (same machine as above, second terminal):
   ```bash
   cd docker
   docker compose --profile interactive logs -f ros2-bridge
   ```
   You should see `[OpenVLA] Inference ENABLED`, then `[OpenVLA] Requesting inference #1 from <JETSON_IP>:<port>`. If you see `Inference failed` or `TCP connect ... failed`, the bridge cannot reach the Jetson (firewall, wrong IP/port, or Caddy not running when using 8443).
3. **If using port 8443**, Caddy must be running on the Jetson and listening on 8443. On the Jetson: `docker compose -f docker-compose.jetson.yml ps` should show both `soarm-openpi-jetson` and `soarm-caddy` running.
4. **Test connectivity** from the client to the Jetson:
   ```bash
   ./scripts/debug_jetson.sh --host <JETSON_IP>
   ./scripts/debug_jetson.sh --host <JETSON_IP> --port 8000   # if using direct OpenPi
   ```

**Jetson-specific notes**

- Default deployment uses **port 8443** (Caddy) for wss. Use `--remote JETSON_IP` (no `--port`) so the bridge uses 8443. If you expose OpenPi directly on 8000, use `--port 8000`.
- Container name on Jetson is **soarm-openpi-jetson** (not `soarm-openpi-server`). Use it in `docker logs` and `docker compose -f docker-compose.jetson.yml logs`.

### Can't connect to remote OpenPi server

1. Check firewall rules allow port 8443:
   ```bash
   ssh user@gpu-server "sudo ufw status"
   ```

2. Verify Caddy is running:
   ```bash
   ssh user@gpu-server "docker logs soarm-caddy"
   ```

3. Test the endpoint:
   ```bash
   curl -k https://gpu-server:8443
   ```

### rsync is slow

For large datasets, use compression and partial transfers:

```bash
rsync -avz --compress --partial --progress data/episodes/ user@host:~/soarm/data/episodes/
```

Or use S3/GCS for cloud storage.

---

## Data

### No Parquet files generated

`pyarrow` must be installed.  It's included in the Docker containers but
not on the host.  Run data collection inside the Isaac Sim container:

```bash
./scripts/collect_sim_data.sh --episodes 5
```

### Videos not generated

`av` (PyAV) must be installed and `--camera` flag must be passed:

```bash
./scripts/collect_sim_data.sh --env reach --episodes 5 --camera
```

---

## Getting Help

1. Check NVIDIA Isaac Sim docs: https://docs.isaacsim.omniverse.nvidia.com
2. Check OpenPi repo: https://github.com/Physical-Intelligence/openpi
3. Check LeRobot docs: https://huggingface.co/docs/lerobot
4. Check SO-ARM100 repo: https://github.com/TheRobotStudio/SO-ARM100
