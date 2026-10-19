# Sunshine / Moonlight GUI Streaming

Run Isaac Sim / Isaac Lab **headless** when you want maximum throughput (training,
bulk data collection), or with a **real GUI window** when you want to watch the robot,
and stream that window to any [Moonlight](https://moonlight-stream.org/) client on
your LAN (PC, Mac, phone, tablet, TV) with low-latency NVENC video.

Compared to the Isaac Sim WebRTC livestream (`--watch`), this streams the regular Kit
window, works with any Moonlight client, survives client reconnects, and allows
several sims to run one after another without reconnecting.

## How it works

```
┌──────────────────────── Host (NVIDIA driver, Docker, nvidia-container-toolkit) ────────┐
│                                                                                        │
│  ┌─ soarm-sunshine (profile: gui, always-on) ─┐     ┌─ isaac-sim (run with --gui) ──┐  │
│  │  Xorg :0 on the NVIDIA GPU                 │     │  Isaac Sim / Isaac Lab         │  │
│  │    virtual monitor (fake EDID, DFP-0)      │◄────┤  DISPLAY=:0, HEADLESS=0        │  │
│  │  Openbox (windows open maximized)          │ X11 │  (docker-compose.gui.yml)      │  │
│  │  Sunshine: X11 capture → NVENC → network   │     └────────────────────────────────┘  │
│  └───────────┬────────────────────────────────┘                                        │
│              │ x11-unix volume (/tmp/.X11-unix) shared between the two containers      │
└──────────────┼─────────────────────────────────────────────────────────────────────────┘
               │ TCP 47984/47989/47990/48010, UDP 47998-48000
               ▼
        Moonlight client(s) on the LAN
```

- `docker/sunshine/` builds the stream host image (Ubuntu 24.04, Xorg, Openbox,
  PulseAudio, Sunshine `.deb`).
- `docker/docker-compose.gui.yml` is an overlay that gives `isaac-sim` the X socket
  and `DISPLAY` and turns headless/WebRTC off. The scripts apply it for you with `--gui`.
- Headless runs don't use the overlay at all, so they behave exactly as before.

## Host prerequisites (one time)

1. **NVIDIA driver** (≥ 580 recommended for Isaac Sim 5.1). Check: `nvidia-smi`.
2. **Docker + NVIDIA Container Toolkit**:
   `docker run --rm --gpus all ubuntu nvidia-smi` should print the GPU.
3. **uinput** for Moonlight keyboard/mouse:
   ```bash
   sudo modprobe uinput
   echo uinput | sudo tee /etc/modules-load.d/uinput.conf
   ```
4. **Firewall** (if `ufw` is enabled):
   ```bash
   sudo ufw allow 47984,47989,47990,48010/tcp
   sudo ufw allow 47998:48000/udp
   ```

5. **Restrict access to LAN/VPN clients** (recommended):
   ```bash
   sudo ./scripts/sunshine-firewall.sh install
   ```
   Sunshine has no source filter for its streaming ports, so this adds a dedicated
   nftables table (`inet soarm_sunshine`, reapplied at boot by
   `soarm-sunshine-fw.service`). It drops traffic to the Sunshine ports from any
   source outside `SUNSHINE_ALLOWED_SUBNETS` in `docker/.env` (your LAN and VPN
   subnets, e.g. `192.168.1.0/24,10.0.0.0/24`; required) and all IPv6 traffic to them. Other
   ports and existing ufw rules are untouched. Rerun `install` after changing the
   subnets; `status` / `remove` are also available.

No monitor or dummy plug is needed; the container fakes one. You don't need a
desktop environment on the host either.

## Start the stream host

```bash
./scripts/sunshine.sh up        # builds soarm-sunshine on first run
```

Then:

1. Open `https://<host-ip>:47990` (self-signed cert) and create the web UI login.
   Alternatively, set `SUNSHINE_USER` / `SUNSHINE_PASS` in `docker/.env` before `up`.
2. In Moonlight, add the host by IP (auto-discovery may also find **soarm-isaac**).
   Moonlight shows a PIN: enter it in the web UI under **PIN**.
3. Open **SO-ARM Isaac Desktop**. You should see an empty dark desktop.

The container restarts automatically (`restart: unless-stopped`), and pairing and
credentials persist in the `sunshine-config` volume.

## Launch Isaac Sim headless or with a GUI

| Task | Headless (fast) | GUI (watch in Moonlight) |
|------|-----------------|--------------------------|
| Any command | `./scripts/isaac.sh --headless <cmd>` | `./scripts/isaac.sh --gui <cmd>` |
| Isaac Sim editor | – | `./scripts/isaac.sh --gui` |
| Data collection | `./scripts/collect_sim_data.sh` | `./scripts/collect_sim_data.sh --gui` |
| Reach env playback | – | `./scripts/play_reach.sh --gui` |
| Closed-loop eval | `./scripts/eval_sim.sh` | `./scripts/eval_sim.sh --gui` |
| Interactive VLA | (WebRTC) `./scripts/interactive_test.sh` | `./scripts/interactive_test.sh --gui ...` |

Isaac Lab examples:

```bash
# RL training: headless for speed
./scripts/isaac.sh --headless /workspace/isaaclab/isaaclab.sh -p \
    /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/train.py --task Isaac-Reach-Franka-v0

# Same policy, played back with a window you can watch
./scripts/isaac.sh --gui /workspace/isaaclab/isaaclab.sh -p \
    /workspace/isaaclab/scripts/reinforcement_learning/rsl_rl/play.py --task Isaac-Reach-Franka-Play-v0 --num_envs 4
```

`isaac.sh` sets `HEADLESS=1` or `HEADLESS=0`, which Isaac Lab's `AppLauncher` honours.
Your own scripts behave the same way as long as they build `AppLauncher` from
`add_app_launcher_args`, or read `HEADLESS` like `sim_data_collector.py` does.

`--gui` fails with a clear message if the Sunshine container isn't running.

## Configuration (`docker/.env`)

| Variable | Default | Meaning |
|----------|---------|---------|
| `SOARM_DISPLAY` | `:0` | X display shared by Sunshine and Isaac Sim |
| `SUNSHINE_DISPLAY_WIDTH` / `_HEIGHT` / `_REFRESH` | `1920` / `1080` / `60` | Virtual monitor mode (e.g. 2560×1440, 3840×2160@60) |
| `SUNSHINE_NVIDIA_OUTPUT` | `DFP-0` | GPU connector to fake the monitor on |
| `SUNSHINE_NAME` | `soarm-isaac` | Name shown in Moonlight |
| `SUNSHINE_USER` / `SUNSHINE_PASS` | empty | Set/reset the web UI login at start |
| `SUNSHINE_CSRF_ORIGINS` | empty | Extra web UI origins to trust (host IPs and hostname are added automatically) |
| `SUNSHINE_ALLOWED_SUBNETS` | (required) | Source subnets allowed to reach Sunshine (`sunshine-firewall.sh`) |
| `SUNSHINE_VERSION` | `2026.914.233613` | Sunshine release baked into the image (`./scripts/sunshine.sh build`) |

After changing resolution, run `./scripts/sunshine.sh restart`. Sunshine's own
settings (bitrate, codec, etc.) can be changed in the web UI and persist in the volume.
To reset everything, run `docker volume rm docker_sunshine-config`.

## GPU budget

The GPU is shared. Xorg and the desktop use about 100–200 MB of VRAM, plus one NVENC
session while a client is streaming. On a 12 GB card, GUI rendering costs more VRAM
and speed than headless. Use `--headless` for long training or collection runs, and
GUI mode to inspect them.

## Troubleshooting

| Symptom | Check / fix |
|---------|-------------|
| `sunshine.sh up` says Xorg exited | `./scripts/sunshine.sh xlog`. **"no screens found"** → try `SUNSHINE_NVIDIA_OUTPUT=DFP-1` (or another `DFP-n` listed in the log). **VT / tty errors** → add `privileged: true` to the `sunshine` service (some Docker setups block Xorg's VT handling). |
| Log shows the driver-bundle download | Normal on first start when the toolkit doesn't mount the NVIDIA Xorg modules. The bundle is cached per driver version in `sunshine-xorg-modules`. With no internet, mount the host's `nvidia_drv.so` and `libglxserver_nvidia.so` into the container instead. |
| Moonlight shows video, but mouse/keyboard do nothing | `/dev/uinput` is missing on the host (see prerequisites). Also check that `/run/udev` exists on the host. |
| Black screen in Moonlight | The display works but nothing is open yet. Launch something with `--gui`. Sunshine logs: `./scripts/sunshine.sh logs`. |
| Isaac Sim: `cannot open display` | `./scripts/sunshine.sh status` should show the display. The `--gui` run must use the overlay; the scripts set `COMPOSE_FILE` for this. |
| Port conflict on UDP 47998 | The Isaac Sim WebRTC livestream (`LIVESTREAM=2`) uses the same port. Don't run `--watch` / WebRTC while Sunshine is up, or change Sunshine's `port` in the web UI. |

## Security notes

- The X server accepts any **local** connection with no auth cookie. TCP is disabled,
  so only processes on this host (or containers mounting the socket) can reach it.
  This is fine for a dedicated sim box, but not for a shared multi-user machine.
- All Sunshine ports are firewalled to `SUNSHINE_ALLOWED_SUBNETS` (LAN + VPN) by
  `sunshine-firewall.sh`.
- The Sunshine web UI is limited to the LAN (`origin_web_ui_allowed = lan`). Moonlight
  clients must be paired with a PIN. Don't forward these ports to the internet; use a
  VPN (WireGuard/Tailscale) for remote access.
