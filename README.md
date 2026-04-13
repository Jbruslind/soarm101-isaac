# VLA Robot Full Stack: Isaac Sim + OpenPi + ROS2 for SO-ARM101

A Docker-based environment for training and deploying Vision-Language-Action (VLA)
models on the SO-ARM101 6-axis robot arm. Uses NVIDIA Isaac Sim for simulation,
Physical Intelligence's OpenPi for pi0/pi0-FAST policy serving, and ROS2 Humble
for robot communication.

## Hardware Requirements

- **GPU**: NVIDIA RTX 3080 Ti (12 GB) or better (RT cores required for Isaac Sim)
- **RAM**: 64 GB minimum, 256 GB recommended for CPU-offloaded training
- **Disk**: 100 GB+ free (Isaac Sim container images are large)
- **OS**: Ubuntu 22.04
- **Driver**: NVIDIA 535.129.03+

## Quick Start

```bash
# 0. Verify project structure
bash scripts/smoke_test.sh

# 1. Download robot URDF + meshes, convert to USD
./scripts/setup_robot_usd.sh

# 2. Collect 100 episodes in headless Isaac Sim
./scripts/collect_sim_data.sh --env reach --episodes 100 --camera

# 3. Train a pi0-FAST LoRA policy (QLoRA on 3080 Ti)
./scripts/train.sh

# 4. Evaluate trained policy in closed-loop simulation
./scripts/eval_sim.sh

# 5. Interactively test the VLA via WebRTC streaming
./scripts/interactive_test.sh

# 6. (Optional) Deploy inference to a cloud GPU
./scripts/deploy_cloud.sh user@gpu-server
./scripts/eval_sim.sh --remote gpu-server.example.com
```

See **[Full Example](docs/FULL_EXAMPLE.md)** for a detailed walkthrough of
every phase.

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│  Local Machine (3080 Ti)                                │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐  │
│  │  Isaac Sim    │  │  ROS2 Bridge │  │   Data/      │  │
│  │  Simulation   │──│  VLA Client  │  │   Episodes   │  │
│  │  + Isaac Lab  │  │  + MoveIt2   │  │   (LeRobot)  │  │
│  └──────────────┘  └──────┬───────┘  └──────────────┘  │
└─────────────────────────────┼───────────────────────────┘
                              │ WebSocket (LAN or WAN)
┌─────────────────────────────┼───────────────────────────┐
│  GPU Machine (local or cloud A100/H100)                 │
│  ┌──────────────┐  ┌───────┴──────┐  ┌──────────────┐  │
│  │  Training     │  │  OpenPi      │  │   Models/    │  │
│  │  QLoRA/Full   │  │  Policy Srv  │  │   Checkpts   │  │
│  └──────────────┘  └──────────────┘  └──────────────┘  │
└─────────────────────────────────────────────────────────┘
```

All containers run on a single GPU (Mode A) or split across local +
remote machines (Mode B). See [Architecture](docs/ARCHITECTURE.md) for
detailed diagrams and data flow.

## Project Structure

```
├── docker/                     Docker infrastructure
│   ├── docker-compose.yml      Local stack (sim + ROS2 + optional GPU)
│   ├── docker-compose.cloud.yml Remote GPU stack
│   ├── .env                    Local configuration
│   ├── .env.cloud              Cloud configuration
│   ├── isaac-sim/Dockerfile
│   ├── openpi-server/Dockerfile + Caddyfile
│   ├── ros2-bridge/Dockerfile
│   └── training/Dockerfile
├── robot_description/          URDF, meshes, USD assets
├── isaac_envs/                 Isaac Lab task environments
├── ros2_ws/src/                ROS2 packages
│   ├── soarm_description/      Robot URDF + launch
│   ├── soarm_moveit_config/    MoveIt2 config
│   └── soarm_vla_bridge/       OpenPi <-> ROS2 bridge node
├── training/                   Training configs + scripts
├── data/episodes/              LeRobot v3.0 dataset
├── models/                     Checkpoints + LoRA weights
├── scripts/                    Workflow shell scripts
└── docs/                       Full documentation
```

## Docker Compose Profiles

| Profile | Services | Use Case |
|---------|----------|----------|
| `collect` | Isaac Sim + ROS2 | Generate sim episodes |
| `train` | Training only | LoRA fine-tuning |
| `eval-local` | Isaac Sim + ROS2 + OpenPi | Evaluate with local GPU |
| `eval-remote` | Isaac Sim + ROS2 | Evaluate with remote OpenPi |
| `interactive` | Isaac Sim + ROS2 + OpenPi | Interactive VLA testing via WebRTC |
| `deploy-local` | ROS2 + OpenPi | Real robot, local inference |
| `deploy-remote` | ROS2 | Real robot, remote inference |

Profiles prevent GPU memory conflicts on a 12 GB card by ensuring only
compatible containers run simultaneously.

## Documentation

| Document | Description |
|----------|-------------|
| **[Quick Start](docs/QUICKSTART.md)** | Step-by-step setup and first run |
| **[Full Example](docs/FULL_EXAMPLE.md)** | Complete end-to-end walkthrough with expected output |
| **[Architecture](docs/ARCHITECTURE.md)** | System diagrams, data flows, container details, file map |
| **[Data Pipeline](docs/DATA_PIPELINE.md)** | LeRobot v3.0 format, collection, normalization |
| **[Training Guide](docs/OPENPI_TRAINING.md)** | OpenPi fine-tuning, memory budget, config reference |
| **[Isaac Sim Environments](docs/ISAAC_SIM_ENVIRONMENTS.md)** | Reach and pick tasks, observation/action spaces |
| **[ROS2 Bridge](docs/ROS2_BRIDGE.md)** | VLA bridge node, topics, action chunking |
| **[Remote Deployment](docs/REMOTE_DEPLOYMENT.md)** | Cloud GPU setup, TLS, data sync, latency |
| **[Interactive Inference](docs/INTERACTIVE_INFERENCE.md)** | Interactive VLA testing with WebRTC streaming |
| **[Troubleshooting](docs/TROUBLESHOOTING.md)** | Common errors and solutions |

## Workflow Scripts

| Script | Purpose |
|--------|---------|
| `scripts/smoke_test.sh` | Validate project structure (offline) |
| `scripts/setup_robot_usd.sh` | Download robot assets, URDF-to-USD conversion |
| `scripts/collect_sim_data.sh` | Run Isaac Sim and record episodes |
| `scripts/train.sh` | LoRA fine-tuning (local or remote) |
| `scripts/eval_sim.sh` | Closed-loop simulation evaluation |
| `scripts/deploy_real.sh` | Deploy to real robot |
| `scripts/interactive_test.sh` | Interactive VLA inference via WebRTC |
| `scripts/deploy_cloud.sh` | Deploy GPU stack to remote machine |
| `scripts/sync_data.sh` | Sync episodes and checkpoints with remote |

## Key Technologies

- **[NVIDIA Isaac Sim 5.1](https://developer.nvidia.com/isaac-sim)** -- High-fidelity physics simulation with RTX rendering
- **[Isaac Lab](https://isaac-sim.github.io/IsaacLab/)** -- Gymnasium-compatible RL/IL environments for Isaac Sim
- **[OpenPi](https://github.com/Physical-Intelligence/openpi)** -- pi0 / pi0-FAST VLA model training and serving
- **[LeRobot v3.0](https://huggingface.co/docs/lerobot)** -- Standardized robot learning dataset format
- **[ROS2 Humble](https://docs.ros.org/en/humble/)** -- Robot communication middleware
- **[SO-ARM101](https://github.com/TheRobotStudio/SO-ARM100)** -- Open-source 6-DOF robot arm

## License

Apache-2.0
