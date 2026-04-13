# Architecture

This document describes the system architecture, data flows, and component
interactions of the VLA Robot Full Stack for SO-ARM101.

---

## System Overview

The stack is split into two logical groups of Docker containers that can run
on the **same machine** (Mode A) or on **separate machines** (Mode B).

```mermaid
graph TD
    subgraph localMachine ["Local Machine  --  3080 Ti + 256 GB RAM"]
        IsaacSim["Isaac Lab\nnvcr.io/nvidia/isaac-lab:2.1.0\n- Isaac Sim 5.1 + Isaac Lab 2.1\n- Physics sim (PhysX 5)\n- RTX camera rendering\n- Headless episode generation"]
        ROS2["ROS2 Bridge\nros:humble-ros-base\n- VLA Bridge Node\n- Joint state / camera subscribers\n- Action publisher"]
        DataVol["data/episodes/\nLeRobot v3.0\nParquet + MP4 + JSON"]
        URDFVol["robot_description/\nURDF + STL + USD"]
    end

    subgraph gpuMachine ["GPU Machine  --  local OR cloud A100/H100"]
        OpenPi["OpenPi Server\npi0 / pi0-FAST\nWebSocket :8000"]
        Training["Training\nQLoRA or Full LoRA\nPyTorch + DeepSpeed"]
        Caddy["Caddy\nTLS reverse proxy\nwss://:8443"]
        ModelsVol["models/\nCheckpoints\nLoRA weights"]
    end

    IsaacSim -- "ROS2 topics\n/joint_states\n/camera/*/image_raw" --> ROS2
    IsaacSim -- "writes episodes" --> DataVol
    ROS2 -- "WebSocket\nobservation -> actions" --> OpenPi
    OpenPi -- "reads" --> ModelsVol
    Training -- "reads episodes" --> DataVol
    Training -- "writes checkpoints" --> ModelsVol
    Caddy -- "reverse_proxy :8000" --> OpenPi
    ROS2 -. "WAN: wss://:8443" .-> Caddy
    IsaacSim -- "reads" --> URDFVol
    ROS2 -- "reads" --> URDFVol
```

---

## Deployment Modes

### Mode A -- All Local

Every container runs on the same GPU machine. Docker Compose **profiles**
time-share the single GPU because Isaac Sim, OpenPi inference, and training
cannot coexist in 12 GB of VRAM simultaneously.

```mermaid
graph LR
    subgraph singleBox ["Single Machine  --  RTX 3080 Ti"]
        direction TB
        IS["Isaac Sim"]
        OP["OpenPi Server"]
        R2["ROS2 Bridge"]
        TR["Training"]
    end
    IS -.->|"profiles\ntime-share GPU"| OP
    IS -.->|"profiles\ntime-share GPU"| TR
```

| Profile | Containers started | GPU usage |
|---|---|---|
| `collect` | Isaac Sim + ROS2 | ~8-10 GB |
| `train` | Training only | ~10-12 GB |
| `eval-local` | Isaac Sim + OpenPi + ROS2 | ~10 GB |
| `deploy-local` | OpenPi + ROS2 | ~4-6 GB |

### Mode B -- Split (Local + Remote)

Isaac Sim and ROS2 run locally.  OpenPi and training run on a remote GPU,
connected via WebSocket over the network.

```mermaid
graph LR
    subgraph local ["Local  --  3080 Ti"]
        IS2["Isaac Sim"]
        R22["ROS2 Bridge"]
    end
    subgraph cloud ["Cloud  --  A100 80 GB"]
        OP2["OpenPi Server"]
        CA["Caddy TLS"]
        TR2["Training"]
    end
    R22 -- "wss:// over WAN" --> CA
    CA --> OP2
```

| Profile | Local containers | Remote containers |
|---|---|---|
| `eval-remote` | Isaac Sim + ROS2 | OpenPi + Caddy |
| `deploy-remote` | ROS2 only | OpenPi + Caddy |
| (training) | -- | Training |

Switching modes requires only changing `OPENPI_HOST` in `docker/.env`.

---

## Container Details

### 1. Isaac Sim

| | |
|---|---|
| **Image** | `nvcr.io/nvidia/isaac-lab:2.1.0` (Isaac Sim 5.1 + Isaac Lab 2.1 pre-installed) |
| **GPU** | Required (RT cores) |
| **Dockerfile** | `docker/isaac-sim/Dockerfile` |
| **Volumes** | `robot_description`, `isaac_envs`, `data` |

Runs NVIDIA Isaac Sim with Isaac Lab extensions.  Two custom environments
load the SO-ARM101 from its USD asset and expose Gymnasium-compatible APIs:

- **SoarmReachEnv** -- Move end-effector to a random XYZ target.
- **SoarmPickEnv** -- Grasp a cube and place it at a target.

The `sim_data_collector.py` script runs a policy inside the environment,
recording observations and actions into LeRobot v3.0 format.

### 2. OpenPi Server

| | |
|---|---|
| **Base** | `nvidia/cuda:12.4.0-runtime-ubuntu22.04` |
| **GPU** | Required for inference |
| **Dockerfile** | `docker/openpi-server/Dockerfile` |
| **Port** | 8000 (WebSocket) |

Clones Physical Intelligence's
[openpi](https://github.com/Physical-Intelligence/openpi) repository,
installs it with `uv`, and starts `serve_policy.py`.  The server loads a
checkpoint (base model or fine-tuned LoRA), accepts observation dicts over
WebSocket, and returns action chunks.

### 3. ROS2 Bridge

| | |
|---|---|
| **Base** | `ros:humble-ros-base-jammy` |
| **GPU** | Not required |
| **Dockerfile** | `docker/ros2-bridge/Dockerfile` |

Contains three ROS2 packages:

| Package | Purpose |
|---|---|
| `soarm_description` | URDF, meshes, TF publishing |
| `soarm_moveit_config` | MoveIt2 SRDF (placeholder) |
| `soarm_vla_bridge` | VLA bridge node (core) |

The **VLA bridge node** is the central piece: it subscribes to sensor topics,
builds an observation dict, sends it to the OpenPi server, and publishes the
returned actions as joint trajectory commands.

### 4. Training

| | |
|---|---|
| **Base** | `nvidia/cuda:12.4.0-devel-ubuntu22.04` |
| **GPU** | Required |
| **Dockerfile** | `docker/training/Dockerfile` |

Same OpenPi codebase as the server, plus `bitsandbytes`, `peft`,
`deepspeed`, and `accelerate` for memory-efficient fine-tuning.

### 5. Caddy (Cloud only)

| | |
|---|---|
| **Image** | `caddy:2-alpine` |
| **GPU** | Not required |
| **Ports** | 8443, 443 |

TLS reverse proxy that terminates `wss://` connections from the local
machine and forwards them to the OpenPi server on port 8000.

---

## Data Flow

```mermaid
sequenceDiagram
    participant Sim as Isaac Sim
    participant Disk as data/episodes/
    participant Train as Training
    participant Models as models/
    participant Server as OpenPi Server
    participant Bridge as ROS2 Bridge
    participant Robot as Robot (sim or real)

    Note over Sim,Disk: Phase 1 -- Data Collection
    loop every episode
        Sim->>Sim: Run policy in environment
        Sim->>Disk: Write Parquet + MP4 + JSON
    end

    Note over Disk,Models: Phase 2 -- Training
    Disk->>Train: Read episodes
    Train->>Train: LoRA fine-tuning
    Train->>Models: Save checkpoint

    Note over Models,Robot: Phase 3 -- Evaluation / Deployment
    Models->>Server: Load checkpoint
    Robot->>Bridge: Joint states + camera images
    Bridge->>Server: WebSocket infer(observation)
    Server->>Bridge: Action chunk [a0, a1, ..., aN]
    loop every control tick
        Bridge->>Robot: Publish joint command
    end
```

---

## ROS2 Topic Map

```mermaid
graph LR
    subgraph isaacOrReal ["Isaac Sim  /  Real Robot"]
        JS["/joint_states\nJointState"]
        WC["/camera/wrist/image_raw\nImage"]
        OC["/camera/overhead/image_raw\nImage"]
    end
    subgraph bridge ["VLA Bridge Node"]
        OB["Observation\nBuilder"]
        AQ["Action Queue\n(deque)"]
    end
    subgraph openpi ["OpenPi Server"]
        INF["infer(obs)\nWebSocket"]
    end
    subgraph commands ["Published"]
        JC["/joint_commands\nJointTrajectory"]
    end
    subgraph prompts ["User Input"]
        PR["/vla/prompt\nString"]
    end

    JS --> OB
    WC --> OB
    OC --> OB
    PR --> OB
    OB -- "WebSocket" --> INF
    INF -- "action chunk" --> AQ
    AQ -- "10 Hz timer" --> JC
```

---

## File Map

```
isaac-sim-soarm101/
├── docker/
│   ├── docker-compose.yml ........... Local stack orchestration
│   ├── docker-compose.cloud.yml ..... Remote GPU stack
│   ├── .env ......................... Local environment vars
│   ├── .env.cloud ................... Cloud environment vars
│   ├── isaac-sim/
│   │   ├── Dockerfile
│   │   └── convert_urdf.py ......... URDF -> USD helper
│   ├── openpi-server/
│   │   ├── Dockerfile
│   │   ├── entrypoint.sh
│   │   └── Caddyfile ............... TLS proxy config
│   ├── ros2-bridge/
│   │   ├── Dockerfile
│   │   └── ros_entrypoint.sh
│   └── training/
│       ├── Dockerfile
│       └── entrypoint.sh
├── robot_description/
│   ├── urdf/soarm101.urdf .......... ROS2 mesh paths
│   ├── urdf/soarm101_isaacsim.urdf . Absolute mesh paths
│   ├── meshes/*.stl ................ 13 STL files
│   └── usd/soarm101.usd ........... Generated by convert_urdf.py
├── isaac_envs/
│   ├── soarm_reach_env.py .......... Reach-target task
│   ├── soarm_pick_env.py ........... Pick-and-place task
│   └── sim_data_collector.py ....... Episode recorder
├── ros2_ws/src/
│   ├── soarm_description/ .......... URDF + launch files
│   ├── soarm_moveit_config/ ........ MoveIt2 SRDF
│   └── soarm_vla_bridge/ ........... VLA bridge ROS2 node
│       └── soarm_vla_bridge/
│           ├── vla_bridge_node.py .. Main bridge node
│           └── observation_builder.py
├── training/
│   ├── configs/soarm_config.py ..... OpenPi DataConfig
│   └── scripts/
│       ├── compute_norm_stats.py
│       ├── train_lora.py
│       └── convert_sim_episodes.py
├── data/episodes/ .................. LeRobot v3.0 dataset
├── models/ ......................... Checkpoints + LoRA
└── scripts/ ........................ Workflow shell scripts
```
