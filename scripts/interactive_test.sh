#!/usr/bin/env bash
# Launch the interactive VLA inference test environment.
#
# Starts Isaac Sim (with WebRTC streaming + omni.ui control panel),
# the OpenPi policy server, and the ROS2 VLA bridge. The user connects
# via the Isaac Sim WebRTC Streaming Client to interact with the VLA.
#
# Usage:
#   ./scripts/interactive_test.sh
#   ./scripts/interactive_test.sh --public-ip 192.168.1.50
#   ./scripts/interactive_test.sh --model-config soarm_pi0
#   ./scripts/interactive_test.sh --checkpoint /path/to/lora/weights
#   ./scripts/interactive_test.sh --remote gpu-server.example.com           # remote inference (port 8443)
#   ./scripts/interactive_test.sh --remote 192.168.1.100 --port 8000         # remote OpenVLA on port 8000
#
# Prerequisites:
#   - NVIDIA GPU with NVENC support (RTX series; A100 NOT supported)
#   - Docker with nvidia-container-toolkit
#   - Isaac Sim WebRTC Streaming Client (download from NVIDIA)
#
# Ports required:
#   TCP 49100  (WebRTC signaling)
#   UDP 47998  (WebRTC media stream)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PUBLIC_IP=""
MODEL_CONFIG="soarm_pi0_fast"
CHECKPOINT=""
SIM_ONLY=0
REMOTE_HOST=""
REMOTE_PORT="8443"

while [[ $# -gt 0 ]]; do
    case $1 in
        --public-ip) PUBLIC_IP="$2"; shift 2 ;;
        --model-config) MODEL_CONFIG="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --sim-only) SIM_ONLY=1; shift ;;
        --remote) REMOTE_HOST="$2"; shift 2 ;;
        --port) REMOTE_PORT="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ -z "$PUBLIC_IP" ]]; then
    PUBLIC_IP="$(hostname -I 2>/dev/null | awk '{print $1}')" || PUBLIC_IP="127.0.0.1"
fi

echo "=== Interactive VLA Inference Test ==="
echo ""
echo "Configuration:"
echo "  Public IP:     $PUBLIC_IP"
if [[ -n "$REMOTE_HOST" ]]; then
    echo "  Inference:     REMOTE → ${REMOTE_HOST}:${REMOTE_PORT}"
else
    echo "  Model config:  $MODEL_CONFIG"
    echo "  Sim-only mode: $( [[ $SIM_ONLY -eq 1 ]] && echo "YES (no VLA server)" || echo "no" )"
    if [[ -n "$CHECKPOINT" ]]; then
        echo "  Checkpoint:    $CHECKPOINT"
    fi
fi

# Check available VRAM before launching (only warn when running local OpenPi + Isaac Sim)
GPU_MEM_TOTAL=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -1)
GPU_MEM_FREE=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>/dev/null | head -1)
if [[ -n "$GPU_MEM_FREE" ]] && [[ $SIM_ONLY -eq 0 ]] && [[ -z "$REMOTE_HOST" ]]; then
    echo ""
    echo "  GPU memory:    ${GPU_MEM_FREE} MiB free / ${GPU_MEM_TOTAL} MiB total"
    if [[ -n "$GPU_MEM_TOTAL" ]] && [[ "$GPU_MEM_TOTAL" -lt 16000 ]]; then
        echo ""
        echo "  WARNING: Running Isaac Sim + local OpenPi together typically"
        echo "  requires >=16 GB VRAM. Your GPU has ${GPU_MEM_TOTAL} MiB."
        echo "  Consider: --sim-only | --remote HOST | or continue anyway."
        echo ""
        read -r -p "  Continue anyway? [y/N] " confirm
        if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
            echo "Aborted."
            exit 0
        fi
    fi
fi

echo ""
echo "Once all services are up and you see 'Interactive inference environment ready':"
echo ""
echo "  1. Open the Isaac Sim WebRTC Streaming Client."
echo "  2. Enter server address: $PUBLIC_IP"
echo "  3. Click Connect."
echo "  4. Use the 'VLA Interactive Test' panel to send commands."
echo ""
echo "  Ports required: TCP 49100, UDP 47998 (host networking)."
echo "  Note: only one WebRTC client can connect at a time."
echo ""

cd "$PROJECT_DIR/docker"

# Build environment variable overrides
EXTRA_ENV=()
EXTRA_ENV+=(-e ISAAC_SIM_HEADLESS=1)
EXTRA_ENV+=(-e LIVESTREAM=2)
EXTRA_ENV+=(-e "PUBLIC_IP=$PUBLIC_IP")
EXTRA_ENV+=(-e ENABLE_CAMERAS=1)
EXTRA_ENV+=(-e PYTHONUNBUFFERED=1)
EXTRA_ENV+=(-e "OPENPI_POLICY_CONFIG=$MODEL_CONFIG")

if [[ -n "$CHECKPOINT" ]]; then
    EXTRA_ENV+=(-e "OPENPI_CHECKPOINT_DIR=$CHECKPOINT")
fi

if [[ -n "$REMOTE_HOST" ]]; then
    # Use remote inference: start only the ROS2 bridge, point at remote server
    echo "Starting ROS2 bridge (remote inference at ${REMOTE_HOST}:${REMOTE_PORT})..."
    echo "  (To watch bridge logs: cd $PROJECT_DIR/docker && docker compose --profile interactive logs -f ros2-bridge)"
    OPENPI_HOST="$REMOTE_HOST" OPENPI_PORT="$REMOTE_PORT" docker compose --profile interactive up -d ros2-bridge
elif [[ $SIM_ONLY -eq 0 ]]; then
    # Local inference: start OpenPi server and ROS2 bridge
    echo "Starting OpenPi server and ROS2 bridge..."
    docker compose --profile interactive up -d openpi-server ros2-bridge
fi

# Run Isaac Sim in the foreground with the interactive script.
echo "Starting Isaac Sim with interactive inference environment..."
docker compose --profile interactive run --rm \
    "${EXTRA_ENV[@]}" \
    isaac-sim \
    /isaac-sim/python.sh /isaac_envs/interactive_inference.py

# Cleanup background services on exit (we started something unless sim-only only)
echo ""
echo "Shutting down services..."
if [[ $SIM_ONLY -eq 0 ]] || [[ -n "$REMOTE_HOST" ]]; then
    docker compose --profile interactive down
fi
echo "=== Done ==="
