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
#   ./scripts/interactive_test.sh --policy-mode soarm --model-config soarm_pi0_fast --checkpoint /models/soarm_lora
#   ./scripts/interactive_test.sh --policy-mode droid --model-config pi0_fast_droid
#   ./scripts/interactive_test.sh --remote gpu-server.example.com           # remote inference (port 8443)
#   ./scripts/interactive_test.sh --remote 192.168.1.100 --port 8000         # remote OpenVLA on port 8000
#   VLA_AUTO_STOP_SEC=300 ./scripts/interactive_test.sh --remote HOST        # lengthen Execute auto-stop (default 120s)
#
# Prerequisites:
#   - NVIDIA GPU with NVENC support (RTX series; A100 NOT supported)
#   - Docker with nvidia-container-toolkit
#   - Isaac Sim WebRTC Streaming Client (download from NVIDIA)
#
# Ports required:
#   TCP 49100  (WebRTC signaling)
#   UDP 47998  (WebRTC media stream)
#   TCP 8080   (Phospho Bridge web dashboard)
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

PUBLIC_IP=""
MODEL_CONFIG="soarm_pi0_fast"
CHECKPOINT=""
SIM_ONLY=0
REMOTE_HOST=""
REMOTE_PORT="8443"
ROS2_GRAPH_MODE="soarm"
POLICY_MODE="${OPENPI_POLICY_MODE:-soarm}"
# Isaac panel "Auto-stop (sec)" default; must exceed slow remote OpenPi (often 60–120s+ first chunk).
VLA_AUTO_STOP_SEC="${VLA_AUTO_STOP_SEC:-120}"

while [[ $# -gt 0 ]]; do
    case $1 in
        --public-ip) PUBLIC_IP="$2"; shift 2 ;;
        --model-config) MODEL_CONFIG="$2"; shift 2 ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --sim-only) SIM_ONLY=1; shift ;;
        --remote) REMOTE_HOST="$2"; shift 2 ;;
        --port) REMOTE_PORT="$2"; shift 2 ;;
        --ros2-graph-mode) ROS2_GRAPH_MODE="$2"; shift 2 ;;
        --policy-mode) POLICY_MODE="$2"; shift 2 ;;
        --auto-stop) VLA_AUTO_STOP_SEC="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

if [[ "$ROS2_GRAPH_MODE" != "soarm" && "$ROS2_GRAPH_MODE" != "minimal" ]]; then
    echo "Invalid --ros2-graph-mode: $ROS2_GRAPH_MODE (use 'soarm' or 'minimal')"
    exit 1
fi
if [[ "$POLICY_MODE" != "soarm" && "$POLICY_MODE" != "droid" ]]; then
    echo "Invalid --policy-mode: $POLICY_MODE (use 'soarm' or 'droid')"
    exit 1
fi
if [[ "$POLICY_MODE" == "soarm" ]]; then
    if [[ "$MODEL_CONFIG" != soarm_* ]]; then
        echo "Invalid --model-config for soarm mode: $MODEL_CONFIG (expected prefix 'soarm_')"
        exit 1
    fi
    if [[ -z "$CHECKPOINT" && -z "$REMOTE_HOST" && $SIM_ONLY -eq 0 ]]; then
        echo "ERROR: --policy-mode soarm requires --checkpoint for local OpenPi server."
        echo "       Refusing implicit fallback to DROID."
        exit 1
    fi
fi

# Prefer the IPv4 used for the default route (usually the LAN address other devices can reach).
# `hostname -I` often lists Docker/Wi-Fi/VPN in arbitrary order; the first entry is frequently wrong.
_default_route_ip() {
    ip -4 route get 1.1.1.1 2>/dev/null | awk '{for (i=1;i<=NF;i++) if ($i=="src") {print $(i+1); exit}}'
}
if [[ -z "$PUBLIC_IP" ]]; then
    PUBLIC_IP="$(_default_route_ip)"
    if [[ -z "$PUBLIC_IP" ]]; then
        PUBLIC_IP="$(hostname -I 2>/dev/null | awk '{print $1}')" || PUBLIC_IP="127.0.0.1"
    fi
fi

echo "=== Interactive VLA Inference Test ==="
echo ""
echo "Configuration:"
echo "  Public IP:     $PUBLIC_IP"
echo "  ROS2 graph:    $ROS2_GRAPH_MODE"
echo "  Policy mode:   $POLICY_MODE"
echo "  VLA auto-stop: ${VLA_AUTO_STOP_SEC}s (Execute timer in Sim panel; env VLA_AUTO_STOP_SEC or --auto-stop)"
if [[ -n "$REMOTE_HOST" ]]; then
    echo "  Inference:     REMOTE → ${REMOTE_HOST}:${REMOTE_PORT}"
    if [[ "$REMOTE_PORT" == "8443" ]]; then
        echo "  Note:          The ROS2 bridge uses openpi-client (ws://)."
        echo "                If your remote uses Caddy TLS on :8443, use --port 8000 (direct)"
        echo "                or configure a trusted TLS endpoint for wss://."
    fi
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

PHOSPHO_PORT="${PHOSPHO_BRIDGE_PORT:-8080}"

echo ""
echo "Once all services are up and you see 'Interactive inference environment ready':"
echo ""
echo "  Web Dashboard (Phospho Bridge):"
echo "    http://${PUBLIC_IP}:${PHOSPHO_PORT}"
echo "    Diagnostics JSON: curl -sS http://${PUBLIC_IP}:${PHOSPHO_PORT}/api/debug/status | jq ."
echo "    Bridge logs: cd $PROJECT_DIR/docker && docker compose --profile interactive logs -f phospho-bridge"
echo "    Quick check from another machine: curl -sS http://${PUBLIC_IP}:${PHOSPHO_PORT}/api/health"
echo "    If that times out, use the correct LAN IP of this host (not 127.0.0.1) and open TCP ${PHOSPHO_PORT}"
echo "    in the firewall (e.g. sudo ufw allow ${PHOSPHO_PORT}/tcp)."
echo "    Control the robot with keyboard/gamepad, view cameras, send VLA commands."
echo ""
echo "  Isaac Sim WebRTC (optional, for full 3D viewport):"
echo "    1. Open the Isaac Sim WebRTC Streaming Client."
echo "    2. Enter server address: $PUBLIC_IP"
echo "    3. Click Connect."
echo "    4. Use the 'VLA Interactive Test' panel to send commands."
echo ""
echo "  Ports required: TCP 49100, UDP 47998, TCP ${PHOSPHO_PORT} (host networking)."
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
EXTRA_ENV+=(-e "OPENPI_POLICY_MODE=$POLICY_MODE")
EXTRA_ENV+=(-e "ROS2_GRAPH_MODE=$ROS2_GRAPH_MODE")
EXTRA_ENV+=(-e "VLA_AUTO_STOP_SEC=$VLA_AUTO_STOP_SEC")

if [[ -n "$CHECKPOINT" ]]; then
    EXTRA_ENV+=(-e "OPENPI_CHECKPOINT_DIR=$CHECKPOINT")
fi

# Start the phospho-bridge web dashboard (always, for all interactive modes).
# Export PHOSPHO_BRIDGE_PORT so compose substitution matches the printed URL.
echo "Starting Phospho Bridge web dashboard on port ${PHOSPHO_PORT}..."
export PHOSPHO_BRIDGE_PORT="${PHOSPHO_PORT}"
docker compose --profile interactive up -d phospho-bridge

if [[ -n "$REMOTE_HOST" ]]; then
    # Use remote inference: start only the ROS2 bridge, point at remote server
    echo "Starting ROS2 bridge (remote inference at ${REMOTE_HOST}:${REMOTE_PORT})..."
    echo "  (To watch bridge logs: cd $PROJECT_DIR/docker && docker compose --profile interactive logs -f ros2-bridge)"
    OPENPI_HOST="$REMOTE_HOST" OPENPI_PORT="$REMOTE_PORT" OPENPI_POLICY_MODE="$POLICY_MODE" docker compose --profile interactive up -d ros2-bridge
elif [[ $SIM_ONLY -eq 0 ]]; then
    # Local inference: start OpenPi server and ROS2 bridge
    echo "Starting OpenPi server and ROS2 bridge..."
    OPENPI_POLICY_MODE="$POLICY_MODE" OPENPI_POLICY_CONFIG="$MODEL_CONFIG" OPENPI_CHECKPOINT_DIR="$CHECKPOINT" docker compose --profile interactive up -d openpi-server ros2-bridge
fi

# Run Isaac Sim in the foreground with the interactive script.
# --kit_args enables the ROS2 bridge extension at Kit startup so it is loaded
# before the Python script tries to import it. On first run the extension is
# downloaded to the persistent isaac-cache-kit volume; subsequent runs use cache.
echo "Starting Isaac Sim with interactive inference environment..."
docker compose --profile interactive run --rm \
    "${EXTRA_ENV[@]}" \
    isaac-sim \
    /isaac-sim/python.sh /isaac_envs/interactive_inference.py \
    --kit_args '--enable isaacsim.ros2.bridge --/rtx/verifyDriverVersion/enabled=false'

# Cleanup background services on exit (phospho-bridge always runs; others depend on mode)
echo ""
echo "Shutting down services..."
docker compose --profile interactive down
echo "=== Done ==="
