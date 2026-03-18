#!/usr/bin/env bash
# Collect simulation episodes from Isaac Sim and save in LeRobot v3.0 format.
# Uses an IK-based scripted policy to generate smooth demonstrations.
#
# Usage:
#   ./scripts/collect_sim_data.sh [--env reach|pick] [--episodes 50]
#   ./scripts/collect_sim_data.sh --no-camera              # skip video recording
#   ./scripts/collect_sim_data.sh --watch                   # live-view via WebRTC
#   ./scripts/collect_sim_data.sh --watch --public-ip 192.168.1.50
#
# Cameras are ON by default (required for VLA training).  Each episode
# produces two MP4 files in data/episodes/videos/:
#   - observation.images.wrist_episode_000000.mp4
#   - observation.images.third_person_episode_000000.mp4
#
# Use --watch to enable WebRTC streaming so you can see the robot move in
# real time via the Isaac Sim WebRTC Streaming Client. Implies --wait so you
# have time to connect before collection starts. Requires an NVENC-capable GPU
# (RTX series; A100 is NOT supported).
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ENV_TYPE="reach"
NUM_EPISODES=50
NO_CAMERA=""
WAIT_FOR_KEY=""
WATCH=""
PUBLIC_IP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --env) ENV_TYPE="$2"; shift 2 ;;
        --episodes) NUM_EPISODES="$2"; shift 2 ;;
        --no-camera) NO_CAMERA="--no-camera"; shift ;;
        --wait) WAIT_FOR_KEY="--wait-for-key"; shift ;;
        --watch) WATCH=1; shift ;;
        --public-ip) PUBLIC_IP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# --watch implies --wait so the user can connect before collection begins.
if [[ -n "$WATCH" ]]; then
    WAIT_FOR_KEY="--wait-for-key"
    if [[ -z "$PUBLIC_IP" ]]; then
        PUBLIC_IP="$(hostname -I 2>/dev/null | awk '{print $1}')" || PUBLIC_IP="127.0.0.1"
    fi
fi

echo "=== Collecting $NUM_EPISODES episodes (env=$ENV_TYPE) ==="

if [[ -n "$WATCH" ]]; then
    echo ""
    echo "WebRTC streaming enabled (LIVESTREAM=2)."
    echo "  1. Wait for 'Environment ready' in the log output."
    echo "  2. Open the Isaac Sim WebRTC Streaming Client."
    echo "  3. Enter server address: $PUBLIC_IP"
    echo "  4. Click Connect."
    echo "  5. Once you see the robot, press Enter here to start collection."
    echo ""
    echo "  Ports required: TCP 8011, TCP 49100, UDP 47998 (host networking)."
    echo "  Note: only one WebRTC client can connect at a time."
    echo ""
fi

cd "$PROJECT_DIR/docker"

# Build environment variable flags for docker compose run.
EXTRA_ENV=()
EXTRA_ENV+=(-e ISAAC_SIM_HEADLESS=1)
EXTRA_ENV+=(-e PYTHONUNBUFFERED=1)
if [[ -z "$NO_CAMERA" ]]; then
    EXTRA_ENV+=(-e ENABLE_CAMERAS=1)
fi
if [[ -n "$WATCH" ]]; then
    EXTRA_ENV+=(-e LIVESTREAM=2)
    EXTRA_ENV+=(-e "PUBLIC_IP=$PUBLIC_IP")
fi

docker compose --profile collect run --rm \
    "${EXTRA_ENV[@]}" \
    isaac-sim \
    /isaac-sim/python.sh /isaac_envs/sim_data_collector.py \
        --env "$ENV_TYPE" \
        --num-episodes "$NUM_EPISODES" \
        --output-dir /data/episodes \
        $NO_CAMERA \
        $WAIT_FOR_KEY

echo "=== Done. Episodes saved to data/episodes/ ==="
if [[ -z "$NO_CAMERA" ]]; then
    echo "Videos (wrist + third_person): data/episodes/videos/"
fi
echo "Next: ./scripts/train.sh"
