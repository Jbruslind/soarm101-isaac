#!/usr/bin/env bash
# Evaluate a trained VLA policy in the Isaac Sim environment.
#
# Starts the OpenPi server (local or remote), Isaac Sim, and ROS2 bridge,
# then runs the VLA bridge node for closed-loop evaluation.
#
# Usage:
#   ./scripts/eval_sim.sh                          # local inference
#   ./scripts/eval_sim.sh --remote gpu-server.com  # remote inference
#   ./scripts/eval_sim.sh --watch                  # live-view via WebRTC
#   ./scripts/eval_sim.sh --watch --public-ip 192.168.1.50
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

REMOTE_HOST=""
WATCH=""
PUBLIC_IP=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --remote) REMOTE_HOST="$2"; shift 2 ;;
        --watch) WATCH=1; shift ;;
        --public-ip) PUBLIC_IP="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_DIR/docker"

WATCH_ENV=""
if [[ -n "$WATCH" ]]; then
    if [[ -z "$PUBLIC_IP" ]]; then
        PUBLIC_IP="$(hostname -I 2>/dev/null | awk '{print $1}')" || PUBLIC_IP="127.0.0.1"
    fi
    WATCH_ENV="LIVESTREAM=2 PUBLIC_IP=$PUBLIC_IP"
    echo ""
    echo "WebRTC streaming enabled (LIVESTREAM=2)."
    echo "  Open the Isaac Sim WebRTC Streaming Client."
    echo "  Server address: $PUBLIC_IP"
    echo ""
fi

if [ -n "$REMOTE_HOST" ]; then
    echo "=== Eval with remote inference ($REMOTE_HOST) ==="
    env $WATCH_ENV OPENPI_HOST="$REMOTE_HOST" OPENPI_PORT=8443 \
        docker compose --profile eval-remote up
else
    echo "=== Eval with local inference ==="
    env $WATCH_ENV \
        docker compose --profile eval-local up
fi
