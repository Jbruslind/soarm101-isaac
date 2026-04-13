#!/usr/bin/env bash
# Deploy the VLA policy to a real SO-ARM101 robot.
#
# Starts the OpenPi server (local or remote) and ROS2 bridge.
# The ROS2 bridge will connect to the real robot via serial/USB.
#
# Usage:
#   ./scripts/deploy_real.sh                          # local inference
#   ./scripts/deploy_real.sh --remote gpu-server.com  # remote inference
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

REMOTE_HOST=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --remote) REMOTE_HOST="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

cd "$PROJECT_DIR/docker"

if [ -n "$REMOTE_HOST" ]; then
    echo "=== Deploy to real robot with remote inference ($REMOTE_HOST) ==="
    echo "Make sure OpenPi is running on the remote host first:"
    echo "  ssh $REMOTE_HOST 'cd ~/soarm/docker && docker compose -f docker-compose.cloud.yml up -d openpi-server caddy'"
    echo ""
    OPENPI_HOST="$REMOTE_HOST" OPENPI_PORT=8443 \
        docker compose --profile deploy-remote up
else
    echo "=== Deploy to real robot with local inference ==="
    docker compose --profile deploy-local up
fi
