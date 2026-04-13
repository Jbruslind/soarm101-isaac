#!/usr/bin/env bash
# Deploy the GPU stack (OpenPi server + training) to a remote cloud machine.
#
# For NVIDIA Jetson, use deploy_jetson.sh instead; this script uses x86
# Dockerfiles and docker-compose.cloud.yml.
#
# Copies Dockerfiles, configs, and compose file to the remote host,
# builds containers, and starts the OpenPi inference server.
#
# Prerequisites on remote host:
#   - Docker Engine 26.0+ with NVIDIA Container Toolkit
#   - SSH access configured (key-based recommended)
#
# Usage:
#   ./scripts/deploy_cloud.sh user@gpu-server
#   ./scripts/deploy_cloud.sh user@gpu-server --force   # skip Jetson check
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

CLOUD_HOST=""
FORCE=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --force) FORCE=1; shift ;;
        -*) echo "Unknown option: $1"; exit 1 ;;
        *)  CLOUD_HOST="$1"; shift ;;
    esac
done

if [[ -z "$CLOUD_HOST" ]]; then
    echo "Usage: deploy_cloud.sh user@host [--force]"
    echo "  For Jetson use: ./scripts/deploy_jetson.sh user@jetson-host"
    exit 1
fi

REMOTE_DIR="~/soarm"

# Detect Jetson / ARM64 and suggest deploy_jetson.sh
ARCH=$(ssh "$CLOUD_HOST" "uname -m" 2>/dev/null || true)
if [[ "$ARCH" == "aarch64" && -z "$FORCE" ]]; then
    echo "ERROR: Remote $CLOUD_HOST is ARM64 (Jetson). Use the Jetson deploy script instead:"
    echo "  ./scripts/deploy_jetson.sh $CLOUD_HOST"
    echo "  ./scripts/deploy_jetson.sh $CLOUD_HOST --l4t r36.3.0   # for JetPack 6 / Orin"
    echo ""
    echo "To override and use cloud compose anyway (may fail or use emulation): --force"
    exit 1
fi

echo "=== Deploying GPU stack to $CLOUD_HOST ==="

# Create remote directory structure
ssh "$CLOUD_HOST" "mkdir -p $REMOTE_DIR/{docker/openpi-server,docker/training,training/configs,models,data/episodes}"

# Copy Docker files
echo "[1/4] Syncing Docker files..."
rsync -avz --progress \
    "$PROJECT_DIR/docker/docker-compose.cloud.yml" \
    "$CLOUD_HOST:$REMOTE_DIR/docker/"

rsync -avz --progress \
    "$PROJECT_DIR/docker/.env.cloud" \
    "$CLOUD_HOST:$REMOTE_DIR/docker/.env"

rsync -avz --progress \
    "$PROJECT_DIR/docker/openpi-server/" \
    "$CLOUD_HOST:$REMOTE_DIR/docker/openpi-server/"

rsync -avz --progress \
    "$PROJECT_DIR/docker/training/" \
    "$CLOUD_HOST:$REMOTE_DIR/docker/training/"

# Copy training configs
echo "[2/4] Syncing training configs..."
rsync -avz --progress \
    "$PROJECT_DIR/training/configs/" \
    "$CLOUD_HOST:$REMOTE_DIR/training/configs/"

# Build containers on remote
echo "[3/4] Building containers on remote..."
ssh "$CLOUD_HOST" "cd $REMOTE_DIR/docker && docker compose -f docker-compose.cloud.yml build"

# Start OpenPi server + Caddy
echo "[4/4] Starting OpenPi server..."
ssh "$CLOUD_HOST" "cd $REMOTE_DIR/docker && docker compose -f docker-compose.cloud.yml up -d openpi-server caddy"

HOSTNAME=$(echo "$CLOUD_HOST" | cut -d@ -f2)
echo ""
echo "=== Deployment complete ==="
echo "OpenPi server: wss://$HOSTNAME:8443"
echo ""
echo "To use remote inference locally, update your .env:"
echo "  OPENPI_HOST=$HOSTNAME"
echo "  OPENPI_PORT=8443"
echo ""
echo "To sync data:  ./scripts/sync_data.sh push $CLOUD_HOST"
echo "To train:      ./scripts/train.sh --remote $CLOUD_HOST"
