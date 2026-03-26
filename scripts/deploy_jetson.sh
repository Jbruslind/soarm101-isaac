#!/usr/bin/env bash
# Deploy the OpenPi inference server to an NVIDIA Jetson device.
#
# Copies Dockerfiles, configs, and compose file to the Jetson, builds
# using the L4T-compatible Dockerfile, and starts the server.
#
# Prerequisites on the Jetson:
#   - JetPack 5.1+ (Xavier) or JetPack 6.0+ (Orin)
#   - Docker with nvidia-container-runtime (ships with JetPack)
#   - SSH access (key-based recommended)
#
# Usage:
#   ./scripts/deploy_jetson.sh user@jetson-host
#   ./scripts/deploy_jetson.sh user@jetson-host --l4t r36.3.0   # Orin / JP6
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

JETSON_HOST=""
L4T_TAG="r35.4.1"

while [[ $# -gt 0 ]]; do
    case $1 in
        --l4t)  L4T_TAG="$2"; shift 2 ;;
        -*)     echo "Unknown option: $1"; exit 1 ;;
        *)      JETSON_HOST="$1"; shift ;;
    esac
done

if [[ -z "$JETSON_HOST" ]]; then
    echo "Usage: deploy_jetson.sh user@jetson-host [--l4t r35.4.1]"
    exit 1
fi

REMOTE_DIR="~/soarm"

echo "=== Deploying OpenPi to Jetson: $JETSON_HOST (L4T $L4T_TAG) ==="

# Verify Jetson has nvidia runtime
echo "[0/5] Checking Jetson prerequisites..."
ssh "$JETSON_HOST" "docker info 2>/dev/null | grep -qi nvidia" || {
    echo "ERROR: nvidia-container-runtime not found on $JETSON_HOST."
    echo "Install it with: sudo apt-get install nvidia-container-toolkit"
    echo "Then: sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    exit 1
}

# Create remote directory structure (include openpi-server/scripts for Python 3.9–compatible serve_policy.py)
ssh "$JETSON_HOST" "mkdir -p $REMOTE_DIR/{docker/openpi-server/scripts,docker/training/configs,models,data/episodes}"

# Copy Docker files
echo "[1/5] Syncing Jetson Dockerfile & compose..."
rsync -avz --progress \
    "$PROJECT_DIR/docker/docker-compose.jetson.yml" \
    "$JETSON_HOST:$REMOTE_DIR/docker/"

rsync -avz --progress \
    "$PROJECT_DIR/docker/openpi-server/Dockerfile.jetson" \
    "$PROJECT_DIR/docker/openpi-server/entrypoint.jetson.sh" \
    "$PROJECT_DIR/docker/openpi-server/Caddyfile" \
    "$PROJECT_DIR/docker/openpi-server/Caddyfile.jetson" \
    "$JETSON_HOST:$REMOTE_DIR/docker/openpi-server/"

# Python 3.9–compatible serve_policy.py (no match/case; required for L4T Python 3.8/3.9)
rsync -avz --progress \
    "$PROJECT_DIR/docker/openpi-server/scripts/serve_policy.py" \
    "$PROJECT_DIR/docker/openpi-server/scripts/preload_checkpoint.py" \
    "$PROJECT_DIR/docker/openpi-server/scripts/register_soarm_configs.py" \
    "$JETSON_HOST:$REMOTE_DIR/docker/openpi-server/scripts/"

# Write .env with L4T tag
echo "[2/5] Writing .env for Jetson..."
ssh "$JETSON_HOST" "cat > $REMOTE_DIR/docker/.env" <<EOF
L4T_TAG=$L4T_TAG
OPENPI_PORT=8000
OPENPI_POLICY_MODE=soarm
OPENPI_POLICY_CONFIG=soarm_pi0_fast
NVIDIA_VISIBLE_DEVICES=all
EOF

# Copy training configs (needed for custom policy configs)
echo "[3/5] Syncing training configs..."
if [[ -d "$PROJECT_DIR/training/configs" ]]; then
    rsync -avz --progress \
        "$PROJECT_DIR/training/configs/" \
        "$JETSON_HOST:$REMOTE_DIR/docker/training/configs/"
fi

# Build on Jetson (ARM64 native build -- may take a while)
echo "[4/5] Building container on Jetson (this may take 20-40 min first time)..."
ssh "$JETSON_HOST" "cd $REMOTE_DIR/docker && \
    docker compose -f docker-compose.jetson.yml build \
        --build-arg L4T_TAG=$L4T_TAG"

# Start services
echo "[5/5] Starting OpenPi server + Caddy..."
ssh "$JETSON_HOST" "cd $REMOTE_DIR/docker && \
    docker compose -f docker-compose.jetson.yml up -d"

HOSTNAME=$(echo "$JETSON_HOST" | cut -d@ -f2)
echo ""
echo "=== Jetson deployment complete ==="
echo "OpenPi server: http://$HOSTNAME:8000  (direct)"
echo "OpenPi server: wss://$HOSTNAME:8443   (via Caddy TLS)"
echo ""
echo "To use Jetson inference from your local machine, update docker/.env:"
echo "  OPENPI_HOST=$HOSTNAME"
echo "  OPENPI_PORT=8443"
echo ""
echo "Check logs:  ssh $JETSON_HOST 'cd $REMOTE_DIR/docker && docker compose -f docker-compose.jetson.yml logs -f'"
