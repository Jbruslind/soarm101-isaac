#!/usr/bin/env bash
# Deploy the OpenPi inference server to a remote GPU machine.
#
# Automatically detects whether the target is an NVIDIA Jetson (ARM64 / L4T)
# or a standard x86_64 GPU server and uses the appropriate Dockerfile and
# compose file.
#
# Usage:
#   ./scripts/deploy.sh user@gpu-server            # auto-detect arch
#   ./scripts/deploy.sh user@jetson --jetson        # force Jetson mode
#   ./scripts/deploy.sh user@jetson --l4t r36.3.0   # Jetson with specific L4T
#   ./scripts/deploy.sh user@cloud  --x86           # force x86 mode
#   ./scripts/deploy.sh user@host   --setup-keys    # install SSH key, then deploy
#
# Prerequisites on remote host:
#   x86:   Docker Engine 26.0+ with NVIDIA Container Toolkit
#   Jetson: JetPack 5.1+ (Xavier) or 6.0+ (Orin), Docker ships with JetPack
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

REMOTE_HOST=""
FORCE_TARGET=""      # "", "jetson", or "x86"
L4T_TAG="r35.4.1"
REMOTE_DIR="~/soarm"
SETUP_KEYS=0

while [[ $# -gt 0 ]]; do
    case $1 in
        --jetson)     FORCE_TARGET="jetson"; shift ;;
        --x86)        FORCE_TARGET="x86";    shift ;;
        --l4t)        L4T_TAG="$2"; FORCE_TARGET="jetson"; shift 2 ;;
        --remote-dir) REMOTE_DIR="$2"; shift 2 ;;
        --setup-keys) SETUP_KEYS=1; shift ;;
        -h|--help)
            echo "Usage: deploy.sh user@host [--jetson|--x86] [--l4t TAG] [--remote-dir DIR] [--setup-keys]"
            exit 0 ;;
        -*)         echo "Unknown option: $1"; exit 1 ;;
        *)          REMOTE_HOST="$1"; shift ;;
    esac
done

if [[ -z "$REMOTE_HOST" ]]; then
    echo "Usage: deploy.sh user@host [--jetson|--x86] [--l4t TAG]"
    exit 1
fi

HOSTNAME=$(echo "$REMOTE_HOST" | cut -d@ -f2)

# ---------------------------------------------------------------------------
# SSH key setup (--setup-keys): one-time passwordless access
# ---------------------------------------------------------------------------
if [[ $SETUP_KEYS -eq 1 ]]; then
    if [[ ! -f "$HOME/.ssh/id_ed25519.pub" ]] && [[ ! -f "$HOME/.ssh/id_rsa.pub" ]]; then
        echo "No SSH key found. Generating one..."
        ssh-keygen -t ed25519 -f "$HOME/.ssh/id_ed25519" -N "" -C "deploy@$(hostname)"
    fi
    echo "Copying SSH key to $REMOTE_HOST (you'll be prompted for the password ONE last time)..."
    ssh-copy-id "$REMOTE_HOST"
    echo "SSH key installed. Future connections will be passwordless."
    echo ""
fi

# ---------------------------------------------------------------------------
# SSH ControlMaster: authenticate once, reuse for all ssh/rsync calls
# ---------------------------------------------------------------------------
SSH_CONTROL_DIR=$(mktemp -d)
SSH_CONTROL_SOCKET="$SSH_CONTROL_DIR/ctrl-%r@%h:%p"

cleanup_ssh() {
    command ssh -o ControlPath="$SSH_CONTROL_SOCKET" -O exit "$REMOTE_HOST" 2>/dev/null || true
    rm -rf "$SSH_CONTROL_DIR"
}
trap cleanup_ssh EXIT

SSH_MUX="-o ControlMaster=auto -o ControlPath=$SSH_CONTROL_SOCKET -o ControlPersist=600"

# Open the master connection (authenticates once)
command ssh $SSH_MUX -fN "$REMOTE_HOST"
echo "SSH session established (multiplexed -- password will not be asked again)."

# Override ssh so every call in this script reuses the master socket
ssh() { command ssh $SSH_MUX "$@"; }

# Tell rsync to use the same multiplexed connection
export RSYNC_RSH="ssh $SSH_MUX"

# rsync flags: recursive, links, compress, progress -- but NOT
# owner/group/perms (-a implies -ogp which fails when dirs are root-owned
# from prior Docker runs).  --omit-dir-times avoids "failed to set times"
# on directories we don't own.
RSYNC_FLAGS="-rlz --omit-dir-times --progress --chmod=ugo=rwX"

# ---------------------------------------------------------------------------
# Fix remote directory ownership (Docker may leave root-owned dirs)
# ---------------------------------------------------------------------------
fix_remote_perms() {
    local target="$1"

    # Check if we can already write to the directory
    if ssh "$REMOTE_HOST" "test -w $target" 2>/dev/null; then
        return 0
    fi

    echo "  Fixing permissions on $target (owned by root from previous Docker run)..."

    # Try 1: passwordless sudo (-n = non-interactive, fails immediately if
    #         a password is needed rather than hanging)
    if ssh "$REMOTE_HOST" "sudo -n chown -R \$(id -u):\$(id -g) $target" 2>/dev/null; then
        return 0
    fi

    # Try 2: sudo with a TTY (allocate pseudo-terminal so sudo can prompt)
    if command ssh $SSH_MUX -t "$REMOTE_HOST" "sudo chown -R \$(id -u):\$(id -g) $target" 2>/dev/null; then
        return 0
    fi

    # Try 3: delete and recreate (works if the PARENT dir is user-owned)
    if ssh "$REMOTE_HOST" "rm -rf $target && mkdir -p $target" 2>/dev/null; then
        return 0
    fi

    echo ""
    echo "ERROR: Cannot fix permissions on $REMOTE_HOST:$target"
    echo "Please SSH into the remote machine and run:"
    echo "  sudo chown -R \$(id -u):\$(id -g) ~/soarm"
    echo "Then re-run this script."
    exit 1
}

# ---------------------------------------------------------------------------
# Detect remote platform
# ---------------------------------------------------------------------------
detect_target() {
    if [[ -n "$FORCE_TARGET" ]]; then
        echo "$FORCE_TARGET"
        return
    fi

    echo "Auto-detecting remote platform..." >&2

    if ssh "$REMOTE_HOST" "test -f /etc/nv_tegra_release" 2>/dev/null; then
        echo "jetson"
    else
        local arch
        arch=$(ssh "$REMOTE_HOST" "uname -m" 2>/dev/null || echo "unknown")
        if [[ "$arch" == "aarch64" ]]; then
            # aarch64 with L4T files = Jetson
            if ssh "$REMOTE_HOST" "dpkg -l nvidia-l4t-core >/dev/null 2>&1" 2>/dev/null; then
                echo "jetson"
            else
                echo "x86"
            fi
        else
            echo "x86"
        fi
    fi
}

TARGET=$(detect_target)
echo "=== Deploying OpenPi to $REMOTE_HOST [target: $TARGET] ==="

# ---------------------------------------------------------------------------
# Verify prerequisites
# ---------------------------------------------------------------------------
echo "[0] Checking prerequisites on $REMOTE_HOST..."
ssh "$REMOTE_HOST" "docker info 2>/dev/null | grep -qi nvidia" || {
    echo "ERROR: nvidia-container-runtime not found on $REMOTE_HOST."
    echo "Install: sudo apt-get install nvidia-container-toolkit"
    echo "Then:    sudo nvidia-ctk runtime configure --runtime=docker && sudo systemctl restart docker"
    exit 1
}

# ---------------------------------------------------------------------------
# Jetson deploy
# ---------------------------------------------------------------------------
deploy_jetson() {
    echo ""
    echo "Platform: NVIDIA Jetson (L4T $L4T_TAG)"
    echo ""

    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR/{docker/openpi-server,docker/training/configs,models,data/episodes}"

    # Fix any root-owned directories left by previous Docker runs
    fix_remote_perms "$REMOTE_DIR/docker"

    echo "[1/5] Syncing Jetson Dockerfile & compose..."
    rsync $RSYNC_FLAGS \
        "$PROJECT_DIR/docker/docker-compose.jetson.yml" \
        "$REMOTE_HOST:$REMOTE_DIR/docker/"

    rsync $RSYNC_FLAGS \
        "$PROJECT_DIR/docker/openpi-server/Dockerfile.jetson" \
        "$PROJECT_DIR/docker/openpi-server/entrypoint.jetson.sh" \
        "$PROJECT_DIR/docker/openpi-server/Caddyfile" \
        "$REMOTE_HOST:$REMOTE_DIR/docker/openpi-server/"

    echo "[2/5] Writing .env..."
    ssh "$REMOTE_HOST" "cat > $REMOTE_DIR/docker/.env" <<EOF
L4T_TAG=$L4T_TAG
OPENPI_PORT=8000
OPENPI_POLICY_CONFIG=soarm_pi0_fast
NVIDIA_VISIBLE_DEVICES=all
EOF

    echo "[3/5] Syncing training configs..."
    if [[ -d "$PROJECT_DIR/training/configs" ]]; then
        rsync $RSYNC_FLAGS \
            "$PROJECT_DIR/training/configs/" \
            "$REMOTE_HOST:$REMOTE_DIR/docker/training/configs/"
    fi

    echo "[4/5] Building container on Jetson (first build may take 20-40 min)..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR/docker && \
        docker compose -f docker-compose.jetson.yml build \
            --build-arg L4T_TAG=$L4T_TAG"

    echo "[5/5] Starting OpenPi server + Caddy..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR/docker && \
        docker compose -f docker-compose.jetson.yml up -d"

    echo ""
    echo "=== Jetson deployment complete ==="
    echo "OpenPi server: http://$HOSTNAME:8000  (direct)"
    echo "OpenPi server: wss://$HOSTNAME:8443   (via Caddy TLS)"
    echo ""
    echo "Check logs:"
    echo "  ssh $REMOTE_HOST 'cd $REMOTE_DIR/docker && docker compose -f docker-compose.jetson.yml logs -f'"
}

# ---------------------------------------------------------------------------
# x86 / cloud deploy
# ---------------------------------------------------------------------------
deploy_x86() {
    echo ""
    echo "Platform: x86_64 GPU server"
    echo ""

    ssh "$REMOTE_HOST" "mkdir -p $REMOTE_DIR/{docker/openpi-server,docker/training,training/configs,models,data/episodes}"

    fix_remote_perms "$REMOTE_DIR/docker"

    echo "[1/4] Syncing Docker files..."
    rsync $RSYNC_FLAGS \
        "$PROJECT_DIR/docker/docker-compose.cloud.yml" \
        "$REMOTE_HOST:$REMOTE_DIR/docker/"

    rsync $RSYNC_FLAGS \
        "$PROJECT_DIR/docker/.env.cloud" \
        "$REMOTE_HOST:$REMOTE_DIR/docker/.env" 2>/dev/null || true

    rsync $RSYNC_FLAGS \
        "$PROJECT_DIR/docker/openpi-server/" \
        "$REMOTE_HOST:$REMOTE_DIR/docker/openpi-server/"

    rsync $RSYNC_FLAGS \
        "$PROJECT_DIR/docker/training/" \
        "$REMOTE_HOST:$REMOTE_DIR/docker/training/"

    echo "[2/4] Syncing training configs..."
    if [[ -d "$PROJECT_DIR/training/configs" ]]; then
        rsync $RSYNC_FLAGS \
            "$PROJECT_DIR/training/configs/" \
            "$REMOTE_HOST:$REMOTE_DIR/training/configs/"
    fi

    echo "[3/4] Building containers on remote..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR/docker && docker compose -f docker-compose.cloud.yml build"

    echo "[4/4] Starting OpenPi server..."
    ssh "$REMOTE_HOST" "cd $REMOTE_DIR/docker && docker compose -f docker-compose.cloud.yml up -d openpi-server caddy"

    echo ""
    echo "=== Cloud deployment complete ==="
    echo "OpenPi server: wss://$HOSTNAME:8443"
}

# ---------------------------------------------------------------------------
# Dispatch
# ---------------------------------------------------------------------------
case "$TARGET" in
    jetson) deploy_jetson ;;
    x86)    deploy_x86    ;;
    *)      echo "ERROR: Unknown target '$TARGET'"; exit 1 ;;
esac

echo ""
echo "To use remote inference locally, update docker/.env:"
echo "  OPENPI_HOST=$HOSTNAME"
echo "  OPENPI_PORT=8443"
echo ""
echo "To sync data:  ./scripts/sync_data.sh push $REMOTE_HOST"
echo "To train:      ./scripts/train.sh --remote $REMOTE_HOST"
