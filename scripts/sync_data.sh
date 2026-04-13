#!/usr/bin/env bash
# Bidirectional sync of episodes and model checkpoints between local and cloud.
#
# Usage:
#   ./scripts/sync_data.sh push user@gpu-server  # upload episodes to cloud
#   ./scripts/sync_data.sh pull user@gpu-server  # download checkpoints from cloud
#   ./scripts/sync_data.sh both user@gpu-server  # push episodes, pull checkpoints
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

ACTION="${1:?Usage: sync_data.sh [push|pull|both] user@host}"
CLOUD_HOST="${2:?Usage: sync_data.sh [push|pull|both] user@host}"
REMOTE_DIR="~/soarm"

push_data() {
    echo "=== Pushing episodes to $CLOUD_HOST ==="
    rsync -avz --progress \
        "$PROJECT_DIR/data/episodes/" \
        "$CLOUD_HOST:$REMOTE_DIR/data/episodes/"

    echo "=== Pushing training configs ==="
    rsync -avz --progress \
        "$PROJECT_DIR/training/configs/" \
        "$CLOUD_HOST:$REMOTE_DIR/training/configs/"
}

pull_data() {
    echo "=== Pulling model checkpoints from $CLOUD_HOST ==="
    rsync -avz --progress \
        "$CLOUD_HOST:$REMOTE_DIR/models/" \
        "$PROJECT_DIR/models/"
}

case "$ACTION" in
    push) push_data ;;
    pull) pull_data ;;
    both) push_data; pull_data ;;
    *) echo "Unknown action: $ACTION (use push, pull, or both)"; exit 1 ;;
esac

echo "=== Sync complete ==="
