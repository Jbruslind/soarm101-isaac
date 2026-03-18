#!/usr/bin/env bash
# Launch LoRA fine-tuning of a VLA model on collected episodes.
#
# Mode A (local 3080 Ti): runs with QLoRA + CPU offloading
# Mode B (remote cloud):  runs with full precision
#
# Usage:
#   ./scripts/train.sh                    # local training
#   ./scripts/train.sh --remote user@host # run on remote GPU machine
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

if [ -n "$REMOTE_HOST" ]; then
    echo "=== Remote training on $REMOTE_HOST ==="

    # Sync data to cloud first
    "$SCRIPT_DIR/sync_data.sh" push "$REMOTE_HOST"

    # Run training on cloud
    ssh "$REMOTE_HOST" "cd ~/soarm/docker && \
        docker compose -f docker-compose.cloud.yml run --rm training"

    # Pull checkpoints back
    "$SCRIPT_DIR/sync_data.sh" pull "$REMOTE_HOST"
else
    echo "=== Local training (3080 Ti mode) ==="
    echo "Using QLoRA with CPU offloading for 12GB VRAM"

    cd "$PROJECT_DIR/docker"
    docker compose --profile train run --rm training
fi

echo "=== Training complete ==="
echo "Checkpoints saved to models/"
echo "Next: ./scripts/eval_sim.sh"
