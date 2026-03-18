#!/bin/bash
set -e

POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-pi0_fast_droid}"
CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-}"
PORT="${OPENPI_PORT:-8000}"

CMD_ARGS="--port $PORT"

if [ -n "$CHECKPOINT_DIR" ] && [ -d "$CHECKPOINT_DIR" ]; then
    echo "Serving policy: config=$POLICY_CONFIG checkpoint=$CHECKPOINT_DIR"
    CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$CHECKPOINT_DIR"
else
    echo "Serving default policy: config=$POLICY_CONFIG"
    CMD_ARGS="$CMD_ARGS --env DROID"
fi

echo "Starting OpenPi server on port $PORT..."
exec uv run scripts/serve_policy.py $CMD_ARGS
