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

cd /app/openpi
export PYTHONPATH="/app/openpi/src${PYTHONPATH:+:$PYTHONPATH}"
# JAX: prefer CUDA, fall back to CPU if no CUDA (avoids AssertionError when jaxlib has no GPU on Jetson)
export JAX_PLATFORMS=cuda,cpu

# Preload default checkpoint so the first inference isn't delayed by a multi-GB download
if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Preloading default policy checkpoint (DROID / pi05_droid)..."
    if command -v uv >/dev/null 2>&1; then
        uv run --python 3.11 scripts/preload_checkpoint.py || true
    else
        python3.11 scripts/preload_checkpoint.py || true
    fi
fi

echo "Starting OpenPi server on port $PORT (Jetson)..."
if command -v uv >/dev/null 2>&1; then
  exec uv run --python 3.11 scripts/serve_policy.py $CMD_ARGS
fi
exec python3.11 scripts/serve_policy.py $CMD_ARGS
