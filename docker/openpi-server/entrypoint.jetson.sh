#!/bin/bash
set -e

POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-pi0_fast_droid}"
CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-}"
PORT="${OPENPI_PORT:-8000}"
POLICY_MODE="${OPENPI_POLICY_MODE:-droid}"

CMD_ARGS="--port $PORT"

if [[ "$POLICY_MODE" != "droid" ]]; then
    echo "ERROR: Jetson entrypoint currently supports OPENPI_POLICY_MODE=droid only."
    echo "       Got OPENPI_POLICY_MODE='$POLICY_MODE'."
    exit 2
fi

if [ -n "$CHECKPOINT_DIR" ] && [ -d "$CHECKPOINT_DIR" ]; then
    # The `policy:checkpoint` path requires both params and norm stats under
    # `$CHECKPOINT_DIR/assets/<asset_id>/norm_stats.json`.
    # During early training exports, it's common to have params but missing
    # norm stats; in that case we fall back to a known-good cached checkpoint.
    REQUIRED_NORM_STATS="$CHECKPOINT_DIR/assets/droid/norm_stats.json"
    if [ -f "$REQUIRED_NORM_STATS" ]; then
        echo "Serving policy: config=$POLICY_CONFIG checkpoint=$CHECKPOINT_DIR"
        CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$CHECKPOINT_DIR"
    else
        FALLBACK_CACHE_DIR="/root/.cache/openpi/openpi-assets/checkpoints/${POLICY_CONFIG}"
        if [ -d "$FALLBACK_CACHE_DIR" ]; then
            echo "WARN: Checkpoint missing norm stats ($REQUIRED_NORM_STATS)."
            echo "      Falling back to cached checkpoint: $FALLBACK_CACHE_DIR"
            CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$FALLBACK_CACHE_DIR"
        else
            echo "WARN: Checkpoint missing norm stats and cache dir not found."
            echo "      Falling back to --env DROID."
            CMD_ARGS="$CMD_ARGS --env DROID"
        fi
    fi
else
    DEFAULT_CACHE_DIR="/root/.cache/openpi/openpi-assets/checkpoints/${POLICY_CONFIG}"
    if [ -d "$DEFAULT_CACHE_DIR" ]; then
        echo "Serving fallback DROID policy checkpoint: config=$POLICY_CONFIG checkpoint=$DEFAULT_CACHE_DIR"
        CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$DEFAULT_CACHE_DIR"
    else
        # Final fallback: use OpenPi's built-in DROID env (slower pi05_droid).
        echo "Serving default policy via --env DROID (cache missing for $POLICY_CONFIG)"
        CMD_ARGS="$CMD_ARGS --env DROID"
    fi
fi

cd /app/openpi
export PYTHONPATH="/app/openpi/src${PYTHONPATH:+:$PYTHONPATH}"
# JAX: prefer CUDA, fall back to CPU if no CUDA (avoids AssertionError when jaxlib has no GPU on Jetson)
export JAX_PLATFORMS=cuda,cpu

# Preload default checkpoint so the first inference isn't delayed by a multi-GB download
if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
    echo "Preloading DROID policy checkpoints (pi05_droid + pi0_fast_droid)..."
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
