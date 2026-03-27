#!/bin/bash
set -e

POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-soarm_pi0_fast}"
CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-}"
PORT="${OPENPI_PORT:-8000}"
POLICY_MODE="${OPENPI_POLICY_MODE:-soarm}"

CMD_ARGS="--port $PORT"

case "$POLICY_MODE" in
    soarm)
        if [[ "$POLICY_CONFIG" != soarm_* ]]; then
            echo "ERROR: OPENPI_POLICY_MODE=soarm requires OPENPI_POLICY_CONFIG to start with 'soarm_'."
            echo "       Got OPENPI_POLICY_CONFIG='$POLICY_CONFIG'."
            exit 2
        fi
        if [ -n "$CHECKPOINT_DIR" ] && [ -d "$CHECKPOINT_DIR" ]; then
            echo "Serving SOARM policy: config=$POLICY_CONFIG checkpoint=$CHECKPOINT_DIR"
            CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$CHECKPOINT_DIR"
        else
            BASE_DIR="${OPENPI_BASE_CHECKPOINT_DIR:-}"
            if [ -z "$BASE_DIR" ]; then
                case "$POLICY_CONFIG" in
                    soarm_pi0_fast|soarm_pi0_fast_bootstrap)
                        BASE_DIR="gs://openpi-assets/checkpoints/pi0_fast_base"
                        ;;
                    soarm_pi0|soarm_pi0_bootstrap)
                        BASE_DIR="gs://openpi-assets/checkpoints/pi0_base"
                        ;;
                    *)
                        echo "ERROR: No local OPENPI_CHECKPOINT_DIR and cannot infer base checkpoint for OPENPI_POLICY_CONFIG='$POLICY_CONFIG'."
                        echo "       Set OPENPI_BASE_CHECKPOINT_DIR or use soarm_pi0 / soarm_pi0_fast."
                        exit 2
                        ;;
                esac
            fi
            POLICY_FOR_RUN="$POLICY_CONFIG"
            case "$POLICY_CONFIG" in
                soarm_pi0_fast) POLICY_FOR_RUN="soarm_pi0_fast_bootstrap" ;;
                soarm_pi0) POLICY_FOR_RUN="soarm_pi0_bootstrap" ;;
            esac
            echo "WARN: OPENPI_CHECKPOINT_DIR missing or not a directory ('${CHECKPOINT_DIR:-<empty>}')."
            echo "      Using pretrained base weights (no LoRA yet): $BASE_DIR"
            echo "      OpenPi config for this run: $POLICY_FOR_RUN (Libero norm stats in base ckpt)"
            CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_FOR_RUN --policy.dir=$BASE_DIR"
        fi
        ;;
    droid)
        if [[ "$POLICY_CONFIG" == soarm_* ]]; then
            echo "ERROR: OPENPI_POLICY_MODE=droid cannot use SOARM config '$POLICY_CONFIG'."
            echo "       Use a built-in OpenPI config (e.g. pi0_fast_droid) or set OPENPI_POLICY_MODE=soarm."
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
        ;;
    *)
        echo "ERROR: Invalid OPENPI_POLICY_MODE='$POLICY_MODE' (expected 'soarm' or 'droid')."
        exit 2
        ;;
esac

cd /app/openpi
export PYTHONPATH="/app/openpi/src${PYTHONPATH:+:$PYTHONPATH}"
# JAX: prefer CUDA, fall back to CPU if no CUDA (avoids AssertionError when jaxlib has no GPU on Jetson)
export JAX_PLATFORMS=cuda,cpu

# Preload weights so the first inference isn't delayed by a multi-GB download
if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
    export OPENPI_POLICY_MODE="$POLICY_MODE"
    export OPENPI_POLICY_CONFIG="$POLICY_CONFIG"
    echo "Preloading policy weights (see scripts/preload_checkpoint.py)..."
    if command -v uv >/dev/null 2>&1; then
        uv run --python 3.11 scripts/preload_checkpoint.py || true
    else
        python3.11 scripts/preload_checkpoint.py || true
    fi
fi

echo "Starting OpenPi server on port $PORT (Jetson)..."
if command -v uv >/dev/null 2>&1; then
  exec uv run --python 3.11 scripts/serve_policy_with_soarm.py $CMD_ARGS
fi
exec python3.11 scripts/serve_policy_with_soarm.py $CMD_ARGS
