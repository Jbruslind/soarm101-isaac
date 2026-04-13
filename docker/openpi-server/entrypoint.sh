#!/bin/bash
set -e

POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-soarm_pi0_fast}"
CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-}"
PORT="${OPENPI_PORT:-8000}"
POLICY_MODE="${OPENPI_POLICY_MODE:-soarm}"

CMD_ARGS="--port $PORT"

case "$POLICY_MODE" in
    soarm)
        # SOARM mode is strict: require explicit SOARM config and checkpoint.
        if [[ "$POLICY_CONFIG" != soarm_* ]]; then
            echo "ERROR: OPENPI_POLICY_MODE=soarm requires OPENPI_POLICY_CONFIG to start with 'soarm_'."
            echo "       Got OPENPI_POLICY_CONFIG='$POLICY_CONFIG'."
            exit 2
        fi
        if [ -z "$CHECKPOINT_DIR" ] || [ ! -d "$CHECKPOINT_DIR" ]; then
            echo "ERROR: OPENPI_POLICY_MODE=soarm requires OPENPI_CHECKPOINT_DIR to exist."
            echo "       Got OPENPI_CHECKPOINT_DIR='${CHECKPOINT_DIR:-<empty>}'"
            echo "       Refusing implicit fallback to DROID."
            exit 2
        fi
        echo "Serving SOARM policy: config=$POLICY_CONFIG checkpoint=$CHECKPOINT_DIR"
        CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$CHECKPOINT_DIR"
        ;;
    droid)
        # DROID mode is explicit. Ignore local SOARM checkpoint/config naming.
        if [[ "$POLICY_CONFIG" == soarm_* ]]; then
            echo "ERROR: OPENPI_POLICY_MODE=droid cannot use SOARM config '$POLICY_CONFIG'."
            echo "       Use a built-in OpenPI config (e.g. pi0_fast_droid) or default --env DROID."
            exit 2
        fi
        if [ -n "$CHECKPOINT_DIR" ] && [ -d "$CHECKPOINT_DIR" ]; then
            echo "Serving DROID-compatible checkpoint policy: config=$POLICY_CONFIG checkpoint=$CHECKPOINT_DIR"
            CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$CHECKPOINT_DIR"
        else
            echo "Serving explicit DROID default policy (--env DROID)"
            CMD_ARGS="$CMD_ARGS --env DROID"
        fi
        ;;
    *)
        echo "ERROR: Invalid OPENPI_POLICY_MODE='$POLICY_MODE' (expected 'soarm' or 'droid')."
        exit 2
        ;;
esac

echo "Starting OpenPi server on port $PORT..."
exec uv run scripts/serve_policy.py $CMD_ARGS
