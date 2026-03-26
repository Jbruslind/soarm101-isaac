#!/bin/bash
set -e

POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-soarm_pi0_fast}"
CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-}"
PORT="${OPENPI_PORT:-8000}"
POLICY_MODE="${OPENPI_POLICY_MODE:-soarm}"

CMD_ARGS="--port $PORT"

case "$POLICY_MODE" in
    soarm)
        # SOARM mode: require a SOARM-named OpenPi config. Local LoRA (OPENPI_CHECKPOINT_DIR) is
        # optional before fine-tuning — if missing, use OpenPi's public base weights that match
        # the model family (same approach as training's weight_loader in upstream OpenPi).
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
                    soarm_pi0_fast)
                        BASE_DIR="gs://openpi-assets/checkpoints/pi0_fast_base"
                        ;;
                    soarm_pi0)
                        BASE_DIR="gs://openpi-assets/checkpoints/pi0_base"
                        ;;
                    *)
                        echo "ERROR: No local OPENPI_CHECKPOINT_DIR and cannot infer base checkpoint for OPENPI_POLICY_CONFIG='$POLICY_CONFIG'."
                        echo "       Fine-tune and mount a checkpoint, set OPENPI_BASE_CHECKPOINT_DIR, or use soarm_pi0 / soarm_pi0_fast."
                        exit 2
                        ;;
                esac
            fi
            echo "WARN: OPENPI_CHECKPOINT_DIR missing or not a directory ('${CHECKPOINT_DIR:-<empty>}')."
            echo "      Using pretrained base weights (no LoRA yet): $BASE_DIR"
            CMD_ARGS="$CMD_ARGS policy:checkpoint --policy.config=$POLICY_CONFIG --policy.dir=$BASE_DIR"
        fi
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
