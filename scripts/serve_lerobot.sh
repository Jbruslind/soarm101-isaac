#!/usr/bin/env bash
# Serve a fine-tuned LeRobot policy (ACT or SmolVLA) on the OpenPi websocket protocol.
#
# Starts training/scripts/lerobot_policy_server.py in the soarm-lerobot image (compose service
# lerobot-server, host network). The ros2-bridge, interactive_inference.py and
# tutorials/fasteners/part1/07_run_policy.py connect to ws://<host>:<port>.
# Works on any GPU: it runs on the GPU when there is room next to whatever is already there
# (e.g. Isaac Sim), otherwise on the CPU. See docs/INTERACTIVE_INFERENCE.md.
#
# Usage:
#   ./scripts/serve_lerobot.sh --checkpoint data/checkpoints/<job> [options]
#
# Options:
#   --port P              websocket port (default 8000)
#   --device D            auto (default), cuda, cuda:1, cpu
#   --n-action-steps N    actions returned per request = how often the policy re-plans
#                         (default 10; 0 = the value it was trained with)
#   --flow-steps K        SmolVLA denoising steps (default: trained value, 10); fewer is faster
#   --detach              run in the background; stop with: docker rm -f soarm-lerobot-server
#
# Examples:
#   ./scripts/serve_lerobot.sh --checkpoint data/checkpoints/smolvla_fastener_m6_10k
#   ./scripts/serve_lerobot.sh --checkpoint data/checkpoints/act_fastener_m6_10k --n-action-steps 25 --detach
#   curl -s http://127.0.0.1:8000/healthz     # -> OK once the policy is loaded
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
# shellcheck source=_gpu.sh
source "$SCRIPT_DIR/_gpu.sh"

CHECKPOINT=""
PORT=8000
DEVICE=auto
N_ACTION_STEPS=10
FLOW_STEPS=0
DETACH=0

usage() { sed -n '2,25p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        --port) PORT="$2"; shift 2 ;;
        --device) DEVICE="$2"; shift 2 ;;
        --n-action-steps) N_ACTION_STEPS="$2"; shift 2 ;;
        --flow-steps) FLOW_STEPS="$2"; shift 2 ;;
        --detach) DETACH=1; shift ;;
        -h|--help) usage 0 ;;
        *) echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$CHECKPOINT" ]] || { echo "--checkpoint is required" >&2; usage 1; }
[[ -d "$CHECKPOINT" ]] || { echo "Checkpoint not found: $CHECKPOINT" >&2; exit 1; }
CK_ABS="$(cd "$CHECKPOINT" && pwd)"
case "$CK_ABS/" in
    "$PROJECT_DIR/data/"*) ;;
    *) echo "Checkpoints must be under $PROJECT_DIR/data/ (got $CK_ABS)" >&2; exit 1 ;;
esac
CK_IN="/workspace/data/${CK_ABS#"$PROJECT_DIR/data/"}"

if read -r GPU_TOTAL GPU_FREE < <(gpu_mem_mib); then
    echo "GPU: $(gpu_name) - $(mib_to_gb "$GPU_FREE") GB free of $(mib_to_gb "$GPU_TOTAL") GB (device=$DEVICE)"
fi
echo "=== LeRobot policy server: $CHECKPOINT on ws://0.0.0.0:$PORT, n_action_steps=$N_ACTION_STEPS ==="

RUN_ARGS=(--rm --name soarm-lerobot-server -e "SOARM_POLICY_PORT=$PORT")
[[ $DETACH -eq 1 ]] && RUN_ARGS+=(-d)

cd "$PROJECT_DIR/docker"
SOARM_UID="$(id -u)" SOARM_GID="$(id -g)" \
    docker compose --profile serve-lerobot run "${RUN_ARGS[@]}" lerobot-server \
    python /training/scripts/lerobot_policy_server.py --checkpoint "$CK_IN" --port "$PORT" \
        --device "$DEVICE" --n-action-steps "$N_ACTION_STEPS" --flow-steps "$FLOW_STEPS"

if [[ $DETACH -eq 1 ]]; then
    echo "Started in the background. Logs: docker logs -f soarm-lerobot-server"
    echo "Ready when: curl -s http://127.0.0.1:$PORT/healthz  prints OK"
fi
