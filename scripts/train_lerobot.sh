#!/usr/bin/env bash
# Fine-tune a LeRobot policy (SmolVLA or ACT) in the soarm-lerobot container.
#
# Wraps `lerobot-train` (lerobot 0.6.1) with TensorBoard logging; checkpoints and
# logs go to data/checkpoints/<job>/. Needs the whole 12 GB GPU: stop Isaac Sim first.
# See docs/LEROBOT_TRAINING.md.
#
# Usage:
#   ./scripts/train_lerobot.sh --policy {act,smolvla} --dataset <hub repo_id | local dir> [options] [-- extra lerobot-train args]
#
# Options:
#   --steps N           training steps (default 20000)
#   --batch B           batch size (default: smolvla 16, act 8)
#   --workers W         dataloader workers (default 6)
#   --amp               ACT only: fp16 autocast + grad scaling. SmolVLA already trains
#                       in bf16 (smolvla_base weights are bf16) and rejects --amp.
#   --rename-map JSON   map dataset camera keys onto the policy's, e.g.
#                       '{"observation.images.up":"observation.images.camera1"}'
#                       (SmolVLA base expects observation.images.camera1..3)
#   --job-name NAME     run name (default <policy>_<dataset>_<timestamp>)
#   --save-freq N       checkpoint every N steps (default 5000; final step always saved)
#   --log-freq N        log every N steps (default 100)
#   --resume            resume data/checkpoints/<job-name> from its last checkpoint
#   --force             train even if an Isaac Sim container is running
#
# Examples:
#   ./scripts/train_lerobot.sh --policy act --dataset lerobot/svla_so101_pickplace --steps 1000
#   ./scripts/train_lerobot.sh --policy smolvla --dataset lerobot/svla_so101_pickplace \
#       --rename-map '{"observation.images.up":"observation.images.camera1","observation.images.side":"observation.images.camera2"}'
#   ./scripts/train_lerobot.sh --policy smolvla --dataset data/lerobot/m6_bolt --steps 20000
#   tensorboard --logdir data/checkpoints        # or run it in the container, see the doc
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

POLICY=""
DATASET=""
STEPS=20000
BATCH=""
WORKERS=6
AMP=0
RENAME_MAP=""
JOB_NAME=""
SAVE_FREQ=5000
LOG_FREQ=100
RESUME=0
FORCE=0
EXTRA=()

usage() { sed -n '2,30p' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

while [[ $# -gt 0 ]]; do
    case $1 in
        --policy) POLICY="$2"; shift 2 ;;
        --dataset) DATASET="$2"; shift 2 ;;
        --steps) STEPS="$2"; shift 2 ;;
        --batch) BATCH="$2"; shift 2 ;;
        --workers) WORKERS="$2"; shift 2 ;;
        --amp) AMP=1; shift ;;
        --rename-map) RENAME_MAP="$2"; shift 2 ;;
        --job-name) JOB_NAME="$2"; shift 2 ;;
        --save-freq) SAVE_FREQ="$2"; shift 2 ;;
        --log-freq) LOG_FREQ="$2"; shift 2 ;;
        --resume) RESUME=1; shift ;;
        --force) FORCE=1; shift ;;
        -h|--help) usage 0 ;;
        --) shift; EXTRA=("$@"); break ;;
        *) echo "Unknown arg: $1" >&2; usage 1 ;;
    esac
done

[[ -n "$POLICY" && -n "$DATASET" ]] || { echo "--policy and --dataset are required" >&2; usage 1; }

if [[ "$POLICY" == smolvla && $AMP -eq 1 ]]; then
    # lerobot 0.6.1: use_amp means fp16 autocast + GradScaler, and the GradScaler cannot
    # unscale SmolVLA's bf16 gradients (NotImplementedError ... for 'BFloat16').
    echo "--amp is not supported for SmolVLA: it already trains in bf16." >&2
    exit 1
fi

case "$POLICY" in
    smolvla) POLICY_ARGS=(--policy.path=lerobot/smolvla_base); BATCH="${BATCH:-16}" ;;
    act)     POLICY_ARGS=(--policy.type=act); BATCH="${BATCH:-8}" ;;
    *) echo "--policy must be act or smolvla" >&2; exit 1 ;;
esac

# Dataset: a local directory (must live under data/, which is mounted at
# /workspace/data) or a Hugging Face Hub repo id (cached in the lerobot-hf-cache volume).
if [[ -d "$DATASET" ]]; then
    DS_ABS="$(cd "$DATASET" && pwd)"
    case "$DS_ABS/" in
        "$PROJECT_DIR/data/"*) ;;
        *) echo "Local datasets must be under $PROJECT_DIR/data/ (got $DS_ABS)" >&2; exit 1 ;;
    esac
    DS_NAME="$(basename "$DS_ABS")"
    DATASET_ARGS=(--dataset.repo_id="local/$DS_NAME" --dataset.root="/workspace/data/${DS_ABS#"$PROJECT_DIR/data/"}")
else
    DS_NAME="${DATASET##*/}"
    DATASET_ARGS=(--dataset.repo_id="$DATASET")
fi

JOB_NAME="${JOB_NAME:-${POLICY}_${DS_NAME}_$(date +%Y%m%d-%H%M%S)}"
OUT_DIR="/workspace/data/checkpoints/$JOB_NAME"

if [[ $FORCE -eq 0 ]] && docker ps --format '{{.Names}}' | grep -qx 'soarm-isaac-sim'; then
    echo "Isaac Sim (soarm-isaac-sim) is running. Training needs the whole 12 GB GPU." >&2
    echo "Stop it first (docker stop soarm-isaac-sim) or pass --force." >&2
    exit 1
fi

mkdir -p "$PROJECT_DIR/data/checkpoints"

if [[ $RESUME -eq 1 ]]; then
    ARGS=(--config_path="$OUT_DIR/checkpoints/last/pretrained_model/train_config.json" --resume=true)
else
    ARGS=(
        "${POLICY_ARGS[@]}"
        "${DATASET_ARGS[@]}"
        --output_dir="$OUT_DIR"
        --job_name="$JOB_NAME"
        --batch_size="$BATCH"
        --steps="$STEPS"
        --num_workers="$WORKERS"
        --save_freq="$SAVE_FREQ"
        --log_freq="$LOG_FREQ"
        --policy.device=cuda
        --policy.push_to_hub=false
        --wandb.enable=true
    )
    [[ $AMP -eq 1 ]] && ARGS+=(--policy.use_amp=true)
    [[ -n "$RENAME_MAP" ]] && ARGS+=(--rename_map="$RENAME_MAP")
fi

echo "=== LeRobot training: policy=$POLICY dataset=$DATASET batch=$BATCH steps=$STEPS ==="
echo "Output: data/checkpoints/$JOB_NAME (TensorBoard: data/checkpoints/$JOB_NAME/tensorboard)"

cd "$PROJECT_DIR/docker"
SOARM_UID="$(id -u)" SOARM_GID="$(id -g)" \
    docker compose --profile train-lerobot run --rm lerobot \
    python /opt/soarm/lerobot_train_tb.py "${ARGS[@]}" "${EXTRA[@]}"
