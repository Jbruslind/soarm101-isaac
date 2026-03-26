#!/bin/bash
set -e

POLICY_CONFIG="${OPENPI_POLICY_CONFIG:-soarm_pi0_fast}"
CHECKPOINT_DIR="${OPENPI_CHECKPOINT_DIR:-/models/soarm_lora}"
BATCH_SIZE="${TRAINING_BATCH_SIZE:-1}"
GRAD_ACCUM="${TRAINING_GRAD_ACCUM:-8}"
LORA_RANK="${TRAINING_LORA_RANK:-32}"
QUANTIZE="${TRAINING_QUANTIZE_BASE:-true}"

echo "=== SO-ARM101 Training ==="
echo "Config:     $POLICY_CONFIG"
echo "Output:     $CHECKPOINT_DIR"
echo "Batch size: $BATCH_SIZE (grad accum: $GRAD_ACCUM)"
echo "LoRA rank:  $LORA_RANK"
echo "Quantize:   $QUANTIZE"

# Copy custom configs into OpenPi source tree if mounted
if [ -d "/training/configs" ]; then
    cp -r /training/configs/*.py /app/openpi/src/openpi/training/ 2>/dev/null || true
fi

cd /app/openpi
uv run python <<'PY'
import openpi.training.config as cfg
from openpi.training import soarm_config as sc

for train_cfg in sc.SOARM_TRAIN_CONFIGS:
    if train_cfg.name not in cfg._CONFIGS_DICT:
        cfg._CONFIGS.append(train_cfg)
        cfg._CONFIGS_DICT[train_cfg.name] = train_cfg
PY

mkdir -p "$CHECKPOINT_DIR"

exec uv run scripts/train.py \
    policy:checkpoint \
    --policy.config="$POLICY_CONFIG" \
    --policy.dir="$CHECKPOINT_DIR" \
    --training.batch_size="$BATCH_SIZE" \
    --training.gradient_accumulation_steps="$GRAD_ACCUM"
