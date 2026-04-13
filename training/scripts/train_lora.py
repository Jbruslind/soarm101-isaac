"""Convenience wrapper for launching OpenPi LoRA training.

Sets up environment variables and calls OpenPi's train.py with
the SO-ARM101 configuration. Handles both local (QLoRA on 3080 Ti)
and cloud (full-precision LoRA on A100) scenarios.

Usage:
    python train_lora.py --data-dir /data/episodes --output-dir /models/soarm_lora
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description="Launch SO-ARM101 LoRA training")
    parser.add_argument("--data-dir", default="/data/episodes")
    parser.add_argument("--output-dir", default="/models/soarm_lora")
    parser.add_argument("--config", default="soarm_pi0_fast")
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--grad-accum", type=int, default=None)
    parser.add_argument("--lora-rank", type=int, default=None)
    parser.add_argument("--quantize", action="store_true", default=None)
    parser.add_argument("--no-quantize", action="store_true")
    parser.add_argument("--max-steps", type=int, default=5000)
    args = parser.parse_args()

    batch_size = args.batch_size or int(os.environ.get("TRAINING_BATCH_SIZE", "1"))
    grad_accum = args.grad_accum or int(os.environ.get("TRAINING_GRAD_ACCUM", "8"))
    lora_rank = args.lora_rank or int(os.environ.get("TRAINING_LORA_RANK", "32"))

    if args.no_quantize:
        quantize = False
    elif args.quantize is not None:
        quantize = args.quantize
    else:
        quantize = os.environ.get("TRAINING_QUANTIZE_BASE", "true").lower() == "true"

    os.makedirs(args.output_dir, exist_ok=True)

    cmd = [
        "uv", "run", "scripts/train.py",
        "policy:checkpoint",
        f"--policy.config={args.config}",
        f"--policy.dir={args.output_dir}",
        f"--training.batch_size={batch_size}",
        f"--training.gradient_accumulation_steps={grad_accum}",
        f"--training.max_steps={args.max_steps}",
    ]

    print(f"Training config: {args.config}")
    print(f"  Batch size: {batch_size} (grad accum: {grad_accum})")
    print(f"  LoRA rank: {lora_rank}, Quantize: {quantize}")
    print(f"  Output: {args.output_dir}")
    print(f"  Max steps: {args.max_steps}")
    print()

    result = subprocess.run(cmd, cwd="/app/openpi")
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
