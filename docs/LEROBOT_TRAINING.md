# LeRobot Training (SmolVLA + ACT) on a 12 GB GPU

How to fine-tune LeRobot **SmolVLA** (from `lerobot/smolvla_base`) and train an **ACT**
baseline locally, and what to expect on an RTX A2000 12 GB. This is the training step of
the fastener tutorials (epic #40, issue #35).

**Short answer:** SmolVLA fine-tunes comfortably on 12 GB. Batch 16 peaks at 4.5 GB and
batch 32 at 7.8 GB; batch 64 runs out of memory. Throughput is about 20 samples/s whatever
the batch size, so 20k steps at batch 16 take about 4.5 h. ACT at full 640x480 resolution
needs more memory than SmolVLA: batch 8 peaks at 3.7 GB, batch 16 at 6.8 GB, and batch 32
runs out of memory.

---

## Stack

| Piece | Version |
|---|---|
| lerobot | **0.6.1** (PyPI, 2026-08-03), extras `smolvla,training` |
| Base image | `pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime` (pinned by digest): Ubuntu 24.04, Python 3.12, torch 2.11.0+cu128 |
| Video decode | torchcodec 0.11.1 + Ubuntu ffmpeg 6.1 (AV1 through libdav1d) |
| transformers / accelerate | 5.5.4 / 1.15.0 (resolved by pip within lerobot's bounds) |
| Logging | TensorBoard (`docker/lerobot/lerobot_train_tb.py`, see below) |

lerobot 0.6.1 caps torch at `<2.12`, so the base image stays on torch 2.11. Driver 570 or
newer runs the cu128 wheels. This host's 615 driver is fine.

Files:

- `docker/lerobot/Dockerfile`: the image (`soarm-lerobot:0.6.1`)
- `docker/lerobot/lerobot_train_tb.py`: runs the stock `lerobot-train` with a TensorBoard writer in place of W&B
- `docker/docker-compose.yml`: service `lerobot`, profile `train-lerobot`. It mounts `../data` at `/workspace/data`, keeps HF hub, LeRobot datasets and torchvision weights in the named volume `lerobot-hf-cache`, and sets `shm_size: 16g` for DataLoader workers.
- `scripts/train_lerobot.sh`: the wrapper you normally call

## Build

```bash
cd docker
docker compose --profile train-lerobot build lerobot      # ~3.5 GB pull + ~1 min pip install
```

## Train

> **Stop Isaac Sim first.** Training keeps the GPU at 96-99 % utilization and
> uses 4.5-8 GB at the recommended batch sizes. Isaac Sim uses another 4-6 GB, so running both
> risks running out of memory and slows both down. `train_lerobot.sh` refuses to start while
> the `soarm-isaac-sim` container is running unless you pass `--force`. The trained policy's
> *inference* (about 1.1 GB) can run next to Isaac Sim, see [Inference](#inference-for-36).

```bash
# ACT baseline
./scripts/train_lerobot.sh --policy act --dataset lerobot/svla_so101_pickplace --steps 20000 --batch 8

# SmolVLA fine-tune from lerobot/smolvla_base
./scripts/train_lerobot.sh --policy smolvla --dataset lerobot/svla_so101_pickplace \
    --steps 20000 --batch 16 \
    --rename-map '{"observation.images.up":"observation.images.camera1","observation.images.side":"observation.images.camera2"}'

# A local LeRobot v3.0 dataset (must be under data/)
./scripts/train_lerobot.sh --policy smolvla --dataset data/lerobot/<name> --rename-map '...'

# Resume an interrupted run (optionally raise the step budget)
./scripts/train_lerobot.sh --policy smolvla --dataset <same> --job-name <job> --resume -- --steps=30000
```

Options: `--steps`, `--batch` (default 16 for SmolVLA, 8 for ACT), `--workers` (default 6),
`--amp` (ACT only), `--rename-map`, `--job-name`, `--save-freq` (default 5000), `--log-freq`
(default 100), `--resume`, `--force`. Anything after `--` goes straight to `lerobot-train`, for
example `-- --save_checkpoint=false` or `-- --policy.num_expert_layers=...`.

What the wrapper passes to `lerobot-train` (verified against the 0.6.1 source):

| Flag | Why |
|---|---|
| `--policy.path=lerobot/smolvla_base` / `--policy.type=act` | SmolVLA fine-tunes from the base checkpoint. ACT starts from scratch with an ImageNet ResNet-18 backbone. |
| `--dataset.repo_id` (+ `--dataset.root` for local data) | A local dir gets `repo_id=local/<dirname>` |
| `--policy.push_to_hub=false` | Defaults to `true` in 0.6.1, and validation then demands `--policy.repo_id` |
| `--policy.device=cuda` | |
| `--rename_map=<json>` | `smolvla_base` expects `observation.images.camera1..3`. Map your camera keys onto them. Cameras you leave out are skipped (`empty_cameras=0`). |
| `--wandb.enable=true` | Turns logging on. The launcher sends it to TensorBoard unless `SOARM_LOGGER=wandb`. |
| `--output_dir=/workspace/data/checkpoints/<job>` | Checkpoints and TensorBoard logs land in `data/checkpoints/<job>/` (gitignored) |

### Outputs

```
data/checkpoints/<job>/
  checkpoints/<step>/pretrained_model/   # config.json, model.safetensors, pre/post-processor (+ norm stats)
  checkpoints/<step>/training_state/     # optimizer, scheduler, RNG, step (for --resume)
  checkpoints/last -> <step>
  tensorboard/
```

Each SmolVLA checkpoint takes about 1.3 GB (865 MB model + 394 MB optimizer state). The
default `--save-freq 5000` over 20k steps writes 4 checkpoints, about 5 GB.

### TensorBoard

```bash
cd docker
docker compose --profile train-lerobot run --rm -p 127.0.0.1:6006:6006 -e NVIDIA_VISIBLE_DEVICES=void \
    lerobot tensorboard --logdir /workspace/data/checkpoints --bind_all
# open http://localhost:6006  (train/loss, train/lr, train/grad_norm, train/gpu_mem_gb, train/samples_per_s, ...)
```

### Datasets

LeRobot **v3.0** format is required. Hub datasets download once into the `lerobot-hf-cache`
volume. Set `HF_TOKEN` in `docker/.env` for private repos or higher rate limits. The
project's own recordings are not LeRobot-valid yet (#34), so the measurements below use the
public `lerobot/svla_so101_pickplace`: SO-101, 50 episodes, 11,939 frames at 30 fps, 2 cameras
(`up`, `side`) at 640x480 AV1, 6-D state and action, 1 task.

---

## Measurements (RTX A2000 12 GB, 2026-10-20)

Method: `train_lerobot.sh` with `--steps 150` (SmolVLA) or `--steps 200` (ACT), `--log-freq 25`,
6 workers, on `lerobot/svla_so101_pickplace`. Each run held the GPU alone. "Peak alloc" is
lerobot's `mem_gb` (`torch.cuda.max_memory_allocated`). "nvidia-smi peak" was sampled every 0.5 s
and includes the CUDA context plus about 0.15 GB of desktop/streaming processes. Step time is the
mean of lerobot's `updt_s + data_s` after the first log window.

### SmolVLA (`lerobot/smolvla_base`, defaults: vision encoder frozen, VLM frozen, expert + projections trained)

| Batch | Precision | Peak alloc | nvidia-smi peak | step/s | samples/s | 20k steps |
|---:|---|---:|---:|---:|---:|---:|
| 4 | bf16 weights (default) | 2.06 GB | 2.5 GB | 4.20 | 17 | 1.3 h |
| 8 | bf16 weights (default) | 2.88 GB | 3.3 GB | 2.44 | 20 | 2.3 h |
| **16** | bf16 weights (default) | **4.51 GB** | 5.0 GB | **1.28** | 20 | **4.3 h** |
| 32 | bf16 weights (default) | 7.78 GB | 8.4 GB | 0.64 | 21 | 8.7 h |
| 64 | bf16 weights (default) | OOM | 11.9 GB | - | - | - |
| 16 / 32 | `--amp` (fp16 autocast) | fails: GradScaler cannot unscale bf16 grads | | | | |

- 450 M parameters, 100 M trainable. `freeze_vision_encoder=True`, `train_expert_only=True` (the
  whole SmolVLM2 backbone is frozen and kept in `eval()`), `train_state_proj=True`. The base
  checkpoint stores all weights in **bf16**: 447 M bf16 parameters, of which 96.6 M are trainable
  expert layers, plus 3 M fp32 projection parameters. Training is therefore already bf16 with no
  autocast. In 0.6.1, `use_amp` means fp16 autocast plus a GradScaler, which crashes with
  `"_amp_foreach_non_finite_check_and_unscale_cuda" not implemented for 'BFloat16'`. The wrapper
  rejects `--amp` for SmolVLA.
- Throughput is **compute-bound** at about 20 samples/s. GPU utilization was 96-99 %, and
  `data_s` stayed at 3-7 ms per step even though both cameras are AV1, which torchcodec decodes on
  the CPU. The dataloader is not a bottleneck at this speed with 6 workers. Each step costs
  about `batch / 20` seconds, so wall time follows the total number of samples, not the step count.
- SmolVLA 0.6.1 has **no gradient-checkpointing option**. To go beyond batch 32, accumulate
  gradients by hand (not in `lerobot-train`) or shrink the model with `--policy.num_vlm_layers`
  or `num_expert_layers` (this changes the model, so it is not a plain fine-tune).
- Startup overhead is about 25 s per run: spawned dataloader workers plus model load. The first
  step takes 5-15 s.
- The scheduler auto-scales: with `steps < scheduler_decay_steps` (30k), lerobot shrinks warmup and
  decay in proportion. At 20k steps that gives about 667 warmup and 20k decay steps.

### ACT (from scratch, ResNet-18 backbone, full 640x480 images, 2 cameras)

| Batch | Precision | Peak alloc | nvidia-smi peak | step/s | samples/s | 20k steps |
|---:|---|---:|---:|---:|---:|---:|
| **8** | fp32 | **3.72 GB** | 4.3 GB | **2.98** | 24 | **1.9 h** |
| 16 | fp32 | 6.84 GB | 7.5 GB | 1.55 | 25 | 3.6 h |
| 16 | `--amp` (fp16) | 6.37 GB | 7.1 GB | 2.04 | 33 | 2.7 h |
| 24 | `--amp` (fp16) | 9.22 GB | 10.0 GB | 1.39 | 33 | 4.0 h |
| 32 | fp32 / `--amp` | OOM | 11.8 GB | - | - | - |

ACT runs at the dataset's full resolution, so it costs more memory per sample than SmolVLA,
which resizes to 512x512 and runs a frozen vision tower without gradients. AMP gives ACT about
1.3x throughput at a small memory saving.

### Inference for #36

`SmolVLAPolicy.from_pretrained(<checkpoint>/pretrained_model)` plus its saved pre/post-processors.
Timing covers one `predict_action_chunk` call (batch 1, 2 cameras, chunk 50 x 6), measured after warm-up:

| Setting | Latency (median) | Peak alloc | nvidia-smi (process) |
|---|---:|---:|---:|
| bf16 weights, 10 flow steps (default) | 271 ms | 0.89 GB | ~1.1 GB |
| + bf16 autocast | 287 ms | 0.89 GB | |
| + fp16 autocast | 315 ms | 0.90 GB | |
| 5 flow steps (`policy.config.num_steps=5`) | 159 ms | 0.89 GB | |

One 50-step chunk covers 1.67 s of 30 fps control, so 271 ms per chunk is fast enough to keep
ahead when chunks are executed open-loop or with async/RTC-style overlap. About 1.1 GB on the
GPU means SmolVLA inference **can co-run with Isaac Sim** (4-6 GB) on this card. Autocast does
not help because the weights are already bf16.

---

## Recommendations (tutorial defaults)

| | SmolVLA | ACT |
|---|---|---|
| Batch | **16** (32 fits if nothing else is on the GPU) | **8** (16 with `--amp`) |
| Steps | **20k** (quick check: 2k, about 26 min) | **20k** (quick check: 2k, about 11 min) |
| Expected time | **~4.5 h** at b16 (2.3 h at b8) | **~1.9 h** at b8 fp32 |
| Peak VRAM | 4.5 GB | 3.7 GB |
| Checkpoints | `--save-freq 5000` (1.3 GB each) | `--save-freq 5000` |

- Isaac Sim must be stopped while training. Inference (#36) can share the GPU with Isaac.
- For the same wall-clock, b16 x 20k and b32 x 10k see the same number of samples. The SmolVLA
  docs' `--batch_size=64 --steps=20000` recipe does not fit in 12 GB and would take about 18 h here.
- Always pass a `--rename-map` that matches your dataset's camera keys to `camera1..3`. Use the
  same camera order at inference time.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `'repo_id' argument missing. Please specify it to push the model to the hub.` | Pass `--policy.push_to_hub=false` (the wrapper does this) |
| `Feature mismatch` / visual feature validation error with SmolVLA | Add `--rename-map` for your camera keys |
| `not implemented for 'BFloat16'` in `grad_scaler` | Drop `--amp`/`--policy.use_amp` for SmolVLA |
| `torch.OutOfMemoryError` | Lower `--batch`. Check `nvidia-smi` for Isaac Sim or other GPU processes. |
| Output directory already exists | Use a new `--job-name` or `--resume` |
| Checkpoints owned by root | The service runs as `SOARM_UID:SOARM_GID` (the wrapper sets them to your user) |
