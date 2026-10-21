# LeRobot training/inference image: SmolVLA + ACT on a single 12 GB GPU.
# See docs/LEROBOT_TRAINING.md. Build with:
#   cd docker && docker compose --profile train-lerobot build lerobot
#
# Pins (bump together):
#   - lerobot 0.6.1 requires python>=3.12, torch>=2.7,<2.12, torchvision<0.27,
#     torchcodec<0.12, transformers>=5.4,<5.6 (smolvla extra).
#   - pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime = Ubuntu 24.04, Python 3.12,
#     torch 2.11.0+cu128, torchvision 0.26.0 (driver >= 570).
ARG BASE_IMAGE=pytorch/pytorch:2.11.0-cuda12.8-cudnn9-runtime@sha256:eee11b3b3872a8c838e35ef48f08b2d5def2080902c7f666831310ca1a0ef2be
FROM ${BASE_IMAGE}

ARG LEROBOT_VERSION=0.6.1
ARG TORCHCODEC_VERSION=0.11.1

# The base image installs torch into the system Python (PEP 668 marked); this
# container is single-purpose, so install alongside it rather than in a venv.
ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_BREAK_SYSTEM_PACKAGES=1

# ffmpeg: shared libs for torchcodec (AV1 via libdav1d, H.264/HEVC) and the CLI for
# dataset re-encoding. git: some HF repos / pip VCS deps.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Keep the base image's torch/torchvision (already satisfy lerobot's bounds) by
# pinning them explicitly; torchcodec must match the torch minor (0.11 <-> 2.11).
RUN TORCH_VER="$(python -c 'import torch; print(torch.__version__.split("+")[0])')" \
    && TV_VER="$(python -c 'import torchvision; print(torchvision.__version__.split("+")[0])')" \
    && pip install \
        "lerobot[smolvla,training]==${LEROBOT_VERSION}" \
        "torch==${TORCH_VER}" "torchvision==${TV_VER}" \
        "torchcodec==${TORCHCODEC_VERSION}" \
        "tensorboard>=2.18,<3" \
    && python -c "import lerobot, torch, torchcodec, transformers; print('lerobot', lerobot.__version__, 'torch', torch.__version__, 'torchcodec', torchcodec.__version__, 'transformers', transformers.__version__)"

# Writable homes for a non-root runtime UID. The HF cache lives in a named volume
# mounted at /hf-cache; creating it world-writable here makes a fresh volume inherit that.
ENV HF_HOME=/hf-cache \
    HF_LEROBOT_HOME=/hf-cache/lerobot \
    TORCH_HOME=/hf-cache/torch \
    HOME=/tmp/home \
    MPLCONFIGDIR=/tmp/home/.mpl
RUN mkdir -p /hf-cache/lerobot /hf-cache/torch /tmp/home /workspace/data \
    && chmod 1777 /hf-cache /hf-cache/lerobot /hf-cache/torch /tmp/home

COPY lerobot_train_tb.py /opt/soarm/lerobot_train_tb.py

WORKDIR /workspace
CMD ["lerobot-train", "--help"]
