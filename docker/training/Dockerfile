FROM nvidia/cuda:12.4.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates python3.11 python3.11-venv python3.11-dev \
    build-essential \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone OpenPi (same codebase as inference, used for training scripts)
WORKDIR /app
RUN git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git

WORKDIR /app/openpi
RUN uv sync --frozen 2>/dev/null || uv sync

# Additional training dependencies for memory-efficient fine-tuning
RUN uv pip install --no-cache-dir \
    bitsandbytes \
    peft \
    deepspeed \
    accelerate \
    wandb

COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

WORKDIR /app/openpi

CMD ["/entrypoint.sh"]
