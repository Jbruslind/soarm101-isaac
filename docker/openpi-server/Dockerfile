FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl ca-certificates python3.11 python3.11-venv python3.11-dev \
    build-essential \
    && update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && rm -rf /var/lib/apt/lists/*

# Install uv (the package manager used by OpenPi)
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
ENV PATH="/root/.local/bin:$PATH"

# Clone OpenPi
WORKDIR /app
RUN git clone --depth 1 https://github.com/Physical-Intelligence/openpi.git

WORKDIR /app/openpi

# Install OpenPi dependencies with uv
RUN uv sync --frozen 2>/dev/null || uv sync

# Pre-download base model weights on build (optional, can be skipped)
# RUN uv run python -c "import openpi.shared.download as dl; dl.maybe_download('gs://openpi-assets/checkpoints/pi0_fast_base')"

EXPOSE 8000

# Entrypoint: serve the policy model
# Override OPENPI_POLICY_CONFIG and OPENPI_CHECKPOINT_DIR via env vars
COPY scripts/register_soarm_configs.py /app/openpi/scripts/register_soarm_configs.py
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

CMD ["/entrypoint.sh"]
