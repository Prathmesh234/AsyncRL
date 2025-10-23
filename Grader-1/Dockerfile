FROM nvidia/cuda:12.4.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    UV_PROJECT_ENVIRONMENT=.venv \
    CUDA_DEVICE_ORDER=PCI_BUS_ID \
    CUDA_VISIBLE_DEVICES=0,1 \
    PREFILL_GPU=0 \
    DECODE_GPU=1 \
    NVIDIA_VISIBLE_DEVICES=0,1 \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    LANG=C.UTF-8 \
    LC_ALL=C.UTF-8

RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ca-certificates \
    curl \
    git \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev && \
    rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    ln -s /root/.local/bin/uv /usr/local/bin/uv

WORKDIR /app

COPY pyproject.toml uv.lock ./

RUN uv sync --frozen --python python3

COPY . .

RUN chmod +x docker-entrypoint.sh && chmod +x scripts/*.sh

ENV PATH="/app/.venv/bin:${PATH}"

ENTRYPOINT ["/app/docker-entrypoint.sh"]
