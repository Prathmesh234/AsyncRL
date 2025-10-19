# Grader-2 environment Dockerfile with LMCache
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBIAN_FRONTEND=noninteractive

# Install system dependencies including UCX for NVLink/RDMA
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 python3-pip \
    python3-dev python3.10-dev \
    curl ca-certificates bash jq git \
    build-essential \
    libibverbs-dev librdmacm-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Python package manager
RUN pip install --no-cache-dir --upgrade pip uv

WORKDIR /app

# Copy dependency files
COPY pyproject.toml uv.lock ./

# Install Python dependencies (including vLLM and LMCache)
RUN uv lock
RUN uv sync --no-dev
RUN uv add "vllm>=0.11.0,<0.12"

# Add virtual environment to PATH
ENV PATH="/app/.venv/bin:$PATH"

# Copy application code and configuration
COPY src/ ./src/
COPY scripts/ ./scripts/
COPY lmcache-prefiller-config.yaml ./
COPY lmcache-decoder-config.yaml ./
COPY .env ./

# Make shell scripts executable
RUN chmod +x ./scripts/*.sh

# Create user
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8001 7100 7200 7300 7400 7500 9100

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["bash", "./scripts/start_all.sh"]
