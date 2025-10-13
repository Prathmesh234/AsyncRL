# Grader environment Dockerfile
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates bash jq git \
    && rm -rf /var/lib/apt/lists/*

# Install uv for dependency management
RUN pip install --no-cache-dir uv

WORKDIR /app

COPY pyproject.toml uv.lock ./
# add build tools (safe for vLLM wheels / fallbacks)
RUN apt-get update && apt-get install -y --no-install-recommends build-essential git && rm -rf /var/lib/apt/lists/*

# Regenerate lock file to ensure all current dependencies are included, then install
RUN uv lock
RUN uv sync --no-dev

COPY src/ ./src/
COPY prefill_disaggregated.py ./
COPY decode_disaggregated.py ./
COPY scripts/ ./scripts/

RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app
USER app

EXPOSE 8001

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["/app/scripts/start.sh"]
