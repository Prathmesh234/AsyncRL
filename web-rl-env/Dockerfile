# Use Python 3.11 slim image as base
FROM python:3.11-slim

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
RUN pip install --no-cache-dir uv

# Set working directory
WORKDIR /app

# Install Python dependencies using lockfile for reproducibility
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev

# Install only the Chromium browser to reduce image size
RUN mkdir -p /ms-playwright \
 && chown -R root:root /ms-playwright \
 && uv run python -m playwright install --with-deps chromium \
 && rm -rf /root/.cache

# Copy application source
COPY src/ ./src/
COPY web_env/ ./web_env/
COPY scripts/ ./scripts/

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash app \
    && chown -R app:app /app \
    && chown -R app:app /ms-playwright
USER app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Run the application
CMD ["uv", "run", "python", "-m", "uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]
