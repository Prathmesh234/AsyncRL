#!/usr/bin/env bash
set -euo pipefail

# Start the FastAPI application
echo "[startup] Starting grader app..."
exec uv run python -m uvicorn src.main:app --host 0.0.0.0 --port 8001
