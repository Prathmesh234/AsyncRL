#!/usr/bin/env bash
set -euo pipefail

# Ensure Service Bus connection string exists
if [[ -z "${SERVICE_BUS_CONNECTION_STRING:-}" ]]; then
  echo "[startup] Missing SERVICE_BUS_CONNECTION_STRING env var" >&2
  exit 1
fi

# Ensure workspace directory exists for persistent projects
CODE_WORKSPACE_DIR="${CODE_WORKSPACE_DIR:-/app/projects}"
echo "[startup] Using workspace directory: '${CODE_WORKSPACE_DIR}'"
mkdir -p "${CODE_WORKSPACE_DIR}"

# Start your app
echo "[startup] Starting app..."
exec uv run python -m uvicorn src.main:app --host 0.0.0.0 --port 8002
