#!/usr/bin/env bash
set -euo pipefail

# Ensure required environment variables are set
if [[ -z "${AZURE_CLIENT_ID:-}" || -z "${AZURE_CLIENT_SECRET:-}" || -z "${AZURE_TENANT_ID:-}" ]]; then
  echo "[startup] Missing one or more required environment variables: AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, AZURE_TENANT_ID" >&2
  exit 1
fi

# Login using service principal and client secret
echo "[startup] Logging into Azure with service principal..."
if ! az login --service-principal \
  --username "$AZURE_CLIENT_ID" \
  --password "$AZURE_CLIENT_SECRET" \
  --tenant "$AZURE_TENANT_ID" >/dev/null; then
  echo "[startup] Azure login failed. Check your client secret or tenant ID." >&2
  exit 1
fi

echo "[startup] Azure login successful."

# Optionally set default subscription if provided
if [[ -n "${AZURE_SUBSCRIPTION_ID:-}" ]]; then
  echo "[startup] Setting default subscription to '$AZURE_SUBSCRIPTION_ID'..."
  az account set --subscription "$AZURE_SUBSCRIPTION_ID" || echo "[startup] Warning: failed to set subscription" >&2
fi

# Start your app
echo "[startup] Starting app..."
exec uv run python -m uvicorn src.main:app --host 0.0.0.0 --port 8001
