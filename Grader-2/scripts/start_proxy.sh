#!/bin/bash

# Load environment variables
set -a
source "$(dirname "$0")/../.env"
set +a

echo "Starting Proxy Server on port ${PROXY_PORT}..."

# Download the disagg_proxy_server.py if not present
PROXY_SCRIPT="$(dirname "$0")/disagg_proxy_server.py"
if [ ! -f "$PROXY_SCRIPT" ]; then
    echo "Downloading disagg_proxy_server.py from vLLM repository..."
    curl -o "$PROXY_SCRIPT" https://raw.githubusercontent.com/vllm-project/vllm/main/examples/others/lmcache/disagg_prefill_lmcache_v1/disagg_proxy_server.py
    chmod +x "$PROXY_SCRIPT"
fi

# Launch Proxy Server
/app/.venv/bin/python3 "$PROXY_SCRIPT" \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  --prefiller-host "${PROXY_HOST}" \
  --prefiller-port "${PREFILL_PORT}" \
  --decoder-host "${PROXY_HOST}" \
  --decoder-port "${DECODE_PORT}"
