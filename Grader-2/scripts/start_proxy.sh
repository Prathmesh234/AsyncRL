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
    curl -o "$PROXY_SCRIPT" https://raw.githubusercontent.com/vllm-project/vllm/main/examples/disagg_proxy_server.py
    chmod +x "$PROXY_SCRIPT"
fi

# Launch Proxy Server
python3 "$PROXY_SCRIPT" \
  --host "${PROXY_HOST}" \
  --port "${PROXY_PORT}" \
  --prefiller-host "${PROXY_HOST}" \
  --prefiller-port "${PREFILL_PORT}" \
  --num-prefillers 1 \
  --decoder-host "${PROXY_HOST}" \
  --decoder-port "${DECODE_PORT}" \
  --decoder-init-port "${DECODER_INIT_PORT}" \
  --decoder-alloc-port "${DECODER_ALLOC_PORT}" \
  --proxy-host "${PROXY_HOST}" \
  --proxy-port "${PROXY_PORT_INTERNAL}" \
  --num-decoders 1
