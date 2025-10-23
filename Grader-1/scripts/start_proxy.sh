#!/bin/bash

# Start disaggregated proxy server
# Usage: ./start_proxy.sh [PREFILL_PORT] [DECODE_PORT] [PROXY_PORT]

PREFILL_PORT=${1:-8100}
DECODE_PORT=${2:-8200}
PROXY_PORT=${3:-8000}

echo "Starting disaggregated proxy server"
echo "Prefill instance: localhost:$PREFILL_PORT"
echo "Decode instance: localhost:$DECODE_PORT"
echo "Proxy listening on port: $PROXY_PORT"

cd ..
uv run disagg_proxy_demo.py \
    --model ./openai/gpt-oss-20b \
    --prefill localhost:$PREFILL_PORT \
    --decode localhost:$DECODE_PORT \
    --port $PROXY_PORT
