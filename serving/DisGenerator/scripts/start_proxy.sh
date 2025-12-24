#!/bin/bash
# =============================================================================
# DisGenerator - Start Proxy Server
# =============================================================================
# This script starts the disaggregated proxy server that coordinates
# KV cache transfer routing between prefill and decode instances.
#
# Usage:
#   ./start_proxy.sh
#
# Environment Variables:
#   PROXY_IP          - IP to bind (default: 0.0.0.0)
#   PROXY_ZMQ_PORT    - ZMQ port for service discovery (default: 30001)
#   PROXY_HTTP_PORT   - HTTP port for API (default: 10001)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Ensure logs directory exists
mkdir -p logs

echo "=============================================="
echo "Starting DisGenerator Proxy Server"
echo "=============================================="

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install from: https://docs.astral.sh/uv/"
    exit 1
fi

# Sync dependencies if needed
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment and installing dependencies..."
    uv sync
fi

# Export environment variables (can be overridden)
export PROXY_IP="${PROXY_IP:-0.0.0.0}"
export PROXY_ZMQ_PORT="${PROXY_ZMQ_PORT:-30001}"
export PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-10001}"

echo "  ZMQ Port: $PROXY_ZMQ_PORT"
echo "  HTTP Port: $PROXY_HTTP_PORT"
echo "  Log file: logs/proxy.log"
echo "=============================================="

# Run the proxy server
uv run python disagg_proxy.py 2>&1 | tee logs/proxy.log
