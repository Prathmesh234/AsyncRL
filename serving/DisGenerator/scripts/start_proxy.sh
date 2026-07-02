#!/bin/bash
# =============================================================================
# DisGenerator - Start Proxy Server
# =============================================================================
# This script starts the disaggregated proxy server that routes requests
# through the two-phase NIXL prefill/decode flow.
#
# Usage:
#   ./start_proxy.sh
#
# Environment Variables:
#   PROXY_IP           - IP to bind (default: 0.0.0.0)
#   PROXY_HTTP_PORT    - HTTP port for API (default: 10001)
#   PREFILL_INSTANCES  - comma-separated host:port list (default: localhost:20001)
#   DECODE_INSTANCES   - comma-separated host:port list (default: localhost:20002)
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
export PROXY_HTTP_PORT="${PROXY_HTTP_PORT:-10001}"
export PREFILL_INSTANCES="${PREFILL_INSTANCES:-localhost:20001}"
export DECODE_INSTANCES="${DECODE_INSTANCES:-localhost:20002}"

echo "  HTTP Port: $PROXY_HTTP_PORT"
echo "  Prefill:   $PREFILL_INSTANCES"
echo "  Decode:    $DECODE_INSTANCES"
echo "  Log file: logs/proxy.log"
echo "=============================================="

# Run the proxy server
uv run python disagg_proxy.py 2>&1 | tee logs/proxy.log
