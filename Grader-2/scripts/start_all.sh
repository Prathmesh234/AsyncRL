#!/bin/bash
set -e

# Load environment variables
set -a
source "$(dirname "$0")/../.env"
set +a

echo "========================================="
echo "Starting Grader-2 with LMCache"
echo "========================================="

# Function to check if a service is ready
wait_for_service() {
    local url=$1
    local name=$2
    local max_attempts=180  # Increased from 60 to 180 (6 minutes)
    local attempt=0

    echo "Waiting for $name to be ready at $url..."
    while [ $attempt -lt $max_attempts ]; do
        if curl -s "$url" > /dev/null 2>&1; then
            echo "$name is ready!"
            return 0
        fi
        attempt=$((attempt + 1))
        sleep 2
    done
    echo "ERROR: $name failed to start within timeout"
    return 1
}

# Start Decode server in background (needs to start first for handshake)
echo "Starting Decode Server on GPU ${DECODE_GPU}..."
bash "$(dirname "$0")/start_decode.sh" > /tmp/decode.log 2>&1 &
DECODE_PID=$!
echo "Decode server started with PID $DECODE_PID"

# Wait a bit for decode server to initialize
sleep 10

# Start Prefill server in background
echo "Starting Prefill Server on GPU ${PREFILL_GPU}..."
bash "$(dirname "$0")/start_prefill.sh" > /tmp/prefill.log 2>&1 &
PREFILL_PID=$!
echo "Prefill server started with PID $PREFILL_PID"

# Wait for both servers to be ready
wait_for_service "http://localhost:${PREFILL_PORT}/health" "Prefill Server"
wait_for_service "http://localhost:${DECODE_PORT}/health" "Decode Server"

# Start Proxy server in background
echo "Starting Proxy Server on port ${PROXY_PORT}..."
bash "$(dirname "$0")/start_proxy.sh" > /tmp/proxy.log 2>&1 &
PROXY_PID=$!
echo "Proxy server started with PID $PROXY_PID"

# Wait for proxy to be ready
wait_for_service "http://localhost:${PROXY_PORT}/health" "Proxy Server"

echo "========================================="
echo "All LMCache services are running!"
echo "  Prefill: http://localhost:${PREFILL_PORT}"
echo "  Decode:  http://localhost:${DECODE_PORT}"
echo "  Proxy:   http://localhost:${PROXY_PORT}"
echo "========================================="

# Start the FastAPI application (this runs in foreground)
echo "Starting FastAPI application..."
cd "$(dirname "$0")/.."
exec /app/.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8001

# Cleanup function (called on exit)
cleanup() {
    echo "Shutting down services..."
    kill $PROXY_PID $PREFILL_PID $DECODE_PID 2>/dev/null || true
    wait $PROXY_PID $PREFILL_PID $DECODE_PID 2>/dev/null || true
    echo "All services stopped."
}

trap cleanup EXIT INT TERM
