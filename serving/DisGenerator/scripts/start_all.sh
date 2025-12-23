#!/bin/bash
# DisGenerator - Start All Servers
# Launches both Prefill and Decode servers in background

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LOG_DIR="${SCRIPT_DIR}/../logs"

# Create log directory
mkdir -p "$LOG_DIR"

# Set KV Transfer Handshake Address
# This is required for Prefill and Decode servers to find each other
export VLLM_KV_IP=${VLLM_KV_IP:-"127.0.0.1"}
export VLLM_KV_PORT=${VLLM_KV_PORT:-"12345"}

echo "============================================"
echo "  Starting DisGenerator Servers"
echo "  KV Transfer: $VLLM_KV_IP:$VLLM_KV_PORT"
echo "============================================"

# Start Prefill Server
echo "Starting Prefill Server (GPUs 0,1 on port 8100)..."
setsid bash "$SCRIPT_DIR/start_prefill.sh" > "$LOG_DIR/prefill.log" 2>&1 &
PREFILL_PID=$!
echo "Prefill Server PID: $PREFILL_PID"

# Wait a bit before starting decode to avoid GPU contention
sleep 5

# Start Decode Server
echo "Starting Decode Server (GPUs 2,3 on port 8200)..."
setsid bash "$SCRIPT_DIR/start_decode.sh" > "$LOG_DIR/decode.log" 2>&1 &
DECODE_PID=$!
echo "Decode Server PID: $DECODE_PID"

echo "============================================"
echo "  Servers Starting..."
echo "============================================"
echo "Prefill: http://localhost:8100 (PID: $PREFILL_PID)"
echo "Decode:  http://localhost:8200 (PID: $DECODE_PID)"
echo ""
echo "Logs:"
echo "  - Prefill: $LOG_DIR/prefill.log"
echo "  - Decode:  $LOG_DIR/decode.log"
echo ""
echo "To monitor: tail -f $LOG_DIR/*.log"
echo "To stop: kill $PREFILL_PID $DECODE_PID"
echo "============================================"

# Save PIDs for later
echo "$PREFILL_PID" > "$LOG_DIR/prefill.pid"
echo "$DECODE_PID" > "$LOG_DIR/decode.pid"

# Wait for servers to be ready
echo "Waiting for servers to initialize..."
sleep 30

# Health check
echo "Checking server health..."
for PORT in 8100 8200; do
    for i in {1..10}; do
        if curl -s "http://localhost:$PORT/health" > /dev/null 2>&1; then
            echo "  Port $PORT: READY"
            break
        fi
        if [ $i -eq 10 ]; then
            echo "  Port $PORT: NOT READY (check logs)"
        fi
        sleep 3
    done
done

echo "============================================"
echo "  Setup Complete!"
echo "============================================"
