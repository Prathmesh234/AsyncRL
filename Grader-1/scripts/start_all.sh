#!/bin/bash

# Start all services for disaggregated inference
# This script starts prefill, decode, and proxy in the background

echo "Starting disaggregated inference setup..."
echo "This will start:"
echo "  - Prefill instance on GPU 0, port 8100"
echo "  - Decode instance on GPU 1, port 8200" 
echo "  - Proxy server on port 8000"
echo ""

# Create log directory
mkdir -p logs

# Start prefill instance in background
echo "Starting prefill instance..."
./start_prefill.sh 0 8100 > logs/prefill.log 2>&1 &
PREFILL_PID=$!
echo "Prefill instance started (PID: $PREFILL_PID)"

# Start decode instance in background
echo "Starting decode instance..."
./start_decode.sh 1 8200 > logs/decode.log 2>&1 &
DECODE_PID=$!
echo "Decode instance started (PID: $DECODE_PID)"

# Wait a bit for instances to start up
echo "Waiting 30 seconds for instances to initialize..."
sleep 30

# Start proxy server
echo "Starting proxy server..."
./start_proxy.sh 8100 8200 8000

# Clean up function
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $PREFILL_PID $DECODE_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Handle Ctrl+C
trap cleanup SIGINT SIGTERM

# Keep script running
wait
