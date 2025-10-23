#!/bin/bash

# Start all services for disaggregated inference and grader API
# This script starts prefill, decode, proxy, and main FastAPI server in the background

echo "Starting disaggregated inference setup..."
echo "This will start:"
echo "  - Prefill instance on GPU 0, port 8100"
echo "  - Decode instance on GPU 1, port 8200" 
echo "  - Proxy server on port 8000"
echo "  - Main FastAPI grader server on port 8080"
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

# Start proxy server in background
echo "Starting proxy server..."
./start_proxy.sh 8100 8200 8000 > logs/proxy.log 2>&1 &
PROXY_PID=$!
echo "Proxy server started (PID: $PROXY_PID)"

# Wait a bit for proxy to start
echo "Waiting 10 seconds for proxy to initialize..."
sleep 10

# Start main FastAPI grader application
echo "Starting main FastAPI grader server..."
cd ..
.venv/bin/uvicorn src.main:app --host 0.0.0.0 --port 8080 > scripts/logs/grader.log 2>&1 &
GRADER_PID=$!
echo "FastAPI grader server started (PID: $GRADER_PID)"
cd scripts

# Clean up function
cleanup() {
    echo ""
    echo "Shutting down services..."
    kill $PREFILL_PID $DECODE_PID $PROXY_PID $GRADER_PID 2>/dev/null
    echo "Services stopped."
    exit 0
}

# Handle Ctrl+C
trap cleanup SIGINT SIGTERM

# Keep script running
wait
