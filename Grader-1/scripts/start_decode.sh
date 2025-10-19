#!/bin/bash

# Start vLLM decode instance
# Usage: ./start_decode.sh [GPU_ID] [PORT]

GPU_ID=${1:-1}
PORT=${2:-8200}

echo "Starting vLLM decode instance on GPU $GPU_ID, port $PORT"

cd ..
CUDA_VISIBLE_DEVICES=$GPU_ID uv run vllm serve ./gpt-oss-20b \
    --port $PORT \
    --disable-log-requests \
    --gpu-memory-utilization 0.8
