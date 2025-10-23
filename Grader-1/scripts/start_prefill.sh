#!/bin/bash

# Start vLLM prefill instance
# Usage: ./start_prefill.sh [GPU_ID] [PORT]

GPU_ID=${1:-0}
PORT=${2:-8100}

echo "Starting vLLM prefill instance on GPU $GPU_ID, port $PORT"

cd ..
CUDA_VISIBLE_DEVICES=$GPU_ID uv run vllm serve ./openai/gpt-oss-20b \
    --port $PORT \
    --disable-log-requests \
    --gpu-memory-utilization 0.75 \
    --max-model-len 65536 \
    --max-num-seqs 4
