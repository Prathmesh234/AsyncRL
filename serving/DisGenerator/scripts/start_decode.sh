#!/bin/bash
# DisGenerator Decode Server (with KV Transfer)
# Runs on GPUs 2, 3 with TP=2
# Receives KV cache from Prefill server via PyNccl connector

set -e

# Configuration
# Base model (from DisTrainer train_config.toml)
MODEL=${MODEL:-"Qwen/Qwen3-4B-Thinking-2507"}

# LoRA adapter from DisTrainer
LORA_ADAPTER_PATH=${LORA_ADAPTER_PATH:-"../../DisTrainer/checkpoints/Qwen3-4B-Thinking-Policy-v1/latest_adapter"}
LORA_ADAPTER_NAME=${LORA_ADAPTER_NAME:-"grpo-adapter"}

# Server settings
PORT=${DECODE_PORT:-8200}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEMORY_UTIL=${GPU_MEMORY_UTIL:-0.85}
MAX_NUM_SEQS=${MAX_NUM_SEQS:-128}
MAX_LORAS=${MAX_LORAS:-4}

# KV Transfer settings (for disaggregated decode)
# Using PyNccl for fast GPU-to-GPU transfer on same machine
KV_CONNECTOR=${KV_CONNECTOR:-"PyNcclConnector"}
KV_ROLE=${KV_ROLE:-"kv_consumer"}  # Decode = consumer
KV_RANK=${KV_RANK:-1}
KV_PARALLEL_SIZE=${KV_PARALLEL_SIZE:-2}  # 2 instances: prefill + decode

# Set GPU visibility
export CUDA_VISIBLE_DEVICES=2,3

echo "============================================"
echo "  DisGenerator Decode Server (KV Consumer)"
echo "============================================"
echo "Base Model: $MODEL"
echo "LoRA Adapter: $LORA_ADAPTER_NAME ($LORA_ADAPTER_PATH)"
echo "Port: $PORT"
echo "GPUs: $CUDA_VISIBLE_DEVICES (TP=2)"
echo "Max Model Length: $MAX_MODEL_LEN"
echo "KV Connector: $KV_CONNECTOR"
echo "KV Role: $KV_ROLE (rank $KV_RANK)"
echo "KV Handshake: $VLLM_KV_IP:$VLLM_KV_PORT"
echo "============================================"

# Build LoRA modules argument
LORA_MODULES="${LORA_ADAPTER_NAME}=${LORA_ADAPTER_PATH}"

uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --tensor-parallel-size 2 \
    --port "$PORT" \
    --max-model-len "$MAX_MODEL_LEN" \
    --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
    --enable-prefix-caching \
    --max-num-seqs "$MAX_NUM_SEQS" \
    --enable-lora \
    --lora-modules "$LORA_MODULES" \
    --max-loras "$MAX_LORAS" \
    --trust-remote-code \
    --kv-connector "$KV_CONNECTOR" \
    --kv-role "$KV_ROLE" \
    --kv-rank "$KV_RANK" \
    --kv-parallel-size "$KV_PARALLEL_SIZE"
