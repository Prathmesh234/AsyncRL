#!/bin/bash
# =============================================================================
# DisGenerator - Start Prefill Server(s)
# =============================================================================
# This script starts vLLM prefill server(s) using the NixlConnector.
# Prefill servers compute the KV cache; decode servers pull it directly via
# NIXL (RDMA/UCX). Routing is handled by the proxy's kv_transfer_params
# handshake — no ZMQ service discovery needed.
#
# Usage:
#   ./start_prefill.sh [GPU_ID] [HTTP_PORT] [SIDE_CHANNEL_PORT] [--use-base-model]
#
# Examples:
#   ./start_prefill.sh              # Use defaults: GPU 0, port 20001, side channel 21001
#   ./start_prefill.sh 0 20001 21001
#   ./start_prefill.sh 1 20003 21002
#   ./start_prefill.sh 0 20001 21001 --use-base-model  # Use base SFT adapter (checkpoint-2280)
#
# Adapter Modes:
#   Default (no flag):     Uses policy-0-initial or latest DisTrainer policy
#   --use-base-model:      Uses checkpoint-2280-openthoughts (SFT from Finetuning.ipynb)
#
# Environment Variables:
#   MODEL             - Model to serve (default: Qwen/Qwen3-4B-Thinking-2507)
#   MAX_MODEL_LEN     - Max sequence length (default: 65536)
#   DTYPE             - Data type (default: float16)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

# Change to project directory
cd "$PROJECT_DIR"

# Ensure logs directory exists
mkdir -p logs

# Parse arguments with defaults
GPU_ID="${1:-0}"
HTTP_PORT="${2:-20001}"
SIDE_CHANNEL_PORT="${3:-21001}"
USE_BASE_MODEL="${4:-}"

# Configuration from environment with defaults
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"

# Models directory (relative to DisGenerator)
MODELS_DIR="$SCRIPT_DIR/../../DisTrainer/models"

# Select adapter based on flag
if [ "$USE_BASE_MODEL" == "--use-base-model" ]; then
    # Use the SFT adapter from Finetuning.ipynb
    LORA_MODULE_NAME="openthoughts-adapter"
    LORA_MODULE_PATH="$(cd "$MODELS_DIR/checkpoint-2280-openthoughts" 2>/dev/null && pwd)"
else
    # Use ToolGRPO adapter (policy-0-initial) or latest trainer policy
    LORA_MODULE_NAME="grpo-adapter"
    # Check for latest_adapter symlink first, fallback to policy-0-initial
    if [ -d "$MODELS_DIR/latest_adapter" ]; then
        LORA_MODULE_PATH="$(cd "$MODELS_DIR/latest_adapter" && pwd)"
    else
        LORA_MODULE_PATH="$(cd "$MODELS_DIR/policy-0-initial" 2>/dev/null && pwd)"
    fi
fi

# Server identification
SERVER_ID="prefill_gpu${GPU_ID}_port${HTTP_PORT}"
LOG_FILE="logs/${SERVER_ID}.log"

echo "=============================================="
echo "Starting Prefill Server (NIXL)"
echo "=============================================="
echo "  GPU:          $GPU_ID"
echo "  HTTP Port:    $HTTP_PORT"
echo "  Side Channel: $SIDE_CHANNEL_PORT"
echo "  Model:        $MODEL"
if [ "$USE_BASE_MODEL" == "--use-base-model" ]; then
    echo "  Mode:         BASE MODEL (SFT checkpoint-2280)"
else
    echo "  Mode:         GRPO FINETUNED"
fi
echo "  LoRA:         $LORA_MODULE_PATH"
echo "  Log file:     $LOG_FILE"
echo "=============================================="

# NixlConnector config: both roles are "kv_both"; the proxy's
# kv_transfer_params handshake decides who produces and who consumes.
KV_CONFIG_INLINE='{"kv_connector":"NixlConnector","kv_role":"kv_both"}'

# Each vLLM instance on the same host needs a unique NIXL side channel port.
export VLLM_NIXL_SIDE_CHANNEL_PORT=$SIDE_CHANNEL_PORT

# Enable /v1/load_lora_adapter + /v1/unload_lora_adapter so PolicyManager can
# hot-swap new policies without restarting the server.
export VLLM_ALLOW_RUNTIME_LORA_UPDATING=True

# Start the server
echo "Launching vLLM server..."

# Build and execute the vLLM command
CUDA_VISIBLE_DEVICES=$GPU_ID uv run vllm serve $MODEL \
    --enforce-eager \
    --host 0.0.0.0 \
    --port $HTTP_PORT \
    --tensor-parallel-size 1 \
    --seed 1024 \
    --dtype $DTYPE \
    --max-model-len $MAX_MODEL_LEN \
    --max-num-batched-tokens $MAX_MODEL_LEN \
    --max-num-seqs 128 \
    --trust-remote-code \
    --gpu-memory-utilization $GPU_MEMORY_UTILIZATION \
    --kv-transfer-config "$KV_CONFIG_INLINE" \
    --enable-lora \
    --lora-modules "$LORA_MODULE_NAME=$LORA_MODULE_PATH" \
    --max-loras 3 \
    --max-cpu-loras 5 \
    2>&1 | tee "$LOG_FILE"
