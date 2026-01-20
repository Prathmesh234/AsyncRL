#!/bin/bash
# =============================================================================
# DisGenerator - Start Decode Server(s)
# =============================================================================
# This script starts vLLM decode server(s) configured as KV consumers.
# Decode servers receive KV cache from prefill servers via NCCL and generate tokens.
#
# Usage:
#   ./start_decode.sh [GPU_ID] [HTTP_PORT] [KV_PORT] [--use-base-model]
#
# Examples:
#   ./start_decode.sh              # Use defaults: GPU 1, port 20002, kv 22001
#   ./start_decode.sh 1 20002 22001
#   ./start_decode.sh 2 20004 22002
#   ./start_decode.sh 1 20002 22001 --use-base-model  # Use base SFT adapter (checkpoint-2280)
#
# Adapter Modes:
#   Default (no flag):     Uses policy-0-initial or latest DisTrainer policy
#   --use-base-model:      Uses checkpoint-2280-openthoughts (SFT from Finetuning.ipynb)
#
# Environment Variables:
#   MODEL             - Model to serve (default: Qwen/Qwen3-4B-Thinking-2507)
#   PROXY_PORT        - Proxy ZMQ port (default: 30001)
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
GPU_ID="${1:-1}"
HTTP_PORT="${2:-20002}"
KV_PORT="${3:-22001}"
USE_BASE_MODEL="${4:-}"

# Configuration from environment with defaults
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
PROXY_PORT="${PROXY_PORT:-30001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
DTYPE="${DTYPE:-float16}"
# Lower GPU memory utilization to leave room for incoming KV cache
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.7}"

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
SERVER_ID="decode_gpu${GPU_ID}_port${HTTP_PORT}"
LOG_FILE="logs/${SERVER_ID}.log"

echo "=============================================="
echo "Starting Decode Server (KV Consumer)"
echo "=============================================="
echo "  GPU:        $GPU_ID"
echo "  HTTP Port:  $HTTP_PORT"
echo "  KV Port:    $KV_PORT"
echo "  Model:      $MODEL"
if [ "$USE_BASE_MODEL" == "--use-base-model" ]; then
    echo "  Mode:       BASE MODEL (SFT checkpoint-2280)"
else
    echo "  Mode:       GRPO FINETUNED"
fi
echo "  LoRA:       $LORA_MODULE_PATH"
echo "  Proxy:      0.0.0.0:$PROXY_PORT"
echo "  Log file:   $LOG_FILE"
echo "=============================================="

# Build KV transfer config JSON
KV_CONFIG=$(cat <<EOF
{
  "kv_connector": "P2pNcclConnector",
  "kv_role": "kv_consumer",
  "kv_buffer_size": "8e9",
  "kv_port": "$KV_PORT",
  "kv_connector_extra_config": {
    "proxy_ip": "0.0.0.0",
    "proxy_port": "$PROXY_PORT",
    "http_port": "$HTTP_PORT",
    "send_type": "PUT_ASYNC",
    "nccl_num_channels": "16"
  }
}
EOF
)

# Convert to single line for command
KV_CONFIG_INLINE=$(echo "$KV_CONFIG" | tr -d '\n' | tr -s ' ')

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
