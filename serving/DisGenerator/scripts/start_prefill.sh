#!/bin/bash
# =============================================================================
# DisGenerator - Start Prefill Server(s)
# =============================================================================
# This script starts vLLM prefill server(s) configured as KV producers.
# Prefill servers process prompts and send KV cache to decode servers via NCCL.
#
# Usage:
#   ./start_prefill.sh [GPU_ID] [HTTP_PORT] [KV_PORT]
#
# Examples:
#   ./start_prefill.sh              # Use defaults: GPU 0, port 20001, kv 21001
#   ./start_prefill.sh 0 20001 21001
#   ./start_prefill.sh 1 20003 21002
#
# Environment Variables:
#   MODEL             - Model to serve (default: Qwen/Qwen3-4B-Thinking-2507)
#   PROXY_PORT        - Proxy ZMQ port (default: 30001)
#   MAX_MODEL_LEN     - Max sequence length (default: 8192)
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
KV_PORT="${3:-21001}"

# Configuration from environment with defaults (matches DisTrainer)
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
PROXY_PORT="${PROXY_PORT:-30001}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-65536}"
DTYPE="${DTYPE:-float16}"
GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.9}"
LORA_MODULE_NAME="${LORA_MODULE_NAME:-grpo-adapter}"
# Default LoRA path relative to serving directory (can be overridden via env var)
# Default LoRA path relative to serving directory (can be overridden via env var)
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
DISTRAINER_ADAPTER="$(cd "$SCRIPT_DIR/../../DisTrainer/models/latest_adapter" 2>/dev/null && pwd || echo '')"
LEGACY_ADAPTER="$(cd "$SCRIPT_DIR/../../ToolGRPOTrainer/grpo-streamed/checkpoint-10" 2>/dev/null && pwd || echo '')"

if [ -n "$DISTRAINER_ADAPTER" ]; then
    DEFAULT_LORA_PATH="$DISTRAINER_ADAPTER"
else
    DEFAULT_LORA_PATH="$LEGACY_ADAPTER"
fi
LORA_MODULE_PATH="${LORA_MODULE_PATH:-$DEFAULT_LORA_PATH}"

# Server identification
SERVER_ID="prefill_gpu${GPU_ID}_port${HTTP_PORT}"
LOG_FILE="logs/${SERVER_ID}.log"

echo "=============================================="
echo "Starting Prefill Server (KV Producer)"
echo "=============================================="
echo "  GPU:        $GPU_ID"
echo "  HTTP Port:  $HTTP_PORT"
echo "  KV Port:    $KV_PORT"
echo "  Model:      $MODEL"
echo "  Proxy:      0.0.0.0:$PROXY_PORT"
echo "  Log file:   $LOG_FILE"
echo "=============================================="

# Build KV transfer config JSON
KV_CONFIG=$(cat <<EOF
{
  "kv_connector": "P2pNcclConnector",
  "kv_role": "kv_producer",
  "kv_buffer_size": "1e9",
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
    2>&1 | tee "$LOG_FILE"
