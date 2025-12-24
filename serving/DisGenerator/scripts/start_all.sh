#!/bin/bash
# =============================================================================
# DisGenerator - Start All Components
# =============================================================================
# This script starts the complete disaggregated serving stack:
#   1. Proxy server (service discovery + API routing)
#   2. Prefill server(s) (KV producers)
#   3. Decode server(s) (KV consumers)
#
# Usage:
#   ./start_all.sh                    # Default: 1P1D configuration (2 GPUs)
#   ./start_all.sh 1p1d               # 1 Prefill + 1 Decode
#   ./start_all.sh 2p2d               # 2 Prefill + 2 Decode
#   ./start_all.sh 1p3d               # 1 Prefill + 3 Decode
#   ./start_all.sh 3p1d               # 3 Prefill + 1 Decode
#
# Environment Variables:
#   MODEL             - Model to serve (default: Qwen/Qwen3-4B-Thinking-2507)
#   HF_TOKEN          - HuggingFace token (required for gated models)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"

cd "$PROJECT_DIR"

# Configuration from arguments
CONFIG="${1:-1p1d}"

# Parse configuration
case "$CONFIG" in
    "1p1d")
        PREFILL_GPUS="0"
        DECODE_GPUS="1"
        PREFILL_HTTP_PORTS="20001"
        DECODE_HTTP_PORTS="20002"
        PREFILL_KV_PORTS="21001"
        DECODE_KV_PORTS="22001"
        echo "Configuration: 1P1D (1 Prefill + 1 Decode)"
        ;;
    "2p2d")
        PREFILL_GPUS="0,1"
        DECODE_GPUS="2,3"
        PREFILL_HTTP_PORTS="20001,20003"
        DECODE_HTTP_PORTS="20002,20004"
        PREFILL_KV_PORTS="21001,21002"
        DECODE_KV_PORTS="22001,22002"
        echo "Configuration: 2P2D (2 Prefill + 2 Decode)"
        ;;
    "1p3d")
        PREFILL_GPUS="0"
        DECODE_GPUS="1,2,3"
        PREFILL_HTTP_PORTS="20001"
        DECODE_HTTP_PORTS="20002,20004,20006"
        PREFILL_KV_PORTS="21001"
        DECODE_KV_PORTS="22001,22002,22003"
        echo "Configuration: 1P3D (1 Prefill + 3 Decode)"
        ;;
    "3p1d")
        PREFILL_GPUS="0,1,2"
        DECODE_GPUS="3"
        PREFILL_HTTP_PORTS="20001,20003,20005"
        DECODE_HTTP_PORTS="20002"
        PREFILL_KV_PORTS="21001,21002,21003"
        DECODE_KV_PORTS="22001"
        echo "Configuration: 3P1D (3 Prefill + 1 Decode)"
        ;;
    *)
        echo "Unknown configuration: $CONFIG"
        echo "Valid options: 1p1d, 2p2d, 1p3d, 3p1d"
        exit 1
        ;;
esac

# Model configuration (matches DisTrainer)
MODEL="${MODEL:-Qwen/Qwen3-4B-Thinking-2507}"
PROXY_PORT="${PROXY_PORT:-30001}"
TIMEOUT_SECONDS="${TIMEOUT_SECONDS:-600}"

echo ""
echo "=============================================="
echo "DisGenerator - Disaggregated Serving Stack"
echo "=============================================="
echo "  Model:          $MODEL"
echo "  Prefill GPUs:   $PREFILL_GPUS"
echo "  Decode GPUs:    $DECODE_GPUS"
echo "  Proxy Port:     $PROXY_PORT"
echo "  API Port:       10001"
echo "=============================================="
echo ""

# Check HF_TOKEN for gated models
if [[ "$MODEL" == *"meta-llama"* ]] || [[ "$MODEL" == *"Llama"* ]]; then
    if [ -z "$HF_TOKEN" ]; then
        echo "Warning: HF_TOKEN is not set. This may be required for gated models."
    fi
fi

# Check number of GPUs
check_gpus() {
    if command -v nvidia-smi &> /dev/null; then
        NUM_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
        echo "Found $NUM_GPUS GPU(s)"
        
        IFS=',' read -ra P_GPUS <<< "$PREFILL_GPUS"
        IFS=',' read -ra D_GPUS <<< "$DECODE_GPUS"
        REQUIRED_GPUS=$((${#P_GPUS[@]} + ${#D_GPUS[@]}))
        
        if [ "$NUM_GPUS" -lt "$REQUIRED_GPUS" ]; then
            echo "Error: Configuration requires $REQUIRED_GPUS GPUs, but only $NUM_GPUS available."
            exit 1
        fi
    else
        echo "Warning: nvidia-smi not found, skipping GPU check"
    fi
}

check_gpus

# Check if uv is available
if ! command -v uv &> /dev/null; then
    echo "Error: uv is not installed. Install from: https://docs.astral.sh/uv/"
    exit 1
fi

# Sync dependencies
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment and installing dependencies..."
    uv sync
fi

# Create logs directory
mkdir -p logs

# Array to store PIDs
PIDS=()

# Cleanup function
cleanup() {
    echo ""
    echo "Shutting down all servers..."
    for pid in "${PIDS[@]}"; do
        if kill -0 "$pid" 2>/dev/null; then
            kill "$pid" 2>/dev/null || true
        fi
    done
    
    # Also kill any vllm processes
    pkill -f "vllm serve" 2>/dev/null || true
    pkill -f "disagg_proxy.py" 2>/dev/null || true
    
    echo "All servers stopped."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Wait for server to be ready
wait_for_server() {
    local port=$1
    local timeout=$TIMEOUT_SECONDS
    local start_time=$(date +%s)
    
    echo "Waiting for server on port $port..."
    
    while true; do
        if curl -s "http://localhost:$port/v1/models" > /dev/null 2>&1; then
            echo "✅ Server on port $port is ready."
            return 0
        fi
        
        local now=$(date +%s)
        if (( now - start_time >= timeout )); then
            echo "❌ Timeout waiting for server on port $port"
            return 1
        fi
        
        sleep 2
    done
}

# =============================================================================
# Start Proxy Server
# =============================================================================
echo ""
echo "🔄 Starting proxy server..."
uv run python disagg_proxy.py > logs/proxy.log 2>&1 &
PIDS+=($!)
echo "   Proxy PID: ${PIDS[-1]}"
sleep 2

# =============================================================================
# Start Prefill Servers
# =============================================================================
IFS=',' read -ra P_GPU_ARR <<< "$PREFILL_GPUS"
IFS=',' read -ra P_HTTP_ARR <<< "$PREFILL_HTTP_PORTS"
IFS=',' read -ra P_KV_ARR <<< "$PREFILL_KV_PORTS"

echo ""
echo "🔄 Starting ${#P_GPU_ARR[@]} prefill server(s)..."

for i in "${!P_GPU_ARR[@]}"; do
    gpu_id="${P_GPU_ARR[$i]}"
    http_port="${P_HTTP_ARR[$i]}"
    kv_port="${P_KV_ARR[$i]}"
    
    echo "   Prefill $((i+1)): GPU $gpu_id, HTTP $http_port, KV $kv_port"
    bash "$SCRIPT_DIR/start_prefill.sh" "$gpu_id" "$http_port" "$kv_port" &
    PIDS+=($!)
done

# =============================================================================
# Start Decode Servers
# =============================================================================
IFS=',' read -ra D_GPU_ARR <<< "$DECODE_GPUS"
IFS=',' read -ra D_HTTP_ARR <<< "$DECODE_HTTP_PORTS"
IFS=',' read -ra D_KV_ARR <<< "$DECODE_KV_PORTS"

echo ""
echo "🔄 Starting ${#D_GPU_ARR[@]} decode server(s)..."

for i in "${!D_GPU_ARR[@]}"; do
    gpu_id="${D_GPU_ARR[$i]}"
    http_port="${D_HTTP_ARR[$i]}"
    kv_port="${D_KV_ARR[$i]}"
    
    echo "   Decode $((i+1)): GPU $gpu_id, HTTP $http_port, KV $kv_port"
    bash "$SCRIPT_DIR/start_decode.sh" "$gpu_id" "$http_port" "$kv_port" &
    PIDS+=($!)
done

# =============================================================================
# Wait for All Servers
# =============================================================================
echo ""
echo "⏳ Waiting for all servers to be ready..."

# Wait for prefill servers
for port in "${P_HTTP_ARR[@]}"; do
    if ! wait_for_server "$port"; then
        echo "Failed to start prefill server on port $port"
        cleanup
    fi
done

# Wait for decode servers
for port in "${D_HTTP_ARR[@]}"; do
    if ! wait_for_server "$port"; then
        echo "Failed to start decode server on port $port"
        cleanup
    fi
done

echo ""
echo "=============================================="
echo "🚀 All servers are up and running!"
echo "=============================================="
echo ""
echo "API Endpoint: http://localhost:10001/v1/chat/completions"
echo ""
echo "Test with:"
echo "  curl -X POST http://localhost:10001/v1/chat/completions \\"
echo "    -H 'Content-Type: application/json' \\"
echo "    -d '{\"model\": \"$MODEL\", \"messages\": [{\"role\": \"user\", \"content\": \"Hello!\"}]}'"
echo ""
echo "Press Ctrl+C to stop all servers."
echo ""

# Wait for all processes
wait
