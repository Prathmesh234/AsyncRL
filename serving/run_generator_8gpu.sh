#!/bin/bash
# =============================================================================
# Run DisGenerator on 8 GPU setup (GPUs 0-3)
# =============================================================================
# This script runs the DisGenerator using a 2P2D (2 Prefill + 2 Decode)
# configuration on GPUs 0-3.
#
# Requirements:
#   - At least 8 GPUs (GPUs 0-3 for generation, GPUs 4-7 for training)
#   - uv installed (https://docs.astral.sh/uv/)
#   - Model weights downloaded or accessible via HuggingFace
#
# Usage:
#   ./run_generator_8gpu.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DisGenerator"

# Use GPUs 0-3 for generation (2P2D)
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=============================================="
echo "Starting DisGenerator (2P2D on GPUs 0,1,2,3)"
echo "=============================================="
echo ""
echo "GPU Allocation:"
echo "  GPU 0: Prefill Server 1"
echo "  GPU 1: Prefill Server 2"
echo "  GPU 2: Decode Server 1"
echo "  GPU 3: Decode Server 2"
echo ""

# Create logs directory
mkdir -p logs

# Start vLLM servers + proxy
echo "Step 1: Starting disaggregated vLLM servers..."
./scripts/start_all.sh 2p2d &
SERVERS_PID=$!

# Give servers time to initialize (more time for 4 servers)
echo "Step 2: Waiting for servers to initialize (90 seconds)..."
sleep 90

# Check if servers are running
if ! kill -0 $SERVERS_PID 2>/dev/null; then
    echo "ERROR: Servers failed to start. Check logs/prefill_*.log and logs/decode_*.log"
    exit 1
fi

echo ""
echo "Step 3: Starting Batch Orchestrator..."
echo "=============================================="
echo ""

# Run the orchestrator (this blocks until interrupted)

uv run python simple_client.py --tool azure --tool code --tool web

# Cleanup on exit
echo "Shutting down servers..."
kill $SERVERS_PID 2>/dev/null || true
./scripts/stop_all.sh 2>/dev/null || true
echo "Done."
