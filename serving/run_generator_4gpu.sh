#!/bin/bash
# =============================================================================
# Run DisGenerator on 4 GPU setup (GPUs 0-1)
# =============================================================================
# This script runs the DisGenerator using a 1P1D (1 Prefill + 1 Decode)
# configuration on GPUs 0 and 1.
#
# Requirements:
#   - At least 4 GPUs (GPUs 0-1 for generation, GPUs 2-3 for training)
#   - uv installed (https://docs.astral.sh/uv/)
#   - Model weights downloaded or accessible via HuggingFace
#
# Usage:
#   ./run_generator_4gpu.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DisGenerator"

# Use GPUs 0 and 1 for generation
export CUDA_VISIBLE_DEVICES=0,1

echo "=============================================="
echo "Starting DisGenerator (1P1D on GPUs 0,1)"
echo "=============================================="
echo ""
echo "GPU Allocation:"
echo "  GPU 0: Prefill Server"
echo "  GPU 1: Decode Server"
echo ""

# Create logs directory
mkdir -p logs

# Start vLLM servers + proxy
echo "Step 1: Starting disaggregated vLLM servers..."
./scripts/start_all.sh 1p1d &
SERVERS_PID=$!

# Give servers time to initialize
echo "Step 2: Waiting for servers to initialize (60 seconds)..."
sleep 60

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
