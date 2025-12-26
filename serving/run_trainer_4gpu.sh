#!/bin/bash
# =============================================================================
# Run DisTrainer on 4 GPU setup (GPUs 2-3)
# =============================================================================
# This script runs the DisTrainer using FSDP2 data parallelism on GPUs 2 and 3.
#
# Requirements:
#   - At least 4 GPUs (GPUs 0-1 for generation, GPUs 2-3 for training)
#   - Model weights downloaded or accessible via HuggingFace
#   - Initial policy checkpoint in DisTrainer/models/policy-0-initial
#
# Usage:
#   ./run_trainer_4gpu.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DisTrainer"

# Use GPUs 2 and 3 for training
export CUDA_VISIBLE_DEVICES=2,3

echo "=============================================="
echo "Starting DisTrainer (FSDP2 on GPUs 2,3)"
echo "=============================================="
echo ""
echo "GPU Allocation:"
echo "  GPU 2: FSDP Shard 0"
echo "  GPU 3: FSDP Shard 1"
echo ""
echo "Configuration: config/train_config.toml"
echo ""

# Check if initial policy exists
if [ ! -d "./models/policy-0-initial" ]; then
    echo "ERROR: Initial policy not found at ./models/policy-0-initial"
    echo ""
    echo "Please copy the checkpoint from ToolGRPOTrainer:"
    echo "  cp -r ../ToolGRPOTrainer/grpo-streamed/checkpoint-10 ./models/policy-0-initial"
    exit 1
fi

echo "Found initial policy: ./models/policy-0-initial"
echo ""

# Create data directory if it doesn't exist
mkdir -p ./data/generations

echo "Step 1: Starting distributed training server..."
echo "=============================================="
echo ""
echo "API will be available at http://localhost:8000"
echo ""
echo "Use these commands to interact:"
echo "  curl http://localhost:8000/status                                  # Check status"
echo '  curl -X POST http://localhost:8000/train -H "Content-Type: application/json" -d '"'"'{"num_steps": 1}'"'"'  # Train 1 step'
echo ""

# Start distributed training with 2 GPUs
# Note: CUDA_VISIBLE_DEVICES makes GPU indices 0,1 inside the process
# We add ".." to PYTHONPATH so we can run DisTrainer as a module from inside its directory
export PYTHONPATH=..:$PYTHONPATH
torchrun --nproc_per_node=2 -m DisTrainer.train --config config/train_config.toml
