#!/bin/bash
# =============================================================================
# Run DisTrainer on 8 GPU setup (GPUs 4-7)
# =============================================================================
# This script runs the DisTrainer using FSDP2 data parallelism on GPUs 4-7.
#
# Requirements:
#   - At least 8 GPUs (GPUs 0-3 for generation, GPUs 4-7 for training)
#   - Model weights downloaded or accessible via HuggingFace
#   - Initial policy checkpoint in DisTrainer/models/policy-0-initial
#
# Usage:
#   ./run_trainer_8gpu.sh
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DisTrainer"

# Use GPUs 4-7 for training
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "=============================================="
echo "Starting DisTrainer (FSDP2 on GPUs 4,5,6,7)"
echo "=============================================="
echo ""
echo "GPU Allocation:"
echo "  GPU 4: FSDP Shard 0"
echo "  GPU 5: FSDP Shard 1"
echo "  GPU 6: FSDP Shard 2"
echo "  GPU 7: FSDP Shard 3"
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

# Update config for 4 GPU training
# Note: We need to update parallel_dims.dp in the config
CONFIG_FILE="config/train_config.toml"
if grep -q "dp = 4" "$CONFIG_FILE"; then
    echo "Config already set for 4 GPUs"
else
    echo "Note: Update config/train_config.toml to set dp = 4 for 4-GPU training"
fi

# Start distributed training with 4 GPUs
# Note: CUDA_VISIBLE_DEVICES makes GPU indices 0,1,2,3 inside the process
# We add ".." to PYTHONPATH so we can run DisTrainer as a module from inside its directory
export PYTHONPATH=..:$PYTHONPATH
uv run torchrun --nproc_per_node=4 -m DisTrainer.train --config config/train_config.toml
