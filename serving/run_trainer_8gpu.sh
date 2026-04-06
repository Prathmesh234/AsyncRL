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

# Check if an initial policy exists (any policy-N directory will do)
MODELS_DIR="./models"
mkdir -p "$MODELS_DIR"
if ! ls "$MODELS_DIR"/policy-* 1>/dev/null 2>&1; then
    echo "WARNING: No initial policy checkpoint found in $MODELS_DIR"
    echo ""
    echo "The trainer will create a fresh LoRA adapter from the base model."
    echo "To start from a pre-trained adapter, place it at:"
    echo "  $MODELS_DIR/policy-0-initial/"
    echo "(must contain adapter_config.json and adapter_model.safetensors)"
    echo ""
    echo "Continuing with --use-base-model flag..."
    USE_BASE_MODEL="--use-base-model"
else
    echo "Found existing policy checkpoint(s) in $MODELS_DIR"
    USE_BASE_MODEL=""
fi
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
uv run torchrun --nproc_per_node=4 -m DisTrainer.train --config config/train_config.toml $USE_BASE_MODEL
