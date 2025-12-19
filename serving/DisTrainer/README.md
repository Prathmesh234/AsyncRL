# DisTrainer

Distributed GRPO Trainer using TorchTitan primitives (FSDP2, DCP).

## Overview

DisTrainer is a training-only component for async RL pipelines. It consumes pre-generated completions from JSONL files and trains the model using GRPO (Group Relative Policy Optimization).

**Model**: Qwen3-4B-Thinking-2507

## Quick Start

### 1. Set GPU Devices

```bash
# Use GPUs 0 and 1 for training
export CUDA_VISIBLE_DEVICES=0,1
```

### 2. Launch the Trainer

```bash
# From the DisTrainer directory
cd serving/DisTrainer

# Launch with torchrun (2 GPUs)
torchrun --nproc_per_node=2 -m DisTrainer.train --config config/train_config.toml
```

### 3. Trigger Training

The trainer exposes HTTP endpoints. Once running, you can trigger training via:

```bash
# Train for 5 steps
curl -X POST "http://localhost:8000/train?num_steps=5" \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 5}'

# Check status
curl http://localhost:8000/status

# Save checkpoint
curl -X POST http://localhost:8000/checkpoint
```

## Configuration

Edit `config/train_config.toml`:

```toml
# TODO: Update the path below to your local model directory
[model]
name = "./models/Qwen3-4B-Thinking-2507"  # <-- FILL THIS: Path to local model
use_lora = true
lora_r = 8
lora_alpha = 16

[training]
learning_rate = 5e-6
beta = 0.01  # KL penalty

[parallel_dims]
dp = 2  # Number of GPUs for data parallelism

[data]
generations_dir = "./data/generations"

[checkpoint]
save_dir = "./checkpoints"
save_interval = 10
```

## Data Format

Place JSONL files in `./data/generations/`:

```
data/generations/
├── batch_00001.jsonl
├── batch_00002.jsonl
└── ...
```

Each line in a JSONL file should be:

```json
{
  "gen_id": 101,
  "prompt": "Why is the sky blue?",
  "prompt_ids": [12800, 15, 245],
  "completions": [
    {
      "text": "Rayleigh scattering...",
      "completion_ids": [45, 12, 98],
      "reward": 0.85,
      "old_logprobs": [-0.12, -0.45]
    },
    {
      "text": "Because of air...",
      "completion_ids": [88, 99, 102],
      "reward": 0.32,
      "old_logprobs": [-0.18, -0.52]
    }
  ]
}
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/train` | POST | Execute training steps |
| `/status` | GET | Get trainer status |
| `/checkpoint` | POST | Save checkpoint |
| `/load_checkpoint` | POST | Load checkpoint |
| `/health` | GET | Health check |

## GPU Configuration

### Single Node, 2 GPUs (Default)

```bash
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node=2 -m DisTrainer.train
```

### Single Node, 4 GPUs

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node=4 -m DisTrainer.train
```

Update `config/train_config.toml`:
```toml
[parallel_dims]
dp = 4
```

### Specific GPUs (e.g., GPUs 2 and 3)

```bash
export CUDA_VISIBLE_DEVICES=2,3
torchrun --nproc_per_node=2 -m DisTrainer.train
```

## Requirements

```
torch >= 2.3.0
transformers
peft
fastapi
uvicorn
tomli
```

## Project Structure

```
DisTrainer/
├── config/
│   └── train_config.toml
├── models/
│   └── qwen3/
│       └── parallelize.py    # FSDP2 sharding
├── components/
│   ├── checkpoint.py         # DCP checkpointing
│   ├── data_loader.py        # JSONL loading
│   ├── loss.py               # GRPO loss
│   └── metrics.py            # Logging
├── trainer.py                # Main Trainer class
├── server.py                 # FastAPI endpoints
├── train.py                  # Entry point
├── mesh.py                   # DeviceMesh utilities
└── README.md
```
