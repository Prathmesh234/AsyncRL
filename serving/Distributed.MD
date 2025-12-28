# AsyncRL Distributed Architecture

## Overview

AsyncRL implements an **asynchronous distributed reinforcement learning** system for training LLMs with tool-use capabilities. The architecture consists of two main components:

1. **DisGenerator** - Generates trajectories using disaggregated vLLM serving with tool interception
2. **DisTrainer** - Trains the model using GRPO (Group Relative Policy Optimization) with FSDP2

```
┌─────────────────────────────────────────────────────────────────────────────────────────┐
│                           ASYNCRL DISTRIBUTED ARCHITECTURE                               │
├─────────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                          │
│   ┌──────────────────────────────────┐        ┌──────────────────────────────────┐       │
│   │         DisGenerator             │        │           DisTrainer             │       │
│   │     (GPUs 0, 1 or 0-3)           │        │       (GPUs 2, 3 or 4-7)         │       │
│   ├──────────────────────────────────┤        ├──────────────────────────────────┤       │
│   │                                  │        │                                  │       │
│   │  Prefill GPU(s)   Decode GPU(s)  │        │     FSDP2 Data Parallel          │       │
│   │      ↓                 ↓         │        │         Training                 │       │
│   │  ┌────────┐     ┌────────┐       │        │                                  │       │
│   │  │ vLLM   │────▶│ vLLM   │       │        │  ┌─────────┐    ┌─────────┐      │       │
│   │  │Prefill │NCCL │Decode  │       │        │  │ GPU 2   │    │ GPU 3   │      │       │
│   │  └────────┘     └────────┘       │        │  │ (Shard) │    │ (Shard) │      │       │
│   │       │              │           │        │  └────┬────┘    └────┬────┘      │       │
│   │       └──────────────┘           │        │       └──────┬───────┘           │       │
│   │              │                   │        │              │                   │       │
│   │      ┌───────▼───────┐           │        │      ┌───────▼───────┐           │       │
│   │      │   Proxy       │           │        │      │  DataLoader   │           │       │
│   │      │  (Port 10001) │           │        │      │  (watches)    │           │       │
│   │      └───────┬───────┘           │        │      └───────┬───────┘           │       │
│   │              │                   │        │              │                   │       │
│   │      ┌───────▼───────┐           │        │              │                   │       │
│   │      │  Orchestrator │           │        │              │                   │       │
│   │      │  + Tool Calls │           │        │              │                   │       │
│   │      └───────┬───────┘           │        │              │                   │       │
│   │              │                   │        │              │                   │       │
│   └──────────────┼───────────────────┘        └──────────────┼───────────────────┘       │
│                  │                                           │                           │
│                  │                                           │                           │
│                  ▼                                           │                           │
│   ┌──────────────────────────────────────────────────────────┼───────────────────┐       │
│   │                    Shared Data Directory                 │                   │       │
│   │                DisTrainer/data/generations/              │                   │       │
│   │         batch_00000.jsonl, batch_00001.jsonl, ...        │                   │       │
│   └──────────────────────────────────────────────────────────────────────────────┘       │
│                                                                                          │
│   ┌──────────────────────────────────────────────────────────────────────────────┐       │
│   │                        Policy Checkpoint Storage                             │       │
│   │                        DisTrainer/models/                                    │       │
│   │   policy-0-initial  →  policy-1-20251226_131234  →  policy-2-20251226_143456 │       │
│   └──────────────────────────────────────────────────────────────────────────────┘       │
│                                                                                          │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

---

## Hardware Requirements

| Configuration | GPUs | DisGenerator | DisTrainer | Description |
|---------------|------|--------------|------------|-------------|
| **Minimum (4 GPUs)** | 4x H100/A100 | GPU 0-1 (1P1D) | GPU 2-3 | Shared node |
| **Recommended (8 GPUs)** | 8x H100/A100 | GPU 0-3 (2P2D) | GPU 4-7 | Full isolation |
| **Large Scale** | 2+ Nodes | Node 1 | Node 2 | Network-connected |

**Note:** The minimum requirement is **4 GPUs** total: 2 for generation (1 prefill + 1 decode) and 2 for training.

---

## Directory Structure

```
serving/
├── DisGenerator/                   # Trajectory Generation
│   ├── batch_orchestrator.py       # Main orchestrator with tool handling
│   ├── config.py                   # Config with get_latest_policy_path()
│   ├── data_processing.py          # JSONL record building
│   ├── disagg_proxy.py             # P2P NCCL proxy server
│   ├── reward_functions.py         # Reward computation
│   ├── simple_client.py            # Client to run the orchestrator
│   ├── prompts.jsonl               # Input prompts
│   └── scripts/
│       ├── start_all.sh            # Start all components
│       ├── start_prefill.sh        # Start prefill server
│       ├── start_decode.sh         # Start decode server
│       └── stop_all.sh             # Stop all components
│
├── DisTrainer/                     # GRPO Training
│   ├── trainer.py                  # Main trainer class
│   ├── server.py                   # FastAPI server for commands
│   ├── train.py                    # Entry point
│   ├── config/
│   │   └── train_config.toml       # Training configuration
│   ├── components/
│   │   ├── checkpoint.py           # Policy versioning (policy-N-YYYYMMDD_HHMMSS)
│   │   ├── data_loader.py          # Watches for new batch files
│   │   ├── loss.py                 # GRPO loss with action masking
│   │   └── metrics.py              # Training metrics
│   ├── models/
│   │   ├── policy-0-initial/       # Starting checkpoint (from ToolGRPOTrainer)
│   │   └── qwen3/                  # FSDP2 parallelization
│   └── data/
│       └── generations/            # Where batch files are written/read
│
└── Distributed.MD                  # This file
```

---

## Data Flow

### 1. Generation Phase (DisGenerator)

```
prompts.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│               AsyncBatchOrchestrator                    │
├─────────────────────────────────────────────────────────┤
│                                                         │
│   1. Load latest policy from DisTrainer/models/         │
│   2. For each prompt, create N completions (GRPO)       │
│   3. Stream tokens, intercept tool calls                │
│   4. Execute tools (web, code, azure)                   │
│   5. Resume generation with tool results                │
│   6. Compute rewards (tool_reward, char_reward, etc.)   │
│   7. Build trajectory record with:                      │
│      - completion_ids                                   │
│      - old_logprobs (from streaming)                    │
│      - action_mask (1=model, 0=tool)                    │
│      - reward                                           │
│   8. Save to DisTrainer/data/generations/batch_*.jsonl  │
│                                                         │
└─────────────────────────────────────────────────────────┘
```

#### Batch File Sizing

Batch files are created based on **complete GRPO groups** (prompts with all completions):

| Parameter | Default | Description |
|-----------|---------|-------------|
| `batch_size` | 10 | Number of **prompts** per batch file |
| `num_completions_per_prompt` | 4 | Completions generated per prompt |
| **Total trajectories per batch** | **40** | 10 prompts × 4 completions |

A batch file is only written when `batch_size` prompts have **all** their completions ready. This ensures:
1. **Complete GRPO groups** - All 4 completions for a prompt are in the same batch
2. **Proper advantage computation** - Trainer can compute per-group advantages correctly
3. **Atomic training steps** - Each batch file = one training step

### 2. Training Phase (DisTrainer)

```
DisTrainer/data/generations/batch_*.jsonl
      │
      ▼
┌─────────────────────────────────────────────────────────┐
│                      DataLoader                         │
│         (watches for new batch files)                   │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              compute_grpo_loss()                        │
├─────────────────────────────────────────────────────────┤
│   1. Group completions by group_id                      │
│   2. Compute advantages (reward - mean_reward)          │
│   3. Forward pass through model                         │
│   4. Extract log probs (skip where action_mask=0)       │
│   5. Compute policy gradient loss                       │
│   6. Add KL penalty                                     │
│   7. Backward + optimizer step                          │
└───────────────────────┬─────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────┐
│              CheckpointManager.save()                   │
│   Saves as: DisTrainer/models/policy-N-YYYYMMDD_HHMMSS  │
└─────────────────────────────────────────────────────────┘
```

### 3. Policy Versioning

| Version | Directory Name | When Created |
|---------|----------------|--------------|
| 0 | `policy-0-initial` | Copied from ToolGRPOTrainer/checkpoint-10 |
| 1 | `policy-1-20251226_131234` | After first training batch |
| 2 | `policy-2-20251226_143456` | After second training batch |
| N | `policy-N-{timestamp}` | After Nth training batch |

The DisGenerator automatically detects the latest policy via `get_latest_policy_path()` in `config.py`.

---

## JSONL Record Format

Each trajectory saved to `batch_*.jsonl`:

```json
{
  "gen_id": "prompt1-0",
  "group_id": "prompt1",
  "prompt": "<|im_start|>system\n...<|im_end|>\n<|im_start|>user\n...",
  "prompt_ids": [1234, 5678, ...],
  "completion": {
    "text": "<think>...</think>\n<web>{...}</web>\n<tool_result>...</tool_result>\n<solution>...</solution>",
    "completion_ids": [9012, 3456, ...],
    "old_logprobs": [-0.1, -0.2, -0.3, 0.0, 0.0, 0.0, -0.4, ...],
    "action_mask": [1, 1, 1, 0, 0, 0, 1, 1, ...],
    "reward": 2.35
  },
  "metadata": {
    "timestamp": 1735234567.123,
    "status": "COMPLETED",
    "num_turns": 3
  }
}
```

**Key fields:**
- `action_mask`: `1` = model token (include in loss), `0` = tool result token (skip)
- `old_logprobs`: Streaming logprobs for KL penalty calculation
- `reward`: Combined reward from `tool_reward + char_reward + format_reward`

---

## Quick Start Commands

### Option 1: 4 GPU Setup (Minimum)

**Terminal 1 - Start DisGenerator (GPUs 0-1):**
```bash
cd serving/DisGenerator

# Set GPU visibility for generation (GPUs 0-1)
export CUDA_VISIBLE_DEVICES=0,1

# Start disaggregated serving (1 Prefill + 1 Decode)
./scripts/start_all.sh 1p1d

# Wait for "All servers are up and running!" message
```

**Terminal 2 - Run Orchestrator:**
```bash
cd serving/DisGenerator

# Run the batch orchestrator client
uv run python simple_client.py
```

**Terminal 3 - Start DisTrainer (GPUs 2-3):**
```bash
cd serving/DisTrainer

# Set GPU visibility for training (GPUs 2-3)
export CUDA_VISIBLE_DEVICES=2,3

# Start distributed training with 2 GPUs (AUTO-TRAINS when batches arrive!)
# Add ".." to PYTHONPATH to run as module
export PYTHONPATH=..:$PYTHONPATH
torchrun --nproc_per_node=2 -m DisTrainer.train --config config/train_config.toml
```

> **Note:** The trainer now automatically polls for new batch files every 5 seconds and trains when data is available. No need to manually call the `/train` endpoint!

---

### Option 2: 8 GPU Setup (Recommended)

**Terminal 1 - Start DisGenerator (GPUs 0-3):**
```bash
cd serving/DisGenerator

# Set GPU visibility for generation (GPUs 0-3)
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Start disaggregated serving (2 Prefill + 2 Decode)
./scripts/start_all.sh 2p2d

# Wait for "All servers are up and running!" message
```

**Terminal 2 - Run Orchestrator:**
```bash
cd serving/DisGenerator

# Run the batch orchestrator client
uv run python simple_client.py
```

**Terminal 3 - Start DisTrainer (GPUs 4-7):**
```bash
cd serving/DisTrainer

# Set GPU visibility for training (GPUs 4-7)
export CUDA_VISIBLE_DEVICES=4,5,6,7

# Start distributed training with 4 GPUs (AUTO-TRAINS when batches arrive!)
# Add ".." to PYTHONPATH to run as module
export PYTHONPATH=..:$PYTHONPATH
torchrun --nproc_per_node=4 -m DisTrainer.train --config config/train_config.toml
```

> **Note:** The trainer now automatically polls for new batch files every 5 seconds and trains when data is available. No need to manually call the `/train` endpoint!

---

## Auto-Training Behavior

The DisTrainer runs in **fully automatic mode** by default:

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                           AUTO-TRAINING LOOP                                         │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                      │
│   Every 5 seconds, the trainer:                                                      │
│                                                                                      │
│   1. 📦 Checks for new batch files in data/generations/                              │
│                                                                                      │
│   2. If new batch found:                                                             │
│      a. 🏋️  Train on the batch (GRPO loss) → 1 JSONL file = 1 training step          │
│      b. 🗑️  Delete the processed batch file                                          │
│      c. 💾 Save checkpoint ONLY if step % save_interval == 0 (default: every 10)     │
│                                                                                      │
│   3. If no new data:                                                                 │
│      ⏳ Wait and continue polling...                                                 │
│                                                                                      │
│   Meanwhile, the HTTP API is still available for:                                    │
│   - GET /status       → Check training progress                                      │
│   - GET /health       → Health check                                                 │
│   - POST /checkpoint  → Manually trigger checkpoint                                  │
│   - POST /train       → Manually trigger training (overrides auto)                   │
│                                                                                      │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

### Checkpoint Interval

Checkpoints are saved based on `checkpoint.save_interval` in `train_config.toml`:
- **Default: 10** → One policy every 10 batches
- Change to `save_interval = 1` if you want one policy per batch

### Why Delete Batch Files?

Processed batch files are deleted to:
1. **Avoid reprocessing** - Prevents training on the same data twice
2. **Save disk space** - Batch files can be large
3. **Keep things clean** - No manual cleanup needed

If you want to keep batch files for debugging, modify `server.py` and set `auto_train=False` in `distributed_worker()`.

---

## All-in-One Scripts

### Script: `run_generator_4gpu.sh`

Save this to `serving/run_generator_4gpu.sh`:

```bash
#!/bin/bash
# =============================================================================
# Run DisGenerator on 4 GPU setup (GPUs 0-1)
# =============================================================================

set -e
cd "$(dirname "$0")/DisGenerator"

# Use GPUs 0 and 1 for generation
export CUDA_VISIBLE_DEVICES=0,1

echo "=============================================="
echo "Starting DisGenerator (1P1D on GPUs 0,1)"
echo "=============================================="

# Start vLLM servers + proxy in background
./scripts/start_all.sh 1p1d &
SERVERS_PID=$!

# Wait for servers to be ready
sleep 60

echo ""
echo "Starting Orchestrator..."
echo ""

# Run the orchestrator
uv run python simple_client.py

# Cleanup on exit
kill $SERVERS_PID 2>/dev/null || true
```

### Script: `run_trainer_4gpu.sh`

Save this to `serving/run_trainer_4gpu.sh`:

```bash
#!/bin/bash
# =============================================================================
# Run DisTrainer on 4 GPU setup (GPUs 2-3)
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DisTrainer"

# Use GPUs 2 and 3 for training
export CUDA_VISIBLE_DEVICES=2,3

echo "=============================================="
echo "Starting DisTrainer (FSDP2 on GPUs 2,3)"
echo "=============================================="

# Start distributed training
# Add ".." to PYTHONPATH to run as module from inside the directory
export PYTHONPATH=..:$PYTHONPATH
torchrun --nproc_per_node=2 -m DisTrainer.train --config config/train_config.toml
```

### Script: `run_generator_8gpu.sh`

Save this to `serving/run_generator_8gpu.sh`:

```bash
#!/bin/bash
# =============================================================================
# Run DisGenerator on 8 GPU setup (GPUs 0-3)
# =============================================================================

set -e
cd "$(dirname "$0")/DisGenerator"

# Use GPUs 0-3 for generation (2P2D)
export CUDA_VISIBLE_DEVICES=0,1,2,3

echo "=============================================="
echo "Starting DisGenerator (2P2D on GPUs 0,1,2,3)"
echo "=============================================="

# Start vLLM servers + proxy in background
./scripts/start_all.sh 2p2d &
SERVERS_PID=$!

# Wait for servers to be ready
sleep 90

echo ""
echo "Starting Orchestrator..."
echo ""

# Run the orchestrator
uv run python simple_client.py

# Cleanup on exit
kill $SERVERS_PID 2>/dev/null || true
```

### Script: `run_trainer_8gpu.sh`

Save this to `serving/run_trainer_8gpu.sh`:

```bash
#!/bin/bash
# =============================================================================
# Run DisTrainer on 8 GPU setup (GPUs 4-7)
# =============================================================================

set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/DisTrainer"

# Use GPUs 4-7 for training
export CUDA_VISIBLE_DEVICES=4,5,6,7

echo "=============================================="
echo "Starting DisTrainer (FSDP2 on GPUs 4,5,6,7)"
echo "=============================================="

# Start distributed training
# Add ".." to PYTHONPATH to run as module from inside the directory
export PYTHONPATH=..:$PYTHONPATH
torchrun --nproc_per_node=4 -m DisTrainer.train --config config/train_config.toml
```

---

## API Endpoints

### DisTrainer API (Port 8000)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/status` | GET | Training status (step, batches available/processed) |
| `/train` | POST | Trigger training steps |
| `/checkpoint` | POST | Save checkpoint |
| `/load_checkpoint` | POST | Load checkpoint |

**Example - Trigger Training:**
```bash
curl -X POST http://localhost:8000/train \
  -H "Content-Type: application/json" \
  -d '{"num_steps": 10}'
```

**Example - Check Status:**
```bash
curl http://localhost:8000/status
```

### DisGenerator Proxy (Port 10001)

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Proxy health + instance counts |
| `/v1/chat/completions` | POST | OpenAI-compatible chat API |

---

## Configuration

### DisGenerator (`config.py`)

Key settings:
- `model`: Base model name (e.g., `Qwen/Qwen3-4B-Thinking-2507`)
- `adapter_path`: Auto-detected from `DisTrainer/models/policy-*`
- `max_model_len`: 32768 tokens

### DisTrainer (`config/train_config.toml`)

```toml
[model]
name = "Qwen/Qwen3-4B-Thinking-2507"
adapter_path = "./models/policy-0-initial"  # Starting point
use_lora = true

[training]
learning_rate = 5e-6
beta = 0.01  # KL penalty coefficient

[data]
generations_dir = "./data/generations"  # Where batch files are read

[checkpoint]
save_dir = "./models"  # Where policies are saved
save_interval = 10
keep_latest_k = 3
```

---

## Monitoring

### Check Generation Progress
```bash
# Watch batch file creation
watch -n 5 'ls -la serving/DisTrainer/data/generations/'

# Check orchestrator logs
tail -f serving/DisGenerator/logs/proxy.log
```

### Check Training Progress
```bash
# View trainer status
curl http://localhost:8000/status | jq

# List saved policies
ls -la serving/DisTrainer/models/policy-*
```

---

## Troubleshooting

### "No prefill/decode instances available"
- Ensure vLLM servers started successfully
- Check `logs/prefill_*.log` and `logs/decode_*.log`
- Verify CUDA_VISIBLE_DEVICES is set correctly

### "No policy checkpoints found"
- Ensure `policy-0-initial` exists in `DisTrainer/models/`
- Copy it from ToolGRPOTrainer: `cp -r ToolGRPOTrainer/grpo-streamed/checkpoint-10 DisTrainer/models/policy-0-initial`

### Out of Memory (OOM)
- Reduce `max_model_len` in `config.py`
- Use 2P2D instead of 1P1D for generation
- Reduce batch size in orchestrator

### Training not starting
- Check if batch files exist in `DisTrainer/data/generations/`
- Use `curl http://localhost:8000/status` to see batches_available
- Trigger training with `/train` endpoint

---

## Full Continuous Loop

For a fully automated loop:

1. **Start Generator** → Produces batch files
2. **Start Trainer** → Consumes batches, saves new policy
3. **Generator detects new policy** → Reloads and continues
4. **Repeat**

The system is designed for asynchronous operation where generation and training run concurrently on different GPU sets.

---

## TODO

### Length-Based Soft Penalizing Reward Function (DAPO)

> **Paper:** [DAPO: An Open-Source LLM Reinforcement Learning System at Scale](https://arxiv.org/abs/2503.14476) (arXiv:2503.14476)

Implement the **Soft Overlong Punishment** mechanism from DAPO to penalize excessively long responses:

#### Formula

```
R_overlong = -B^((L_generated - L_expected) / penalty)
```

#### Parameters

| Parameter | Description |
|-----------|-------------|
| `R_overlong` | The length-based penalty reward (negative value) |
| `B` | Base of the exponential penalty (hyperparameter, e.g., 2 or e) |
| `L_generated` | Actual length of the generated response (in tokens) |
| `L_expected` | Expected/maximum allowed length threshold |
| `penalty` | Scaling factor to control the steepness of the penalty curve |

#### Explanation

When a response exceeds the predefined maximum length (`L_expected`):
1. A **punishment interval** is activated
2. Within this interval, **longer responses receive greater punishment**
3. This penalty is added to the original rule-based correctness reward
4. The mechanism signals to the model to **avoid excessively long responses**

This is a **"soft punishment"** approach that:
- Does not harshly penalize potentially correct but lengthy reasoning
- Encourages conciseness while preserving the validity of reasoning
- Stabilizes training and reduces reward noise for truncated samples

#### Implementation Status

- [x] ~~Add `L_expected` and `penalty` to `train_config.toml`~~ → Configured in `char_reward()` function
- [x] ~~Implement `compute_overlong_penalty()` in `reward_functions.py`~~ → Integrated into `char_reward()` in `DisGenerator/reward_functions.py`
- [x] ~~Integrate with existing reward computation~~ → Now part of `compute_reward()` via `char_reward()`
- [ ] Consider Overlong Filtering as an alternative (mask loss of truncated samples)

**Current Configuration (in `char_reward()`):**
```python
DAPO_BASE = 2.0              # B: Base of exponential penalty
DAPO_EXPECTED_LENGTH = 8192  # L_expected: Max expected length (chars)
DAPO_PENALTY_SCALE = 2048    # Scaling factor for penalty steepness
DAPO_MAX_PENALTY = 2.0       # Cap on maximum penalty
```

**Reference:** Section 3.4 "Hide and Seek: Overlong Reward Shaping" in the DAPO paper.
