# DisTrainer - Distributed GRPO Trainer (TorchTitan Architecture)

## Overview

`DisTrainer` is a distributed training system for reinforcement learning using GRPO (Group Relative Policy Optimization). It is **training-only**—generation is handled by a separate process. This trainer follows TorchTitan's architecture patterns for production-grade distributed training.

Unlike `ToolGRPOTrainer` which handles both generation and training in a single process, DisTrainer:
- Consumes pre-generated completions from JSONL files
- Uses native PyTorch distributed primitives (FSDP2, DCP)
- Exposes HTTP endpoints for async training control
- Follows TorchTitan's modular, scalable design

---

## Architecture Principles (Inspired by TorchTitan)

### 1. **Separation of Concerns**
- **Model definition** lives in one place (e.g., `models/qwen3/model.py`)
- **Parallelization logic** lives separately (`models/qwen3/parallelize.py`)
- **Training orchestration** is in `trainer.py`
- **Checkpointing** is abstracted into `CheckpointManager`

### 2. **Composable FSDP2**
- Uses `torch.distributed._composable.fsdp.fully_shard` 
- Applied per-layer for fine-grained control (following TorchTitan's `parallelize.py` pattern)
- Supports mixed precision and gradient checkpointing

### 3. **Distributed Checkpointing (DCP)**
- Uses `torch.distributed.checkpoint` for sharded saves/loads
- Avoids rank-0 bottleneck by having each GPU save its own shard
- Async checkpoint writing for minimal training interruption

### 4. **Configuration-Driven**
- TOML-based configuration (TorchTitan style)
- All hyperparameters, parallelism settings, and paths defined in config
- No hardcoded values in training code

---

## Component Breakdown

```
DisTrainer/
├── config/
│   └── train_config.toml       # Training configuration (TOML)
├── models/
│   └── qwen3/
│       ├── model.py            # Qwen3 model definition (or use transformers)
│       └── parallelize.py      # FSDP2 application logic
├── components/
│   ├── checkpoint.py           # CheckpointManager (DCP-based)
│   ├── data_loader.py          # JSONL file monitoring and batching
│   ├── loss.py                 # GRPO loss computation
│   └── metrics.py              # Metrics logging
├── trainer.py                  # Main Trainer class
├── server.py                   # FastAPI server for endpoints
├── train.py                    # Entry point (torchrun compatible)
└── AGENTS.MD                   # This file
```

---

## Data Flow

```
┌─────────────────────────┐
│  Generator Process      │ (separate, not part of DisTrainer)
│  (writes JSONL files)   │
└───────────┬─────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  Shared Directory               │
│  ./data/generations/            │
│    ├── batch_00001.jsonl        │
│    ├── batch_00002.jsonl        │
│    └── batch_00003.jsonl        │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  DataLoader (monitors dir)      │
│  - Tracks processed files       │
│  - Loads new batches on demand  │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  Trainer (FSDP2, 2 GPUs)        │
│  ├── Compute GRPO loss          │
│  ├── Backward pass              │
│  ├── Optimizer step             │
│  └── CheckpointManager          │
└───────────┬─────────────────────┘
            │
            ▼
┌─────────────────────────────────┐
│  ./checkpoints/                 │
│    └── step_XXXX/               │
│         ├── __0_0.dist_cp.pt    │ (shard from GPU 0)
│         ├── __1_0.dist_cp.pt    │ (shard from GPU 1)
│         └── metadata.pt         │
└─────────────────────────────────┘
```

---

## JSONL Data Format

Each line in a JSONL file represents **one prompt with N completions** (a "group" for GRPO):

```jsonl
{"gen_id": 101, "prompt": "Why is the sky blue?", "prompt_ids": [12800, 15, 245], "completions": [{"text": "Rayleigh scattering...", "completion_ids": [45, 12, 98], "reward": 0.85, "old_logprobs": [-0.12, -0.45]}, {"text": "Because of air...", "completion_ids": [88, 99, 102], "reward": 0.32, "old_logprobs": [-0.18, -0.52]}], "metadata": {"timestamp": "2025-12-18T15:00:00Z"}}
```

### Field Descriptions

| Field | Type | Description |
|-------|------|-------------|
| `gen_id` | int | Unique generation batch ID |
| `prompt` | str | Original prompt text |
| `prompt_ids` | List[int] | Tokenized prompt |
| `completions` | List[dict] | N completions for this prompt |
| `completions[i].text` | str | Completion text |
| `completions[i].completion_ids` | List[int] | Tokenized completion |
| `completions[i].reward` | float | Reward score (from reward model) |
| `completions[i].old_logprobs` | List[float] | Per-token log probs from generator (for KL) |
| `metadata` | dict | Optional metadata |

**Design Rationale**: 
- Grouping completions by prompt is natural for GRPO (which computes **group-relative advantages**)
- Including `old_logprobs` allows KL penalty computation without re-running the old policy

---

## GRPO Loss Computation

GRPO (Group Relative Policy Optimization) uses **advantages relative to the group mean** instead of a baseline:

### Algorithm

For each group (prompt + N completions):

1. **Compute Group Statistics**:
   ```python
   rewards = [c["reward"] for c in group["completions"]]
   mean_reward = np.mean(rewards)
   std_reward = np.std(rewards)
   ```

2. **Compute Normalized Advantages**:
   ```python
   advantages = [(r - mean_reward) / (std_reward + 1e-8) for r in rewards]
   ```

3. **Get Current Policy Log Probs**:
   ```python
   # Run forward pass with current policy
   tokens = torch.cat([prompt_ids, completion_ids])
   logits = model(tokens)
   new_logprobs = F.log_softmax(logits, dim=-1)
   ```

4. **Compute Policy Gradient Loss**:
   ```python
   # Per completion
   policy_loss = -advantage * sum(new_logprobs)
   ```

5. **Add KL Penalty** (optional):
   ```python
   kl = sum(old_logprobs - new_logprobs)
   total_loss = policy_loss + beta * kl
   ```

### Implementation in `loss.py`

```python
def compute_grpo_loss(
    model, 
    group_batch: List[dict], 
    beta: float = 0.01
) -> torch.Tensor:
    """
    Computes GRPO loss for a batch of groups.
    
    Args:
        model: Policy model (FSDP2-wrapped)
        group_batch: List of groups (each group = 1 prompt + N completions)
        beta: KL penalty coefficient
    
    Returns:
        Scalar loss tensor
    """
    total_loss = 0.0
    
    for group in group_batch:
        # 1. Extract rewards and compute advantages
        rewards = torch.tensor([c["reward"] for c in group["completions"]])
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)
        
        # 2. Forward pass for each completion
        for i, completion in enumerate(group["completions"]):
            # Concatenate prompt + completion
            input_ids = torch.tensor(
                group["prompt_ids"] + completion["completion_ids"]
            ).cuda()
            
            # Get new logprobs
            with torch.cuda.amp.autocast(dtype=torch.bfloat16):
                logits = model(input_ids)
                new_logprobs = F.log_softmax(logits, dim=-1)
            
            # 3. Compute policy loss
            old_logprobs = torch.tensor(completion["old_logprobs"]).cuda()
            action_logprobs = new_logprobs[range(len(input_ids)), input_ids]
            
            policy_loss = -advantages[i] * action_logprobs.sum()
            
            # 4. KL penalty
            kl = (old_logprobs - action_logprobs).sum()
            
            total_loss += policy_loss + beta * kl
    
    return total_loss / len(group_batch)
```

---

## Distributed Setup (TorchTitan Style)

### 1. **DeviceMesh Creation (`mesh.py`)**

```python
from torch.distributed.device_mesh import init_device_mesh

class ParallelDims:
    """TorchTitan-style parallel dimension config"""
    def __init__(self, dp: int = 2, tp: int = 1, pp: int = 1):
        self.dp = dp  # Data parallel
        self.tp = tp  # Tensor parallel
        self.pp = pp  # Pipeline parallel
        self.world_size = dp * tp * pp

def build_mesh(parallel_dims: ParallelDims):
    """
    For 2 GPUs with data-parallel only:
        ParallelDims(dp=2, tp=1, pp=1)
    Returns a 1D mesh: ["dp"]
    """
    return init_device_mesh(
        "cuda",
        (parallel_dims.dp,),
        mesh_dim_names=("dp",)
    )
```

### 2. **FSDP2 Application (`models/qwen3/parallelize.py`)**

Following TorchTitan's `llama/infra/parallelize.py` pattern:

```python
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy

def apply_fsdp(
    model: nn.Module,
    mesh,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32
):
    """
    Applies FSDP2 to Qwen3 model.
    Shards each TransformerBlock individually.
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype
    )
    
    # Shard each transformer layer
    for layer in model.model.layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
    
    # Shard embedding and LM head
    fully_shard(model.model.embed_tokens, mesh=mesh, mp_policy=mp_policy)
    fully_shard(model.lm_head, mesh=mesh, mp_policy=mp_policy)
    
    # Final outer wrap
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    
    return model
```

### 3. **CheckpointManager (`components/checkpoint.py`)**

```python
import torch.distributed.checkpoint as dcp
from pathlib import Path

class CheckpointManager:
    """TorchTitan-style checkpoint manager using DCP"""
    
    def __init__(self, checkpoint_dir: str, keep_latest_k: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_latest_k = keep_latest_k
    
    def save(self, state_dict: dict, step: int):
        """
        Saves sharded checkpoint using DCP.
        Each rank saves its own shard.
        """
        checkpoint_path = self.checkpoint_dir / f"step_{step:06d}"
        checkpoint_path.mkdir(exist_ok=True)
        
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path)
        )
        
        # Cleanup old checkpoints
        self._cleanup_old_checkpoints()
    
    def load(self, step: int, state_dict: dict):
        """Loads checkpoint from step"""
        checkpoint_path = self.checkpoint_dir / f"step_{step:06d}"
        
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path)
        )
    
    def _cleanup_old_checkpoints(self):
        """Keep only the latest K checkpoints"""
        checkpoints = sorted(self.checkpoint_dir.glob("step_*"))
        if len(checkpoints) > self.keep_latest_k:
            for ckpt in checkpoints[:-self.keep_latest_k]:
                shutil.rmtree(ckpt)
```

---

## Trainer Class (`trainer.py`)

```python
class Trainer:
    """
    Main training orchestrator. Inspired by TorchTitan's Trainer class.
    """
    def __init__(self, config):
        self.config = config
        
        # Distributed setup
        self.mesh = build_mesh(config.parallel_dims)
        
        # Model
        self.model = self._build_model()
        self.model = apply_fsdp(self.model, self.mesh)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate
        )
        
        # Checkpointing
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint.save_dir
        )
        
        # Data loader
        self.data_loader = DataLoader(config.data.generations_dir)
        
        # Training state
        self.step = 0
    
    def _build_model(self):
        """Load Qwen3 model"""
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            torch_dtype=torch.bfloat16
        )
        
        # Apply LoRA if configured
        if self.config.model.use_lora:
            from peft import get_peft_model, LoraConfig
            lora_config = LoraConfig(
                r=self.config.model.lora_r,
                lora_alpha=self.config.model.lora_alpha,
                target_modules=["q_proj", "v_proj"]
            )
            model = get_peft_model(model, lora_config)
        
        return model
    
    def train_step(self) -> dict:
        """
        Executes one training step.
        Returns metrics dict.
        """
        # Load next batch
        batch = self.data_loader.get_next_batch()
        if batch is None:
            return {"status": "no_data"}
        
        # Compute loss
        loss = compute_grpo_loss(
            self.model,
            batch,
            beta=self.config.training.beta
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        self.step += 1
        
        # Checkpoint if needed
        if self.step % self.config.checkpoint.save_interval == 0:
            self.save_checkpoint()
        
        return {
            "step": self.step,
            "loss": loss.item(),
            "batches_processed": len(batch)
        }
    
    def save_checkpoint(self):
        """Saves checkpoint using DCP"""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step
        }
        self.checkpoint_manager.save(state_dict, self.step)
```

---

## Configuration (TOML)

`config/train_config.toml`:

```toml
# Model Configuration
[model]
name = "Qwen/Qwen3-4B-Thinking-2507"  # Qwen3 4B Thinking model
use_lora = true
lora_r = 8
lora_alpha = 16
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]  # LoRA targets

# Training Hyperparameters
[training]
learning_rate = 5e-6
beta = 0.01  # KL penalty coefficient
max_grad_norm = 1.0
num_generations_per_prompt = 4

# Distributed Configuration
[parallel_dims]
dp = 2   # Data parallel across 2 GPUs
tp = 1   # No tensor parallelism
pp = 1   # No pipeline parallelism

[distributed]
backend = "nccl"

# Data Configuration
[data]
generations_dir = "./data/generations"
max_sequence_length = 4096

# Checkpoint Configuration
[checkpoint]
save_dir = "./checkpoints"
save_interval = 10  # Save every 10 steps
keep_latest_k = 3   # Keep last 3 checkpoints
```

---

## Qwen3-4B Thinking Model Compatibility

### Architecture
The Qwen3-4B thinking model (`Qwen/Qwen3-4B-Thinking-2507`) is fully compatible with our FSDP2 trainer:
- **36 transformer layers** (same as standard Qwen3)
- **Grouped Query Attention (GQA)**: 32 query heads, 8 KV heads
- **Context length**: 256K tokens native (262,144 tokens)
- **4B parameters** (3.6B non-embedding)
- **Enhanced for reasoning**: Optimized for logical reasoning, mathematics, science, and coding tasks

### Thinking Mode Behavior
The model uses special tokens `<think>` and `</think>` to show internal reasoning:

**Example Generation**:
```
<think>
Let me break this down step by step:
1. First, I need to understand what the user is asking
2. Then, I'll consider different approaches
3. Finally, I'll synthesize a clear answer
</think>

Based on my analysis, the answer is...
```

### GRPO Training Considerations

**During Generation** (handled by your separate Generator):
- The model may output `<think>` tokens as part of Chain-of-Thought reasoning
- Your reward model can either:
  - **Reward the entire sequence** (prompt + thinking + answer)
  - **Reward only the final answer** (excluding `<think>` blocks)
  - **Separately reward reasoning quality** (advanced approach)

**During Training** (in DisTrainer):
- The trainer treats `<think>` tokens like any other token
- They're included in `completion_ids` and trained via GRPO loss
- The policy learns when to "think" based on reward signals

**Recommendation**: Include thinking tokens in your completions—this allows the policy to learn:
- When to use internal reasoning
- How much reasoning is needed
- Quality of reasoning chains

---

## HTTP API (`server.py`)

FastAPI server that wraps the Trainer:

```python
from fastapi import FastAPI
import uvicorn

app = FastAPI()

# Global trainer instance (initialized on startup)
trainer = None

@app.on_event("startup")
async def startup():
    global trainer
    config = load_config("config/train_config.toml")
    trainer = Trainer(config)

@app.post("/train")
async def train(num_steps: int = 1):
    """Execute training steps"""
    metrics_list = []
    for _ in range(num_steps):
        metrics = trainer.train_step()
        metrics_list.append(metrics)
    
    return {
        "status": "success",
        "steps_completed": num_steps,
        "metrics": metrics_list
    }

@app.get("/status")
async def status():
    """Get trainer status"""
    return {
        "step": trainer.step,
        "batches_available": trainer.data_loader.count_available(),
        "model_loaded": True
    }

@app.post("/checkpoint")
async def checkpoint():
    """Save checkpoint immediately"""
    trainer.save_checkpoint()
    return {"status": "saved", "step": trainer.step}
```

---

## Entry Point (`train.py`)

For launching with `torchrun`:

```python
import torch.distributed as dist
from server import app
import uvicorn

def main():
    # Initialize distributed
    dist.init_process_group(backend="nccl")
    
    # Only rank 0 runs the server
    if dist.get_rank() == 0:
        uvicorn.run(app, host="0.0.0.0", port=8000)
    else:
        # Other ranks wait for RPC calls from rank 0
        while True:
            dist.barrier()

if __name__ == "__main__":
    main()
```

---

## Usage

### 1. Launch Trainer

```bash
torchrun --nproc_per_node=2 train.py
```

### 2. Trigger Training

```bash
curl -X POST http://localhost:8000/train?num_steps=5
```

### 3. Check Status

```bash
curl http://localhost:8000/status
```

---

## Implementation Checklist

### Core Components
- [ ] `mesh.py` - DeviceMesh and ParallelDims
- [ ] `models/qwen3/parallelize.py` - FSDP2 application
- [ ] `components/checkpoint.py` - CheckpointManager
- [ ] `components/data_loader.py` - JSONL file watcher
- [ ] `components/loss.py` - GRPO loss computation
- [ ] `trainer.py` - Main Trainer class
- [ ] `server.py` - FastAPI endpoints
- [ ] `train.py` - Entry point

### Configuration
- [ ] `config/train_config.toml` - Training configuration

### Testing
- [ ] Unit tests for loss computation
- [ ] Integration test for full training loop
- [ ] Checkpoint save/load test

---

## Key Differences from Vanilla TorchTitan

| Feature | TorchTitan | DisTrainer |
|---------|-----------|------------|
| **Data Source** | Standard datasets (C4, etc.) | JSONL files from Generator |
| **Loss Function** | Cross-entropy (pretraining) | GRPO (policy gradient + KL) |
| **Activation** | CLI (`torchrun train.py`) | HTTP endpoints |
| **Use Case** | Pretraining LLMs | RL fine-tuning |

---

## Dependencies

```txt
torch >= 2.3.0
transformers
peft
fastapi
uvicorn
tomli  # For TOML parsing
```

---

## Notes

- **Stateful Training**: Unlike typical API servers, this trainer maintains state (model, optimizer) across requests
- **Single-Node Only**: Current design assumes 2 GPUs on one machine. Multi-node requires extending the mesh to 2D.
- **No Generation**: This trainer does **not** generate completions. That's handled by a separate Generator process.
- **Reward Computation**: Rewards must be pre-computed and included in the JSONL files.
