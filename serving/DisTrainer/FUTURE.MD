# Future Optimizations for DisTrainer

This document outlines planned performance and architectural upgrades for the DisTrainer system.

## 1. Torch.compile Integration
Leverage PyTorch 2.0+ compilation stack for maximized H100 throughput.

### Proposed Strategy
- **Sharded Compilation**: Apply `torch.compile(model)` *after* FSDP2 sharding (`fully_shard`) to allow the compiler to optimize across the distributed communication graph.
- **Loss Function Optimization**: Apply `@torch.compile` to the `compute_grpo_loss` function in `components/loss.py`.

### Expected Optimizations
- **Kernel Fusion**: Fusing multiple element-wise operations (e.g., Softmax + Log + Sum) into single Triton kernels to reduce H100 kernel launch overhead.
- **Reduction of Python Overhead**: TorchDynamo will capture the training loop into FX graphs, moving execution from slower Python bytecode to high-speed C++/Triton.
- **CUDA Graphs**: Use `mode="reduce-overhead"` to capture static execution snapshots, significantly reducing CPU-to-GPU latency.

## 2. Distributed Enhancements
- **Tensor Parallelism (TP)**: If scaling to larger models (e.g., 70B), integrate TP within the DeviceMesh to shard individual weight matrices.
- **Overlapped Communication**: Refine FSDP2 settings to overlap the `All-Gather` of Layer N+1 with the computation of Layer N.

## 3. Context Parallelism (CP) for Long Sequences

Enable training with very long sequences (8K-1M+ tokens) by sharding the sequence dimension across GPUs using Ring Attention.

### What is Context Parallelism?

Context Parallelism reduces peak activation memory by:
1. **Sharding the sequence dimension** across a CP group (not just parameters like FSDP)
2. **Ring Attention**: Each GPU holds a slice of the KV cache; attention is computed by rotating KV shards around a "ring" of GPUs
3. **Memory reduction**: From O(seq_len²) to O(seq_len²/cp_degree)

### Implementation Plan

#### Step 1: Update ParallelDims
```python
# mesh.py
@dataclass
class ParallelDims:
    dp: int = 2   # Data parallel
    cp: int = 1   # Context parallel (NEW)
    tp: int = 1   # Tensor parallel
    pp: int = 1   # Pipeline parallel
```

#### Step 2: Build 2D DeviceMesh
```python
def build_mesh(parallel_dims: ParallelDims):
    if parallel_dims.cp > 1:
        # 2D mesh: (dp, cp)
        return init_device_mesh(
            "cuda",
            (parallel_dims.dp, parallel_dims.cp),
            mesh_dim_names=("dp", "cp")
        )
    else:
        return init_device_mesh("cuda", (parallel_dims.dp,), mesh_dim_names=("dp",))
```

#### Step 3: Integrate TorchTitan's Ring Attention
```python
from torch.distributed.tensor.experimental import context_parallel

# In loss computation
if cp_mesh is not None:
    with context_parallel(cp_mesh, input_ids):
        outputs = model(input_ids)  # Ring Attention auto-applied to SDPA
else:
    outputs = model(input_ids)
```

#### Step 4: Update FSDP to use DP submesh
```python
# FSDP uses only the DP dimension when CP is enabled
dp_mesh = mesh["dp"] if "cp" in mesh.mesh_dim_names else mesh
fully_shard(model, mesh=dp_mesh, mp_policy=mp_policy)
```

### Configuration Example
```toml
# 4x H100 with Context Parallelism
[parallel_dims]
dp = 2   # 2 data parallel groups
cp = 2   # Each sequence split across 2 GPUs within group
tp = 1
pp = 1
# Enables ~2x longer sequences (e.g., 16K instead of 8K)
```

### GPU Layout Visualization
```
4 GPUs with dp=2, cp=2:

CP Group 0: [GPU0 ←→ GPU1]  (tokens 0-4095) ←→ (tokens 4096-8191)
CP Group 1: [GPU2 ←→ GPU3]  (tokens 0-4095) ←→ (tokens 4096-8191)
     ↑                ↑
     └── DP sync ─────┘
```

### Requirements
- **PyTorch 2.4+** for stable `context_parallel` APIs
- **TorchTitan** patterns for Ring Attention implementation
- Careful testing with PEFT/LoRA (should be compatible)

### Expected Benefits
| Metric | Without CP | With CP (cp=2) | With CP (cp=4) |
|--------|-----------|----------------|----------------|
| Max Seq Length | 8K | 16K | 32K |
| KV Cache Memory | 100% | 50% | 25% |
| Communication | None | Ring all-to-all | More all-to-all |

## 4. Data Pipeline
- **Prefetching JSONL**: Update `DataLoader` to proactively load and tokenize the next batch in a background thread while the current batch is training.
- **S3/Cloud Loading**: Support streaming data directly from object storage to avoid local disk bottlenecks on multi-node setups.


## 5. Prime Intellect Verifiers Integration
Integrate Prime Intellect's Verifiers and Environment Hub to enhance model testing and reward generation.

### Overview
Use 10-15 code-dominant environments from the Prime Intellect Environment Hub to test model outputs and generate reward signals. This will involve a new feedback loop where problems are fetched from these environments, solved by the model, and then verified.

### Implementation Plan

#### 1. Environment Setup
- Install `verifiers` and `prime-cli` libraries.
- Configure authentication (`PRIME_API_KEY`) in `.env`.

#### 2. Environment Selection & Discovery
- Discover and select 10-15 code-domain environments (e.g., `prime/code-contest`, `prime/mbpp`).
- Create `serving/DisGenerator/verifier_config.json` to manage enabled environments.

#### 3. Integration Logic
- **Verifier Manager**: creates `serving/DisGenerator/verifiers_bridge.py` with a `PrimeVerifierManager` class to handle:
    - Loading environments.
    - Executing solutions against environments using `verify(prompt, solution, env_name)`.
    - Returning scores/feedback.
- **Orchestrator Updates**:
    - Add `VerificationMode` to `batch_orchestrator.py` to fetch prompts from `PrimeVerifierManager` instead of the API queue.
    - Track environment IDs in `Trajectory` objects.

#### 4. Reward Integration
- Update `serving/DisGenerator/reward_functions.py` to include `prime_verifier_reward(content, env_id)`.
- Use verification scores as part of the total reward calculation.

#### 5. Execution Workflow
- Create `serving/run_prime_eval.py` to drive the evaluation loop:
    1. Fetch problem from `PrimeVerifierManager`.
    2. Generate solution via `DisGenerator`.
    3. Verify solution.
    4. Log results and rewards.
