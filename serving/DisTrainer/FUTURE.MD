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

## 3. Data Pipeline
- **Prefetching JSONL**: Update `DataLoader` to proactively load and tokenize the next batch in a background thread while the current batch is training.
- **S3/Cloud Loading**: Support streaming data directly from object storage to avoid local disk bottlenecks on multi-node setups.

