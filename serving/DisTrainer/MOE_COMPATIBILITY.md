# MoE Compatibility Analysis for DisTrainer

## Executive Summary

This document outlines the changes required to make DisTrainer compatible with Mixture-of-Experts (MoE) models, specifically **Arcee Trinity Mini**, while training only the non-MLP layers.

## Arcee Trinity Mini Architecture

**Model Specifications:**
- **Total Parameters:** 26B (3B activated per token)
- **Architecture:** Sparse MoE based on DeepSeekMoE design
- **MoE Configuration:**
  - 128 routed experts per MoE layer
  - 8 experts activated per token
  - 1 shared expert (always active)
  - First 2 layers are **dense** (not MoE)
- **Routing:** Sigmoid routing with aux-loss-free load balancing
- **Context Window:** 128K tokens
- **Special Features:** Interleaved local/global attention, gated attention, depth-scaled sandwich norm

**Sources:**
- [Arcee AI Trinity Models](https://www.arcee.ai/trinity)
- [Trinity Large: 400B Sparse MoE](https://www.arcee.ai/blog/trinity-large)

## Current Trainer Limitations

### 1. Model-Specific FSDP Implementation
**File:** `models/qwen3/parallelize.py`

**Issue:** The current `apply_fsdp()` function assumes a standard transformer architecture:
```python
# Current code assumes:
model.model.layers  # Standard transformer blocks
model.model.embed_tokens
model.lm_head
```

**Problem for MoE:**
- MoE models have different layer structures with expert modules
- Expert layers need specialized sharding strategies (Expert Parallelism)
- Non-MoE and MoE layers require different process groups

### 2. No Support for Selective Layer Freezing
**File:** `trainer.py`

**Issue:** Current implementation trains all LoRA parameters without selective freezing.

**Required for MoE:**
- Need to **freeze MLP/expert layers** (because vLLM/sglang don't support adapters on experts yet)
- Only train **attention layers, layer norms, and embedding/head**
- Set `requires_grad=False` for expert parameters

### 3. No Expert Parallelism (EP) Support

**Issue:** FSDP2 alone is insufficient for large MoE models.

**Required:**
- Combine FSDP (for non-expert layers) with Expert Parallelism
- Use 3D device mesh: `[EP dimension, DP dimension, replicate dimension]`
- Separate process groups for expert vs non-expert layers

## Required Code Changes

### Change 1: Generalize FSDP Module

**Action:** Rename `models/qwen3/` to `models/parallelization/` and make it model-agnostic.

**New structure:**
```
models/
├── __init__.py
├── parallelization/
│   ├── __init__.py
│   ├── dense_model.py      # For standard transformers (Qwen, Llama, etc.)
│   └── moe_model.py         # For MoE models (Arcee Trinity, DeepSeek, etc.)
```

### Change 2: Implement MoE-Specific FSDP

**File:** `models/parallelization/moe_model.py`

**Key Features:**
```python
def apply_fsdp_moe(
    model: nn.Module,
    mesh,
    freeze_experts: bool = True,
    expert_parallel_dim: str = "ep",
    data_parallel_dim: str = "dp"
):
    """
    Apply FSDP2 with expert parallelism for MoE models.

    Args:
        model: MoE model (e.g., Arcee Trinity Mini)
        mesh: 2D or 3D DeviceMesh with EP and DP dimensions
        freeze_experts: If True, freeze all expert/MLP layers
        expert_parallel_dim: Name of expert parallel dimension in mesh
        data_parallel_dim: Name of data parallel dimension in mesh
    """
    # 1. Freeze expert/MLP layers if requested
    if freeze_experts:
        freeze_moe_experts(model)

    # 2. Wrap each layer with appropriate strategy
    for layer in model.model.layers:
        if is_moe_layer(layer):
            # Expert Parallelism for MoE layers
            apply_expert_parallelism(layer, mesh, expert_parallel_dim)
        else:
            # Standard FSDP for dense layers
            fully_shard(layer, mesh=mesh.get_submesh(data_parallel_dim))

    # 3. Shard embeddings and LM head
    fully_shard(model.model.embed_tokens, mesh=mesh.get_submesh(data_parallel_dim))
    fully_shard(model.lm_head, mesh=mesh.get_submesh(data_parallel_dim))

    # 4. Final outer wrap
    fully_shard(model, mesh=mesh)
```

### Change 3: Add Layer Freezing Utility

**File:** `models/parallelization/freezing.py`

```python
def freeze_moe_experts(model: nn.Module):
    """
    Freeze all expert/MLP layers in an MoE model.

    This is required because current vLLM/sglang don't support
    LoRA adapters on expert layers.
    """
    for name, param in model.named_parameters():
        # Freeze expert layers (common patterns)
        if any(pattern in name.lower() for pattern in [
            'expert', 'mlp', 'gate', 'moe_layer',
            'shared_expert', 'routed_expert'
        ]):
            param.requires_grad = False

    # Verify frozen parameters
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable: {trainable_params:,} / Total: {total_params:,} "
          f"({100 * trainable_params / total_params:.2f}%)")
```

### Change 4: Update ParallelDims Configuration

**File:** `mesh.py`

**Add EP support:**
```python
@dataclass
class ParallelDims:
    dp: int = 1  # Data Parallel
    tp: int = 1  # Tensor Parallel (unused for now)
    pp: int = 1  # Pipeline Parallel (unused)
    ep: int = 1  # Expert Parallel (NEW for MoE)

    def build_mesh(self) -> DeviceMesh:
        """Build device mesh with optional EP dimension."""
        world_size = self.dp * self.tp * self.pp * self.ep

        if self.ep > 1:
            # 3D mesh: [EP, DP, replicate]
            return init_device_mesh(
                "cuda",
                mesh_shape=(self.ep, self.dp),
                mesh_dim_names=("ep", "dp")
            )
        else:
            # 2D mesh: [DP]
            return init_device_mesh(
                "cuda",
                mesh_shape=(self.dp,),
                mesh_dim_names=("dp",)
            )
```

### Change 5: Update Trainer Configuration

**File:** `config/train_config.toml`

**Add MoE configuration:**
```toml
[model]
name = "arcee-ai/Arcee-Trinity-Mini"
use_lora = true
lora_r = 8
lora_alpha = 16
# For MoE: Only target attention layers, not MLPs
target_modules = ["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
adapter_path = "./models/policy-0-initial"

# NEW: MoE-specific config
model_type = "moe"  # or "dense"
freeze_experts = true  # Don't train expert/MLP layers

[parallel_dims]
dp = 4   # Data parallel
tp = 1   # Tensor parallel (unused)
pp = 1   # Pipeline parallel (unused)
ep = 1   # Expert parallel (set to 2+ for large MoE)
```

### Change 6: Update Trainer to Use Generic Parallelization

**File:** `trainer.py` (line 15)

**Current:**
```python
from .models.qwen3 import apply_fsdp
```

**Updated:**
```python
from .models.parallelization import get_fsdp_strategy

# In _build_model():
if config.model.model_type == "moe":
    self.model = apply_fsdp_moe(
        self.model,
        self.mesh,
        freeze_experts=config.model.freeze_experts
    )
else:
    self.model = apply_fsdp_dense(self.model, self.mesh)
```

## Testing Strategy

### Phase 1: Validate Generalized Dense Model
1. Refactor `qwen3` module to `parallelization/dense_model.py`
2. Test with current Qwen3 model
3. Verify no regression in training

### Phase 2: Add MoE Support (Without Real MoE Model)
1. Implement `moe_model.py` with freezing logic
2. Mock MoE layer detection
3. Unit test layer freezing

### Phase 3: Test with Arcee Trinity Mini
1. Download Arcee Trinity Mini model
2. Configure for MoE training
3. Run training loop with frozen experts
4. Verify only attention layers are being updated

## Known Limitations & Workarounds

### Limitation 1: FSDP Nested Wrapping
**Issue:** FSDP doesn't fully support nested wrapping with different process groups for EP and DP.

**Workaround:** Use Hybrid Sharded Data Parallel (HSDP) or manual sharding of expert layers.

**Reference:** [PyTorch Issue #149396](https://github.com/pytorch/pytorch/issues/149396)

### Limitation 2: Traditional Layer Freezing Not Fully Supported
**Issue:** FSDP has limitations with traditional layer freezing.

**Workaround:** Set `requires_grad=False` BEFORE applying FSDP wrapping.

**Reference:** [PyTorch FSDP Discussion](https://www.linkedin.com/posts/pytorch_traditional-layer-freezing-fine-tuning-is-activity-6981340269786898432-HyAN)

### Limitation 3: Generator LoRA Adapter Limitations
**Issue:** vLLM/sglang don't support LoRA adapters on MLP/expert layers yet.

**Impact:** Must train only attention layers for now.

**Future:** When vLLM/sglang support expert adapters, remove `freeze_experts` flag.

## Migration Path

### Immediate (Now):
1. ✅ Rename `models/qwen3/` → `models/parallelization/dense_model.py`
2. ✅ Add `model_type` config option
3. ✅ Keep current Qwen3 training working

### Short-term (1-2 weeks):
1. Implement `moe_model.py` with layer freezing
2. Add EP dimension to `ParallelDims`
3. Test with Arcee Trinity Mini (frozen experts)

### Long-term (When vLLM supports expert adapters):
1. Remove `freeze_experts` constraint
2. Train full MoE model with adapters on experts
3. Implement full Expert Parallelism across multiple nodes

## Resources

### Arcee Trinity
- [Trinity Mini on Together AI](https://www.together.ai/models/trinity-mini)
- [Trinity Manifesto](https://www.arcee.ai/blog/the-trinity-manifesto)

### FSDP2 & MoE Training
- [Training MoEs at Scale with PyTorch](https://pytorch.org/blog/training-moes/)
- [FSDP Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
- [MoE with FSDP Discussion](https://discuss.pytorch.org/t/how-to-train-mixture-of-experts-moe-model-with-fully-sharded-data-parallel-fsdp/192073)

### Expert Parallelism
- [NVIDIA MoE Training Blog](https://developer.nvidia.com/blog/accelerating-large-scale-mixture-of-experts-training-in-pytorch/)
- [DeepSpeed MoE](https://www.deepspeed.ai/tutorials/mixture-of-experts/)

## Conclusion

The current DisTrainer is **ready for dense models** (Qwen, Llama, etc.) but needs:

1. **Code refactoring** to generalize the FSDP module (rename qwen3 → parallelization)
2. **MoE-specific FSDP strategy** with expert freezing
3. **Configuration updates** to specify model type and EP dimensions

With these changes, DisTrainer will support:
- ✅ Dense transformers (Qwen3, Llama, etc.)
- ✅ MoE models with frozen experts (Arcee Trinity Mini)
- 🔮 Full MoE training (when vLLM supports expert adapters)
