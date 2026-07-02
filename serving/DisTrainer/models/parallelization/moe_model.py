"""
FSDP2 parallelization for Mixture-of-Experts (MoE) models.

Supports MoE architectures like:
- Qwen3-MoE (e.g. Qwen3-30B-A3B)
- Arcee Trinity (Mini, Large)
- DeepSeek-V2, DeepSeek-V3
- Mixtral
- Qwen2-MoE

Key features:
- Optional expert layer freezing (attention-only LoRA for generators pinned
  to vLLM < 0.15, which cannot serve expert-layer adapters)
- Expert Parallelism (EP) support (future enhancement)
- Handles both routed experts and shared experts
"""

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy
from typing import Optional

from .freezing import freeze_moe_experts


def is_moe_layer(layer: nn.Module) -> bool:
    """
    Detect if a layer is an MoE layer by checking for expert-related attributes.

    Common patterns:
    - DeepSeekMoE: has 'experts', 'gate', 'shared_expert'
    - Mixtral: has 'block_sparse_moe'
    - Arcee Trinity: has 'experts', 'router'

    Args:
        layer: A transformer layer module

    Returns:
        True if the layer contains MoE expert modules
    """
    # Check for common MoE attribute names
    for attr_name in ['experts', 'moe', 'block_sparse_moe', 'router', 'gate']:
        if hasattr(layer, attr_name):
            return True

    # Check child modules for expert-related names
    for name, _ in layer.named_modules():
        if any(keyword in name.lower() for keyword in ['expert', 'moe', 'router', 'gate']):
            return True

    return False


def apply_fsdp_moe(
    model: nn.Module,
    mesh,
    freeze_experts: bool = True,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32,
    expert_parallel_mesh: Optional[str] = None
) -> nn.Module:
    """
    Apply FSDP2 with optional expert freezing for MoE models.

    freeze_experts=True freezes expert/MLP layers BEFORE applying FSDP
    (attention-only training). For LoRA models this freezes the expert-side
    LoRA params — the base weights are already frozen by PEFT. Use this when
    the generator's vLLM is < 0.15 and cannot serve expert-layer adapters.

    freeze_experts=False leaves expert LoRA params trainable. Serving the
    resulting adapter requires vLLM >= 0.15 (fused_moe_lora kernel; Qwen3-MoE,
    GPT-OSS, DeepSeek families).

    Args:
        model: MoE model (e.g., Qwen3-30B-A3B)
        mesh: DeviceMesh for distributed training
        freeze_experts: If True, freeze all expert/MLP layers (default: True)
        param_dtype: Data type for parameters (bfloat16 for efficiency)
        reduce_dtype: Data type for gradient reduction (float32 for stability)
        expert_parallel_mesh: Future: Sub-mesh for Expert Parallelism (EP)

    Returns:
        The FSDP2-wrapped model with frozen experts (if requested)

    Note:
        Expert Parallelism (EP) is not yet implemented. Currently, experts
        are either frozen or trained with standard FSDP data parallelism.
        For true EP support, we need:
        1. 3D device mesh: [EP dimension, DP dimension, replicate dimension]
        2. Separate FSDP wrapping for expert vs non-expert layers
        3. Custom process groups for expert communication
    """
    # Step 1: Freeze expert layers if requested
    # This MUST happen before FSDP wrapping to avoid issues with param sharding
    if freeze_experts:
        freeze_moe_experts(model)

    # Step 2: Apply FSDP with mixed precision policy
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype
    )

    # Find transformer layers (same as dense model)
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        embed_tokens = getattr(model.model, 'embed_tokens', None)
    elif hasattr(model, 'layers'):
        layers = model.layers
        embed_tokens = getattr(model, 'embed_tokens', None)
    else:
        raise ValueError(
            "Could not find transformer layers. Model must have 'model.layers' or 'layers' attribute."
        )

    # Step 3: Shard each transformer layer
    # MoE and dense layers both use standard FSDP data parallelism; with
    # freeze_experts=False the expert LoRA params simply stay trainable.
    # TODO: Expert Parallelism (EP) as a future optimization for large expert counts
    expert_lora_training = False
    for i, layer in enumerate(layers):
        if is_moe_layer(layer) and not freeze_experts:
            expert_lora_training = True
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)

    if expert_lora_training:
        print(
            "MoE: freeze_experts=False — expert LoRA params are trainable. "
            "Serving these adapters requires vLLM >= 0.15 (fused_moe_lora) "
            "on the DisGenerator prefill/decode servers."
        )

    # Step 4: Shard embedding layer
    if embed_tokens is not None:
        fully_shard(embed_tokens, mesh=mesh, mp_policy=mp_policy)

    # Step 5: Shard LM head
    if hasattr(model, 'lm_head'):
        fully_shard(model.lm_head, mesh=mesh, mp_policy=mp_policy)

    # Step 6: Final outer wrap
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    return model
