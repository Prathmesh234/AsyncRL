"""
FSDP2 parallelization for Qwen3 models.
Based on TorchTitan's llama/infra/parallelize.py pattern.
"""

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


def apply_fsdp(
    model: nn.Module,
    mesh,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32
) -> nn.Module:
    """
    Apply FSDP2 sharding to Qwen3 model.
    
    Shards each TransformerBlock individually for better memory efficiency.
    
    Args:
        model: The Qwen3 model to shard
        mesh: DeviceMesh for distributed training
        param_dtype: Data type for parameters (bfloat16 for efficiency)
        reduce_dtype: Data type for gradient reduction (float32 for stability)
    
    Returns:
        The FSDP2-wrapped model
    """
    mp_policy = MixedPrecisionPolicy(
        param_dtype=param_dtype,
        reduce_dtype=reduce_dtype
    )
    
    # Shard each transformer layer individually
    # Qwen3 uses model.model.layers for transformer blocks
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        for layer in model.model.layers:
            fully_shard(layer, mesh=mesh, mp_policy=mp_policy)
        
        # Shard embedding layer
        if hasattr(model.model, 'embed_tokens'):
            fully_shard(model.model.embed_tokens, mesh=mesh, mp_policy=mp_policy)
        
        # Shard LM head
        if hasattr(model, 'lm_head'):
            fully_shard(model.lm_head, mesh=mesh, mp_policy=mp_policy)
    
    # Final outer wrap
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)
    
    return model
