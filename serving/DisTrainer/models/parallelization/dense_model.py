"""
FSDP2 parallelization for dense transformer models.

Supports standard transformer architectures like:
- Qwen3, Qwen2
- Llama 2, Llama 3
- Mistral
- Phi
- Any standard decoder-only transformer
"""

import torch
import torch.nn as nn
from torch.distributed._composable.fsdp import fully_shard, MixedPrecisionPolicy


def apply_fsdp_dense(
    model: nn.Module,
    mesh,
    param_dtype=torch.bfloat16,
    reduce_dtype=torch.float32
) -> nn.Module:
    """
    Apply FSDP2 sharding to a dense transformer model.

    Shards each TransformerBlock individually for better memory efficiency.
    Works with standard decoder-only transformers.

    Args:
        model: The transformer model to shard
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
    # Most models use model.model.layers or model.layers for transformer blocks
    layers = None
    if hasattr(model, 'model') and hasattr(model.model, 'layers'):
        layers = model.model.layers
        embed_tokens = getattr(model.model, 'embed_tokens', None)
        model_wrapper = model.model
    elif hasattr(model, 'layers'):
        layers = model.layers
        embed_tokens = getattr(model, 'embed_tokens', None)
        model_wrapper = model
    else:
        raise ValueError(
            "Could not find transformer layers. Model must have 'model.layers' or 'layers' attribute."
        )

    # Shard each transformer layer
    for layer in layers:
        fully_shard(layer, mesh=mesh, mp_policy=mp_policy)

    # Shard embedding layer
    if embed_tokens is not None:
        fully_shard(embed_tokens, mesh=mesh, mp_policy=mp_policy)

    # Shard LM head
    if hasattr(model, 'lm_head'):
        fully_shard(model.lm_head, mesh=mesh, mp_policy=mp_policy)

    # Final outer wrap
    fully_shard(model, mesh=mesh, mp_policy=mp_policy)

    return model
