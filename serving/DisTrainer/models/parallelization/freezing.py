"""
Layer freezing utilities for selective training.

Used primarily for MoE models where expert/MLP layers need to be frozen
because the inference generator (vLLM/sglang) doesn't support LoRA adapters
on expert layers yet.
"""

import torch.nn as nn
from typing import List, Optional


def freeze_moe_experts(
    model: nn.Module,
    patterns: Optional[List[str]] = None,
    verbose: bool = True
) -> None:
    """
    Freeze all expert/MLP layers in an MoE model.

    This sets requires_grad=False for all parameters matching expert patterns.
    Used when the inference generator (vLLM/sglang) doesn't support LoRA
    adapters on expert layers.

    Args:
        model: The MoE model
        patterns: List of substrings to match in parameter names for freezing.
                  Default: ['expert', 'mlp', 'gate', 'moe_layer', 'router',
                            'shared_expert', 'routed_expert']
        verbose: If True, print freezing statistics

    Note:
        This should be called BEFORE applying FSDP wrapping to avoid
        issues with parameter sharding.
    """
    if patterns is None:
        patterns = [
            'expert',
            'mlp',
            'gate',
            'moe_layer',
            'router',
            'shared_expert',
            'routed_expert',
            'block_sparse_moe',
        ]

    frozen_count = 0
    total_count = 0
    frozen_params = 0
    total_params = 0

    for name, param in model.named_parameters():
        total_count += 1
        total_params += param.numel()

        # Check if parameter name matches any expert pattern
        should_freeze = any(pattern in name.lower() for pattern in patterns)

        if should_freeze:
            param.requires_grad = False
            frozen_count += 1
            frozen_params += param.numel()

    trainable_params = total_params - frozen_params

    if verbose:
        print(f"\n{'='*70}")
        print(f"Expert Layer Freezing Summary")
        print(f"{'='*70}")
        print(f"Frozen parameters:    {frozen_count:,} / {total_count:,} "
              f"({100 * frozen_count / max(total_count, 1):.1f}%)")
        print(f"Frozen param count:   {frozen_params:,} / {total_params:,} "
              f"({100 * frozen_params / max(total_params, 1):.1f}%)")
        print(f"Trainable param count: {trainable_params:,} "
              f"({100 * trainable_params / max(total_params, 1):.1f}%)")
        print(f"{'='*70}\n")


def print_trainable_parameters(model: nn.Module) -> None:
    """
    Print trainable parameter statistics for a model.

    Useful for verifying that freezing worked correctly.

    Args:
        model: The model to analyze
    """
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = total_params - trainable_params

    print(f"\n{'='*70}")
    print(f"Model Parameter Summary")
    print(f"{'='*70}")
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} "
          f"({100 * trainable_params / max(total_params, 1):.2f}%)")
    print(f"Frozen parameters:    {frozen_params:,} "
          f"({100 * frozen_params / max(total_params, 1):.2f}%)")
    print(f"{'='*70}\n")


def freeze_layers_by_name(
    model: nn.Module,
    layer_names: List[str],
    verbose: bool = True
) -> None:
    """
    Freeze specific layers by exact name match.

    Args:
        model: The model
        layer_names: List of exact layer names to freeze
        verbose: If True, print what was frozen
    """
    frozen = []
    for name, param in model.named_parameters():
        if name in layer_names:
            param.requires_grad = False
            frozen.append(name)

    if verbose:
        print(f"Frozen {len(frozen)} layers:")
        for name in frozen:
            print(f"  - {name}")


def unfreeze_all(model: nn.Module) -> None:
    """
    Unfreeze all parameters in the model.

    Args:
        model: The model to unfreeze
    """
    for param in model.parameters():
        param.requires_grad = True
