"""
Parallelization strategies for different model architectures.

Supports:
- Dense transformers (Qwen, Llama, etc.)
- Mixture-of-Experts models (Arcee Trinity, DeepSeek, etc.)
"""

from .dense_model import apply_fsdp_dense
from .moe_model import apply_fsdp_moe
from .freezing import freeze_moe_experts, print_trainable_parameters


def get_fsdp_strategy(model_type: str = "dense"):
    """
    Get the appropriate FSDP strategy for a model type.

    Args:
        model_type: Type of model ("dense" or "moe")

    Returns:
        Function to apply FSDP sharding
    """
    if model_type == "moe":
        return apply_fsdp_moe
    elif model_type == "dense":
        return apply_fsdp_dense
    else:
        raise ValueError(f"Unknown model_type: {model_type}. Must be 'dense' or 'moe'")


__all__ = [
    "apply_fsdp_dense",
    "apply_fsdp_moe",
    "freeze_moe_experts",
    "print_trainable_parameters",
    "get_fsdp_strategy",
]
