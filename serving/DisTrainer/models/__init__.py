# Models package

# Backwards compatibility: Import from parallelization module
from .parallelization import (
    apply_fsdp_dense,
    apply_fsdp_moe,
    get_fsdp_strategy,
    freeze_moe_experts,
    print_trainable_parameters,
)

# Legacy alias for backwards compatibility
from .parallelization.dense_model import apply_fsdp_dense as apply_fsdp

__all__ = [
    "apply_fsdp",  # Legacy
    "apply_fsdp_dense",
    "apply_fsdp_moe",
    "get_fsdp_strategy",
    "freeze_moe_experts",
    "print_trainable_parameters",
]
