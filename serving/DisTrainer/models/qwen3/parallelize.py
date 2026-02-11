"""
DEPRECATED: This module has been moved to models/parallelization/dense_model.py

This file is kept for backwards compatibility only.
Please use the new parallelization module structure:
- models.parallelization.dense_model - For standard transformers (Qwen, Llama, etc.)
- models.parallelization.moe_model - For MoE models (Arcee Trinity, DeepSeek, etc.)
"""

import warnings
from ..parallelization.dense_model import apply_fsdp_dense as apply_fsdp

warnings.warn(
    "The 'models.qwen3.parallelize' module is deprecated. "
    "Please use 'models.parallelization.dense_model' instead. "
    "This compatibility wrapper will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["apply_fsdp"]
