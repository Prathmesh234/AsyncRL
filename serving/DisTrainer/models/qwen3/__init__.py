# Backwards compatibility wrapper
# The qwen3 module has been moved to parallelization/dense_model.py
# This module is kept for backwards compatibility with existing code

import warnings
from ..parallelization.dense_model import apply_fsdp_dense as apply_fsdp

warnings.warn(
    "The 'models.qwen3' module is deprecated. "
    "Please use 'models.parallelization.dense_model' instead. "
    "This compatibility wrapper will be removed in a future version.",
    DeprecationWarning,
    stacklevel=2
)

__all__ = ["apply_fsdp"]
