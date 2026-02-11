# Changelog - DisTrainer

## [Unreleased] - MoE Compatibility Update

### Added
- **MoE Model Support**: Added support for Mixture-of-Experts models (Arcee Trinity, DeepSeek-V2/V3, Mixtral, etc.)
  - New `models/parallelization/` module with separate strategies for dense and MoE models
  - `models/parallelization/dense_model.py` - Generalized FSDP for standard transformers
  - `models/parallelization/moe_model.py` - FSDP with expert freezing for MoE models
  - `models/parallelization/freezing.py` - Utilities for selective layer freezing

- **Model Configuration Options**:
  - `model_type`: Specify "dense" or "moe" architecture type
  - `freeze_experts`: For MoE models, freeze expert/MLP layers during training

- **Documentation**:
  - `MOE_COMPATIBILITY.md` - Comprehensive guide for MoE model support
  - `config/train_config_moe.toml` - Example configuration for Arcee Trinity Mini
  - `CHANGELOG.md` - This file

### Changed
- **Refactored FSDP Module**: Renamed `models/qwen3/` to `models/parallelization/`
  - More generic and supports multiple model architectures
  - Backwards compatibility maintained via deprecation wrappers

- **Enhanced Trainer**:
  - `trainer.py` now supports both dense and MoE model types
  - Automatic FSDP strategy selection based on `model_type` config
  - Improved parameter statistics logging for MoE models

### Deprecated
- `models/qwen3/` module - Use `models/parallelization/` instead
  - Compatibility wrappers added with deprecation warnings
  - Will be removed in a future version

### Technical Details

#### MoE Architecture Support
The trainer now supports MoE models with the following features:
1. **Selective Layer Freezing**: Freeze expert/MLP layers when generator doesn't support LoRA on experts
2. **Automatic MoE Detection**: Identifies MoE layers by checking for expert-related attributes
3. **FSDP Strategy Selection**: Applies appropriate sharding strategy based on model type

#### Backwards Compatibility
All existing configurations and code using `models.qwen3` will continue to work with deprecation warnings. To migrate:
```python
# Old (deprecated)
from .models.qwen3 import apply_fsdp

# New (recommended)
from .models.parallelization import get_fsdp_strategy
fsdp_strategy = get_fsdp_strategy("dense")  # or "moe"
```

### Future Enhancements
- Expert Parallelism (EP) for distributed expert computation
- 3D device mesh support: `[EP dimension, DP dimension, replicate dimension]`
- Hybrid Sharded Data Parallel (HSDP) for large MoE models
- Training unfrozen experts when vLLM/sglang add LoRA support

### Testing
- Tested with Qwen3-4B (dense model) - ✅ No regression
- MoE implementation ready for Arcee Trinity Mini testing
- Freeze functionality validated with mock MoE layers

### References
- [Arcee Trinity Mini](https://www.arcee.ai/trinity)
- [Training MoEs with PyTorch](https://pytorch.org/blog/training-moes/)
- [FSDP2 Tutorial](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)
