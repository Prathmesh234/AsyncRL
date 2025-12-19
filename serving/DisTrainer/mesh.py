"""
DeviceMesh initialization for distributed training.
Based on TorchTitan's ParallelDims pattern.
"""

import os
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from dataclasses import dataclass


@dataclass
class ParallelDims:
    """TorchTitan-style parallel dimension configuration."""
    dp: int = 2   # Data parallel
    tp: int = 1   # Tensor parallel
    pp: int = 1   # Pipeline parallel
    
    @property
    def world_size(self) -> int:
        return self.dp * self.tp * self.pp


def build_mesh(parallel_dims: ParallelDims):
    """
    Build a DeviceMesh for distributed training.
    
    For 2 GPUs with data-parallel only:
        ParallelDims(dp=2, tp=1, pp=1)
    Returns a 1D mesh: ["dp"]
    """
    return init_device_mesh(
        "cuda",
        (parallel_dims.dp,),
        mesh_dim_names=("dp",)
    )


def init_distributed(backend: str = "nccl"):
    """Initialize the distributed process group."""
    if not dist.is_initialized():
        dist.init_process_group(backend=backend)
    
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    
    return local_rank


def get_rank() -> int:
    """Get current process rank."""
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size() -> int:
    """Get total number of processes."""
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main_rank() -> bool:
    """Check if this is the main (rank 0) process."""
    return get_rank() == 0
