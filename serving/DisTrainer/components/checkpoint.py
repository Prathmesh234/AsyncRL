"""
CheckpointManager using Distributed Checkpointing (DCP).
Based on TorchTitan's checkpoint management pattern.
"""

import shutil
import torch
import torch.distributed.checkpoint as dcp
from pathlib import Path
from typing import Dict, Any, Optional

from ..mesh import is_main_rank


class CheckpointManager:
    """Manages distributed checkpoints using PyTorch DCP."""
    
    def __init__(
        self,
        checkpoint_dir: str,
        keep_latest_k: int = 3
    ):
        """
        Initialize CheckpointManager.
        
        Args:
            checkpoint_dir: Directory to save checkpoints
            keep_latest_k: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_latest_k = keep_latest_k
    
    def save(self, state_dict: Dict[str, Any], step: int) -> str:
        """
        Save a sharded checkpoint using DCP.
        Each rank saves its own shard.
        
        Args:
            state_dict: Dictionary containing model and optimizer state
            step: Current training step
        
        Returns:
            Path to the saved checkpoint
        """
        checkpoint_path = self.checkpoint_dir / f"step_{step:06d}"
        checkpoint_path.mkdir(exist_ok=True)
        
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path)
        )
        
        # Cleanup old checkpoints (only on main rank)
        if is_main_rank():
            self._cleanup_old_checkpoints()
        
        return str(checkpoint_path)
    
    def load(self, state_dict: Dict[str, Any], step: Optional[int] = None) -> int:
        """
        Load a checkpoint from step or latest.
        
        Args:
            state_dict: Dictionary to load state into (model and optimizer)
            step: Specific step to load, or None for latest
        
        Returns:
            The step number that was loaded
        """
        if step is None:
            step = self._get_latest_step()
        
        if step is None:
            raise ValueError("No checkpoints found")
        
        checkpoint_path = self.checkpoint_dir / f"step_{step:06d}"
        
        if not checkpoint_path.exists():
            raise ValueError(f"Checkpoint at step {step} not found")
        
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path)
        )
        
        return step
    
    def _get_latest_step(self) -> Optional[int]:
        """Get the latest checkpoint step number."""
        checkpoints = sorted(self.checkpoint_dir.glob("step_*"))
        if not checkpoints:
            return None
        
        # Extract step number from directory name
        latest = checkpoints[-1].name
        return int(latest.replace("step_", ""))
    
    def _cleanup_old_checkpoints(self):
        """Keep only the latest K checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("step_*"))
        
        if len(checkpoints) > self.keep_latest_k:
            for ckpt in checkpoints[:-self.keep_latest_k]:
                shutil.rmtree(ckpt)
    
    def list_checkpoints(self) -> list:
        """List all available checkpoints."""
        checkpoints = sorted(self.checkpoint_dir.glob("step_*"))
        return [int(ckpt.name.replace("step_", "")) for ckpt in checkpoints]
    
    def has_checkpoint(self) -> bool:
        """Check if any checkpoint exists."""
        return len(list(self.checkpoint_dir.glob("step_*"))) > 0
