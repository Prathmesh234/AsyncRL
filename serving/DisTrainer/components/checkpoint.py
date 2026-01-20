"""
CheckpointManager using Distributed Checkpointing (DCP).
Based on TorchTitan's checkpoint management pattern.

Policy naming convention: policy-{N}-{YYYYMMDD_HHMMSS}
- policy-0-initial: Starting checkpoint (from ToolGRPOTrainer)
- policy-1-20251226_131234: First trained policy
- policy-2-20251226_143456: Second trained policy
"""

import shutil
import torch
import torch.distributed.checkpoint as dcp
from pathlib import Path
from typing import Dict, Any, Optional
from datetime import datetime
import re

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
            checkpoint_dir: Directory to save checkpoints (e.g., DisTrainer/models)
            keep_latest_k: Number of recent checkpoints to keep
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_latest_k = keep_latest_k
    
    def _get_next_policy_version(self) -> int:
        """Get the next policy version number by scanning existing policies."""
        max_version = -1
        pattern = re.compile(r"policy-(\d+)")
        
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                match = pattern.match(path.name)
                if match:
                    version = int(match.group(1))
                    max_version = max(max_version, version)
        
        return max_version + 1
    
    def save(self, state_dict: Dict[str, Any], step: int) -> str:
        """
        Save a sharded checkpoint using DCP.
        Each rank saves its own shard.
        
        Names checkpoints as: policy-{N}-{YYYYMMDD_HHMMSS}
        
        Args:
            state_dict: Dictionary containing model and optimizer state
            step: Current training step (included in metadata)
        
        Returns:
            Path to the saved checkpoint
        """
        # Get next policy version
        policy_version = self._get_next_policy_version()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        checkpoint_name = f"policy-{policy_version}-{timestamp}"
        checkpoint_path = self.checkpoint_dir / checkpoint_name
        checkpoint_path.mkdir(exist_ok=True)
        
        # Add step to state_dict for reference
        state_dict["step"] = step
        state_dict["policy_version"] = policy_version
        
        dcp.save(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path)
        )
        
        # Cleanup and update symlink (only on main rank)
        if is_main_rank():
            self._cleanup_old_checkpoints()
            self._update_latest_symlink(checkpoint_path)
            print(f"💾 Saved checkpoint: {checkpoint_name} (step {step})")
        
        return str(checkpoint_path)

    
    def load(self, state_dict: Dict[str, Any], step: Optional[int] = None) -> int:
        """
        Load a checkpoint (latest policy or specific policy version).
        
        Args:
            state_dict: Dictionary to load state into (model and optimizer)
            step: Specific policy version to load, or None for latest
        
        Returns:
            The policy version that was loaded
        """
        if step is None:
            step = self._get_latest_policy_version()
        
        if step is None:
            raise ValueError("No policy checkpoints found")
        
        # Find the checkpoint directory for this policy version
        checkpoint_path = self._find_policy_path(step)
        
        if checkpoint_path is None or not checkpoint_path.exists():
            raise ValueError(f"Policy version {step} not found")
        
        dcp.load(
            state_dict=state_dict,
            checkpoint_id=str(checkpoint_path)
        )
        
        if is_main_rank():
            print(f"📂 Loaded checkpoint: {checkpoint_path.name}")
        
        return step
    
    def _is_valid_dcp_checkpoint(self, path: Path) -> bool:
        """Check if a directory is a valid DCP checkpoint (has .metadata file)."""
        return (path / ".metadata").exists()
    
    def _get_latest_policy_version(self) -> Optional[int]:
        """Get the latest policy version number (only considers valid DCP checkpoints)."""
        max_version = None
        pattern = re.compile(r"policy-(\d+)")
        
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                match = pattern.match(path.name)
                # Only consider directories with .metadata (valid DCP checkpoints)
                if match and self._is_valid_dcp_checkpoint(path):
                    version = int(match.group(1))
                    if max_version is None or version > max_version:
                        max_version = version
        
        return max_version
    
    def _find_policy_path(self, version: int) -> Optional[Path]:
        """Find the checkpoint path for a specific policy version."""
        pattern = re.compile(rf"policy-{version}-")
        
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and pattern.match(path.name):
                return path
        
        return None
    
    def _cleanup_old_checkpoints(self):
        """Keep only the latest K policy checkpoints."""
        # Collect all policy directories with their versions
        policies = []
        pattern = re.compile(r"policy-(\d+)")
        
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                match = pattern.match(path.name)
                if match:
                    version = int(match.group(1))
                    policies.append((version, path))
        
        # Sort by version
        policies.sort(key=lambda x: x[0])
        
        # Remove old ones (keep latest K)
        if len(policies) > self.keep_latest_k:
            for version, ckpt_path in policies[:-self.keep_latest_k]:
                if is_main_rank():
                    print(f"🗑️ Removing old checkpoint: {ckpt_path.name}")
                try:
                    shutil.rmtree(ckpt_path)
                except OSError as e:
                    print(f"⚠️ Failed to remove old checkpoint {ckpt_path.name}: {e}")
    
    def list_checkpoints(self) -> list:
        """List all available policy versions."""
        policies = []
        pattern = re.compile(r"policy-(\d+)")
        
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir():
                match = pattern.match(path.name)
                if match:
                    policies.append(int(match.group(1)))
        
        return sorted(policies)
    
    def has_checkpoint(self) -> bool:
        """Check if any valid DCP policy checkpoint exists."""
        pattern = re.compile(r"policy-\d+")
        for path in self.checkpoint_dir.iterdir():
            if path.is_dir() and pattern.match(path.name):
                # Only count directories with .metadata as valid checkpoints
                if self._is_valid_dcp_checkpoint(path):
                    return True
        return False

    def _update_latest_symlink(self, latest_path: Path):
        """Update a symlink named 'latest_adapter' to the latest checkpoint."""
        symlink_path = self.checkpoint_dir / "latest_adapter"
        try:
            if symlink_path.is_symlink() or symlink_path.exists():
                symlink_path.unlink()

            # Create symlink to the latest directory name
            # Using relative path (target.name) makes the symlink portable
            symlink_path.symlink_to(latest_path.name, target_is_directory=True)

            # Create .policy_ready signal file to notify DisGenerator
            # This enables hot-swap of LoRA policies without restarting vLLM
            self._create_policy_ready_signal()
        except Exception as e:
            # Don't fail training if symlink fails
            import logging
            logging.getLogger(__name__).warning(f"Failed to update 'latest_adapter' symlink: {e}")

    def _create_policy_ready_signal(self):
        """
        Create a .policy_ready signal file to notify DisGenerator.

        DisGenerator's PolicyManager watches for this file and triggers
        a hot-swap of the LoRA adapter when detected.
        """
        signal_file = self.checkpoint_dir / ".policy_ready"
        try:
            # Write timestamp to signal file
            signal_file.write_text(f"{datetime.now().isoformat()}\n")
            import logging
            logging.getLogger(__name__).info("📢 Created .policy_ready signal for DisGenerator")
        except Exception as e:
            import logging
            logging.getLogger(__name__).warning(f"Failed to create .policy_ready signal: {e}")
