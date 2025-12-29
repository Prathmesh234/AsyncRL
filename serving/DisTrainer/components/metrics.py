"""
Metrics logging for training.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..mesh import is_main_rank
import os

try:
    import wandb
except ImportError:
    wandb = None


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    avg_reward: float = 0.0
    batches_processed: int = 0
    tokens_per_second: float = 0.0
    avg_prompt_length: float = 0.0
    avg_completion_length: float = 0.0
    max_completion_length: int = 0
    timestamp: float = field(default_factory=time.time)


class MetricsLogger:
    """Simple metrics logger for training."""
    
    def __init__(self):
        self.history: list = []
        self.start_time: Optional[float] = None
    
    def start(self):
        """Start timing and initialize W&B if enabled."""
        self.start_time = time.time()
        
        # Initialize W&B on main rank if enabled via env var or if not explicitly disabled
        if is_main_rank() and wandb is not None:
            # Check env vars for enablement, default to True if wandb is installed
            disabled = os.getenv("WANDB_DISABLED", "false").lower() in ("true", "1", "yes")
            if not disabled and wandb.run is None:
                project = os.getenv("WANDB_PROJECT", "asyncrl-distrainer")
                run_name = os.getenv("WANDB_RUN_NAME", None)
                entity = os.getenv("WANDB_ENTITY", None)
                
                wandb.init(
                    project=project,
                    name=run_name,
                    entity=entity,
                    settings=wandb.Settings(start_method="fork") # generic safety
                )
                print(f"🚀 W&B Initialized: {project} / {wandb.run.name}")

    def log(self, metrics: TrainingMetrics):
        """Log metrics (only on main rank)."""
        if is_main_rank():
            self.history.append(metrics)
            self._print_metrics(metrics)
            
            # Log to W&B
            if wandb is not None and wandb.run is not None:
                wandb.log({
                    "train/loss": metrics.loss,
                    "train/learning_rate": metrics.learning_rate,
                    "train/avg_reward": metrics.avg_reward,
                    "train/tokens_per_second": metrics.tokens_per_second,
                    "train/avg_prompt_length": metrics.avg_prompt_length,
                    "train/avg_completion_length": metrics.avg_completion_length,
                    "train/max_completion_length": metrics.max_completion_length,
                    "train/batches_processed": metrics.batches_processed,
                    "train/step": metrics.step
                }, step=metrics.step)
    
    def _print_metrics(self, metrics: TrainingMetrics):
        """Print metrics to console."""
        elapsed = time.time() - (self.start_time or time.time())
        print(
            f"[Step {metrics.step:06d}] "
            f"Loss: {metrics.loss:.4f} | "
            f"Avg Reward: {metrics.avg_reward:.4f} | "
            f"Len: {int(metrics.avg_prompt_length)}/{int(metrics.avg_completion_length)} | "
            f"LR: {metrics.learning_rate:.2e} | "
            f"Time: {elapsed:.1f}s"
        )
    
    def get_summary(self) -> Dict[str, Any]:
        """Get summary of training metrics."""
        if not self.history:
            return {}
        
        losses = [m.loss for m in self.history]
        rewards = [m.avg_reward for m in self.history]
        
        return {
            "total_steps": len(self.history),
            "avg_loss": sum(losses) / len(losses),
            "avg_reward": sum(rewards) / len(rewards),
            "final_loss": losses[-1],
            "final_reward": rewards[-1],
        }
