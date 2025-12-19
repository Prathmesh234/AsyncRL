"""
Metrics logging for training.
"""

import time
from typing import Dict, Any, Optional
from dataclasses import dataclass, field

from ..mesh import is_main_rank


@dataclass
class TrainingMetrics:
    """Container for training metrics."""
    step: int = 0
    loss: float = 0.0
    learning_rate: float = 0.0
    avg_reward: float = 0.0
    batches_processed: int = 0
    tokens_per_second: float = 0.0
    timestamp: float = field(default_factory=time.time)


class MetricsLogger:
    """Simple metrics logger for training."""
    
    def __init__(self):
        self.history: list = []
        self.start_time: Optional[float] = None
    
    def start(self):
        """Start timing."""
        self.start_time = time.time()
    
    def log(self, metrics: TrainingMetrics):
        """Log metrics (only on main rank)."""
        if is_main_rank():
            self.history.append(metrics)
            self._print_metrics(metrics)
    
    def _print_metrics(self, metrics: TrainingMetrics):
        """Print metrics to console."""
        elapsed = time.time() - (self.start_time or time.time())
        print(
            f"[Step {metrics.step:06d}] "
            f"Loss: {metrics.loss:.4f} | "
            f"Avg Reward: {metrics.avg_reward:.4f} | "
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
