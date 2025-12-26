"""
Main Trainer class for DisTrainer.
Orchestrates distributed GRPO training with FSDP2.
"""

import torch
import tomli
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer

from .mesh import ParallelDims, build_mesh, init_distributed, is_main_rank
from .models.qwen3 import apply_fsdp
from .components import CheckpointManager, DataLoader, compute_grpo_loss, MetricsLogger
from .components.metrics import TrainingMetrics


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-4B-Thinking-2507"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    target_modules: list = None
    adapter_path: Optional[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            self.target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]


@dataclass
class TrainingConfig:
    learning_rate: float = 5e-6
    beta: float = 0.01
    max_grad_norm: float = 1.0
    num_generations_per_prompt: int = 4


@dataclass
class DataConfig:
    generations_dir: str = "./data/generations"
    max_sequence_length: int = 8192


@dataclass
class CheckpointConfig:
    save_dir: str = "./checkpoints"
    save_interval: int = 10
    keep_latest_k: int = 3


@dataclass
class Config:
    model: ModelConfig
    training: TrainingConfig
    parallel_dims: ParallelDims
    data: DataConfig
    checkpoint: CheckpointConfig
    
    @classmethod
    def from_toml(cls, path: str) -> "Config":
        """Load configuration from TOML file."""
        with open(path, "rb") as f:
            data = tomli.load(f)
        
        return cls(
            model=ModelConfig(**data.get("model", {})),
            training=TrainingConfig(**data.get("training", {})),
            parallel_dims=ParallelDims(**data.get("parallel_dims", {})),
            data=DataConfig(**data.get("data", {})),
            checkpoint=CheckpointConfig(**data.get("checkpoint", {})),
        )


class Trainer:
    """
    Main training orchestrator for DisTrainer.
    Handles FSDP2 setup, training loop, and checkpointing.
    """
    
    def __init__(self, config: Config):
        """
        Initialize the Trainer.
        
        Args:
            config: Training configuration
        """
        self.config = config
        
        # Initialize distributed
        self.local_rank = init_distributed()
        
        # Build device mesh
        self.mesh = build_mesh(config.parallel_dims)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._build_model()
        
        # Apply FSDP2
        self.model = apply_fsdp(self.model, self.mesh)
        
        # Setup optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.training.learning_rate
        )
        
        # Setup checkpoint manager
        self.checkpoint_manager = CheckpointManager(
            checkpoint_dir=config.checkpoint.save_dir,
            keep_latest_k=config.checkpoint.keep_latest_k
        )
        
        # Setup data loader
        self.data_loader = DataLoader(config.data.generations_dir)
        
        # Setup metrics logger
        self.metrics_logger = MetricsLogger()
        
        # Training state
        self.step = 0
        
        # Try to resume from checkpoint
        self._try_resume()
        
        if is_main_rank():
            print(f"Trainer initialized on {config.parallel_dims.dp} GPUs")
            print(f"Model: {config.model.name}")
            print(f"Starting from step: {self.step}")
    
    def _build_model(self):
        """Load the model and tokenizer."""
        if is_main_rank():
            print(f"Loading model: {self.config.model.name}")
        
        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(
            self.config.model.name,
            trust_remote_code=True
        )
        
        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            self.config.model.name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        
        # Apply LoRA if configured
        # Apply LoRA if configured
        if self.config.model.use_lora:
            from peft import get_peft_model, LoraConfig, PeftModel
            
            if self.config.model.adapter_path:
                if is_main_rank():
                    print(f"Loading LoRA adapter from: {self.config.model.adapter_path}")
                # Load existing adapter and ensure it's trainable
                model = PeftModel.from_pretrained(
                    model, 
                    self.config.model.adapter_path,
                    is_trainable=True
                )
            else:
                lora_config = LoraConfig(
                    r=self.config.model.lora_r,
                    lora_alpha=self.config.model.lora_alpha,
                    target_modules=self.config.model.target_modules,
                    lora_dropout=0.05,
                    bias="none",
                    task_type="CAUSAL_LM"
                )
                model = get_peft_model(model, lora_config)
            
            if is_main_rank():
                print("LoRA applied to model")
                model.print_trainable_parameters()
        
        # Ensure entire model is in bfloat16 to match FSDP expectation
        model = model.to(torch.bfloat16)
        
        # Enable gradient checkpointing to save memory
        model.gradient_checkpointing_enable()
        
        return model, tokenizer
    
    def _try_resume(self):
        """Try to resume from latest checkpoint."""
        if self.checkpoint_manager.has_checkpoint():
            try:
                state_dict = {
                    "model": self.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "step": self.step
                }
                loaded_step = self.checkpoint_manager.load(state_dict)
                self.step = state_dict.get("step", loaded_step)
                
                if is_main_rank():
                    print(f"Resumed from checkpoint at step {self.step}")
            except Exception as e:
                if is_main_rank():
                    print(f"Failed to resume from checkpoint: {e}")
    
    def train_step(self) -> Dict[str, Any]:
        """
        Execute one training step.
        
        Returns:
            Dictionary with training metrics
        """
        # Load next batch
        batch = self.data_loader.get_next_batch()
        if batch is None:
            return {"status": "no_data", "step": self.step}
        
        self.model.train()
        
        # Compute GRPO loss
        loss = compute_grpo_loss(
            self.model,
            batch,
            beta=self.config.training.beta
        )
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        self.step += 1
        
        # Compute average reward from batch
        all_rewards = []
        for group in batch:
            for completion in group.get("completions", []):
                all_rewards.append(completion.get("reward", 0.0))
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        
        # Log metrics
        metrics = TrainingMetrics(
            step=self.step,
            loss=loss.item(),
            learning_rate=self.config.training.learning_rate,
            avg_reward=avg_reward,
            batches_processed=len(batch)
        )
        self.metrics_logger.log(metrics)
        
        # Checkpoint if needed
        if self.step % self.config.checkpoint.save_interval == 0:
            self.save_checkpoint()
        
        return {
            "status": "success",
            "step": self.step,
            "loss": loss.item(),
            "avg_reward": avg_reward,
            "batches_processed": len(batch)
        }
    
    def save_checkpoint(self) -> str:
        """Save a checkpoint."""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step
        }
        path = self.checkpoint_manager.save(state_dict, self.step)
        
        if is_main_rank():
            print(f"Saved checkpoint at step {self.step}")
        
        return path
    
    def load_checkpoint(self, step: Optional[int] = None) -> int:
        """Load a checkpoint."""
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step
        }
        loaded_step = self.checkpoint_manager.load(state_dict, step)
        self.step = state_dict.get("step", loaded_step)
        
        if is_main_rank():
            print(f"Loaded checkpoint from step {self.step}")
        
        return self.step
    
    def get_status(self) -> Dict[str, Any]:
        """Get current trainer status."""
        return {
            "step": self.step,
            "batches_available": self.data_loader.count_available(),
            "batches_processed": self.data_loader.count_processed(),
            "checkpoints": self.checkpoint_manager.list_checkpoints(),
            "model": self.config.model.name,
            "gpu_count": self.config.parallel_dims.dp
        }
