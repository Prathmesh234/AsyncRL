"""
Main Trainer class for DisTrainer.
Orchestrates distributed GRPO training with FSDP2.
"""

import torch
import torch.distributed as dist
import tomli
from pathlib import Path
from typing import Dict, Any, Optional
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.distributed.checkpoint.state_dict import get_state_dict, StateDictOptions

from .mesh import ParallelDims, build_mesh, init_distributed, is_main_rank
from .models.parallelization import get_fsdp_strategy, print_trainable_parameters
from .components import CheckpointManager, DataLoader, compute_grpo_loss, MetricsLogger, BufferQueue
from .components.metrics import TrainingMetrics


@dataclass
class ModelConfig:
    name: str = "Qwen/Qwen3-4B-Thinking-2507"
    use_lora: bool = True
    lora_r: int = 8
    lora_alpha: int = 16
    target_modules: list = None
    adapter_path: Optional[str] = None
    model_type: str = "dense"  # "dense" or "moe"
    freeze_experts: bool = True  # For MoE: freeze expert/MLP layers

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
    # Queue mode: rank 0 holds an in-memory BufferQueue fed by the generator
    # via POST /push_group. Set false to fall back to file-based batches.
    use_buffer_queue: bool = True
    buffer_maxlen: int = 32


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

    def __init__(self, config: Config, use_base_model: bool = False, buffer_queue: Optional[BufferQueue] = None):
        """
        Initialize the Trainer.

        Args:
            config: Training configuration
            use_base_model: If True, use base model without LoRA adapter (creates fresh LoRA)
        """
        self.config = config
        self.use_base_model = use_base_model
        
        # Initialize distributed
        self.local_rank = init_distributed()
        
        # Build device mesh
        self.mesh = build_mesh(config.parallel_dims)
        
        # Load model and tokenizer
        self.model, self.tokenizer = self._build_model()

        # Apply FSDP2 (strategy depends on model_type)
        fsdp_strategy = get_fsdp_strategy(config.model.model_type)
        if config.model.model_type == "moe":
            self.model = fsdp_strategy(
                self.model,
                self.mesh,
                freeze_experts=config.model.freeze_experts
            )
        else:
            self.model = fsdp_strategy(self.model, self.mesh)
        
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
        
        # Setup data loader (queue mode when buffer_queue provided, else file mode)
        self.data_loader = DataLoader(
            config.data.generations_dir,
            buffer_queue=buffer_queue,
            batch_size=config.training.num_generations_per_prompt,
        )
        
        # Setup metrics logger
        self.metrics_logger = MetricsLogger()
        
        # Training state
        self.step = 0
        
        # Try to resume from checkpoint
        self._try_resume()
        
        if is_main_rank():
            print(f"Trainer initialized on {config.parallel_dims.dp} GPUs")
            print(f"Model: {config.model.name}")
            if self.use_base_model:
                print(f"Mode: BASE MODEL (no finetuned adapter)")
            else:
                print(f"Mode: FINETUNED (with LoRA adapter)")
            print(f"Starting from step: {self.step}")
            self.metrics_logger.start()
    
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
        if self.config.model.use_lora:
            from peft import get_peft_model, LoraConfig, PeftModel

            # If use_base_model is True, skip loading adapter and create fresh LoRA
            if self.config.model.adapter_path and not self.use_base_model:
                if is_main_rank():
                    print(f"Loading LoRA adapter from: {self.config.model.adapter_path}")
                # Load existing adapter and ensure it's trainable
                model = PeftModel.from_pretrained(
                    model,
                    self.config.model.adapter_path,
                    is_trainable=True
                )
            else:
                if is_main_rank():
                    if self.use_base_model:
                        print("Using BASE MODEL with fresh LoRA (--use-base-model flag)")
                    else:
                        print("No adapter path configured, creating fresh LoRA")
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
                # Additional statistics after potential expert freezing
                if self.config.model.model_type == "moe":
                    print(f"\nMoE Configuration:")
                    print(f"  - Model Type: MoE (Mixture-of-Experts)")
                    print(f"  - Freeze Experts: {self.config.model.freeze_experts}")
                    if self.config.model.freeze_experts:
                        print(f"  - Training: Attention-only LoRA (expert LoRA frozen)")
                    else:
                        print(f"  - Training: Attention + expert LoRA")
                        print(f"    Serving requires vLLM >= 0.15 (fused_moe_lora) on DisGenerator")
        
        # Convert to bfloat16 for memory efficiency
        model = model.to(torch.bfloat16)
        
        # CRITICAL: Disable use_cache to prevent gradient checkpointing auto-enable
        # This is necessary because gradient checkpointing conflicts with caching
        model.config.use_cache = False
        
        # Explicitly enable gradients on trainable parameters (LoRA layers)
        # This is required after dtype conversion
        for name, param in model.named_parameters():
            if param.requires_grad:
                param.requires_grad_(True)
        
        # Set model to training mode
        model.train()
        
        # Enable gradient checkpointing for memory efficiency
        # use_reentrant=False is REQUIRED for PEFT/LoRA models to avoid
        # "None of the inputs have requires_grad=True" errors.
        # This works because non-reentrant checkpointing doesn't require
        # input tensors to have requires_grad=True - it tracks gradients differently.
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False}
        )
        
        if is_main_rank():
            print("✅ Gradient checkpointing enabled (use_reentrant=False)")
        
        return model, tokenizer
    
    def _try_resume(self):
        """Try to resume from the latest DCP checkpoint."""
        if not self.checkpoint_manager.has_checkpoint():
            return
        try:
            state_dict = {
                "model": self.model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": 0,  # Will be overwritten by DCP load
            }
            self.checkpoint_manager.load(state_dict)
            # DCP restores the "step" key in-place; fall back to 0 if it
            # wasn't persisted (e.g. very old checkpoint format).
            self.step = state_dict.get("step", 0)

            if is_main_rank():
                print(f"Resumed from checkpoint at step {self.step}")
        except Exception as e:
            if is_main_rank():
                print(f"Failed to resume from checkpoint: {e}. Starting from step 0.")
    
    def train_step(self) -> Dict[str, Any]:
        """
        Execute one training step.
        
        Note: compute_grpo_loss now accumulates gradients internally via per-completion
        backward() calls for memory efficiency. We zero gradients BEFORE the loss call.
        
        Returns:
            Dictionary with training metrics
        """
        # Load next batch on rank 0 and broadcast to all ranks.
        # In queue mode only rank 0 receives data (the BufferQueue is fed by
        # its HTTP server), so other ranks must get the batch via broadcast.
        # All ranks reach this point in lockstep (distributed_worker broadcasts
        # the TRAIN command), so the collective is safe.
        batch = None
        if is_main_rank():
            batch = self.data_loader.get_next_batch()
        if dist.is_initialized() and dist.get_world_size() > 1:
            obj_list = [batch]
            dist.broadcast_object_list(obj_list, src=0)
            batch = obj_list[0]
        if batch is None:
            return {"status": "no_data", "step": self.step}
        
        self.model.train()
        
        # Zero gradients BEFORE loss computation 
        # (backward() is called inside compute_grpo_loss for memory efficiency)
        self.optimizer.zero_grad()
        
        # Compute GRPO loss (gradients are accumulated inside)
        loss = compute_grpo_loss(
            self.model,
            batch,
            beta=self.config.training.beta
        )
        
        # Note: backward() already called inside compute_grpo_loss
        # The returned loss is for logging purposes only
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(
            self.model.parameters(),
            self.config.training.max_grad_norm
        )
        
        # Optimizer step
        self.optimizer.step()
        
        self.step += 1
        
        # Compute metrics from batch
        all_rewards = []
        prompt_lengths = []
        completion_lengths = []
        
        for group in batch:
            p_len = len(group.get("prompt_ids", []))
            for completion in group.get("completions", []):
                all_rewards.append(completion.get("reward", 0.0))
                # completion_ids might be missing if failed, handle gracefully
                c_ids = completion.get("completion_ids", [])
                completion_lengths.append(len(c_ids))
                prompt_lengths.append(p_len)
                
        avg_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        avg_prompt_len = sum(prompt_lengths) / len(prompt_lengths) if prompt_lengths else 0.0
        avg_completion_len = sum(completion_lengths) / len(completion_lengths) if completion_lengths else 0.0
        max_completion_len = max(completion_lengths) if completion_lengths else 0
        
        # Log metrics
        metrics = TrainingMetrics(
            step=self.step,
            loss=loss.item(),
            learning_rate=self.config.training.learning_rate,
            avg_reward=avg_reward,
            avg_prompt_length=avg_prompt_len,
            avg_completion_length=avg_completion_len,
            max_completion_length=max_completion_len,
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
        """
        Save a checkpoint and expose the new policy to DisGenerator.

        Order of operations (critical for correctness):
          1. Save DCP checkpoint (training state for resumption)
          2. Save HF PEFT adapter (vLLM-compatible LoRA weights)
          3. Update latest_adapter symlink + .policy_ready signal
             → DisGenerator only sees a fully-written checkpoint
        """
        state_dict = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step
        }
        path = self.checkpoint_manager.save(state_dict, self.step)

        # ------------------------------------------------------------------
        # Save HF PEFT Adapter for vLLM hot-swap compatibility.
        # vLLM requires adapter_config.json + adapter_model.safetensors.
        # We must finish writing these BEFORE signalling DisGenerator.
        # ------------------------------------------------------------------
        adapter_saved = False
        if self.config.model.use_lora:
            try:
                from peft import get_peft_model_state_dict

                if is_main_rank():
                    print("Gathering LoRA weights for vLLM export...")

                # Gather full (un-sharded) state dict on CPU to save GPU mem
                options = StateDictOptions(full_state_dict=True, cpu_offload=True)
                full_state_dict = get_state_dict(self.model, options=options)

                if is_main_rank():
                    try:
                        adapter_state_dict = get_peft_model_state_dict(
                            self.model,
                            state_dict=full_state_dict
                        )
                        self.model.save_pretrained(path, state_dict=adapter_state_dict)
                        del adapter_state_dict
                        adapter_saved = True
                        print(f"Saved HF Adapter to {path}")
                    except Exception as e:
                        print(f"Warning: Failed to save HF Adapter: {e}")
                        import traceback
                        traceback.print_exc()
                    finally:
                        del full_state_dict
                else:
                    del full_state_dict

            except ImportError as e:
                if is_main_rank():
                    print(f"Warning: PEFT not available for HF adapter export: {e}")

        # ------------------------------------------------------------------
        # Signal DisGenerator ONLY after adapter is fully on disk.
        # finalize_policy() updates the symlink + .policy_ready file.
        # ------------------------------------------------------------------
        if adapter_saved:
            self.checkpoint_manager.finalize_policy(path)

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
