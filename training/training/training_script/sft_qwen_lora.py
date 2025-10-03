#!/usr/bin/env python3
"""Supervised fine-tuning script for Qwen with LoRA adapters.

This script keeps the system prompt fixed while training only on the
assistant-generated text extracted from synthetic trajectory JSONL files.
"""

import argparse
import json
import os
from glob import glob
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from datasets import Dataset
from peft import LoraConfig
from safetensors.torch import load_file
from transformers import AutoTokenizer, BitsAndBytesConfig
from trl import SFTConfig, SFTTrainer


DEFAULT_MODEL = "Qwen/Qwen3-4B-Thinking-2507"
DEFAULT_DATA_DIR = "training/synthetic_trajectories/synthetic_traj_gpt"
RESPONSE_TEMPLATE = "<|assistant|>\n"


def _default_lora_dir() -> Optional[Path]:
    """Return the on-disk LoRA directory used elsewhere, if present."""

    repo_root = Path(__file__).resolve().parents[3]
    candidates = [
        repo_root / "GeneratorFS" / "qwen3-4b-thinking-openthoughts-lora",
        repo_root / "qwen3-4b-thinking-openthoughts-lora",
    ]
    for candidate in candidates:
        if (candidate / "adapter_model.safetensors").exists():
            return candidate
    return None


def load_trajectories(data_dir: Path) -> Tuple[Dataset, str]:
    """Load JSONL trajectories, returning a Dataset and the shared system prompt."""

    records: List[dict] = []
    system_prompts = set()

    jsonl_paths = sorted(glob(str(data_dir / "*.jsonl")))
    if not jsonl_paths:
        raise FileNotFoundError(f"No *.jsonl files found in {data_dir}")

    for path in jsonl_paths:
        with open(path, "r", encoding="utf-8") as handle:
            for line in handle:
                obj = json.loads(line)
                body = obj.get("response", {}).get("body", {})

                system_prompt = (body.get("instructions", "") or "").strip()
                if not system_prompt:
                    continue
                system_prompts.add(system_prompt)

                assistant_messages: List[str] = []
                for entry in body.get("output", []):
                    if entry.get("type") != "message":
                        continue
                    content = entry.get("content", [])
                    text = "\n".join(piece.get("text", "") for piece in content).strip()
                    if text:
                        assistant_messages.append(text)

                if not assistant_messages:
                    continue

                assistant_text = assistant_messages[-1].strip()
                if not assistant_text:
                    continue

                formatted = (
                    f"<|system|>\n{system_prompt}\n"
                    f"{RESPONSE_TEMPLATE}{assistant_text}"
                )

                records.append(
                    {
                        "text": formatted,
                        "assistant_text": assistant_text,
                        "system_prompt": system_prompt,
                    }
                )

    if not records:
        raise ValueError("No usable trajectories found in dataset.")

    if len(system_prompts) != 1:
        raise ValueError(
            "Expected exactly one shared system prompt; "
            f"found {len(system_prompts)} prompts."
        )

    dataset = Dataset.from_list(records)
    system_prompt = system_prompts.pop()
    return dataset, system_prompt


def maybe_split(dataset: Dataset, seed: int = 42) -> Tuple[Dataset, Optional[Dataset]]:
    """Optionally create a validation split if enough samples exist."""

    if len(dataset) < 3:
        return dataset, None

    test_size = max(1, int(len(dataset) * 0.1))
    split = dataset.train_test_split(test_size=test_size, seed=seed, shuffle=True)
    return split["train"], split["test"]


def build_tokenizer(model_name: str):
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        trust_remote_code=True,
        use_fast=False,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_bnb_config(enabled: bool) -> Optional[BitsAndBytesConfig]:
    if not enabled:
        return None
    return BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16 if torch.cuda.is_available() else None,
    )


def build_peft_config() -> LoraConfig:
    return LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )


def build_sft_config(
    output_dir: str,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    learning_rate: float,
    max_seq_length: int,
    eval_dataset: Optional[Dataset],
    use_cuda: bool,
    bnb_config: Optional[BitsAndBytesConfig],
):
    return SFTConfig(
        output_dir=output_dir,
        dataset_text_field="text",
        max_length=max_seq_length,
        packing=False,
        per_device_train_batch_size=batch_size,
        num_train_epochs=epochs,
        learning_rate=learning_rate,
        bf16=False,  # Use fp16 compute in 4-bit mode like notebook
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        logging_steps=10,
        save_strategy="epoch",
        eval_strategy="epoch" if eval_dataset is not None else "no",
        report_to="none",
        model_init_kwargs={
            "torch_dtype": torch.float16,
            "trust_remote_code": True,
            "quantization_config": bnb_config,
            "device_map": "auto",
            "low_cpu_mem_usage": True
        },
    )


def save_system_prompt(output_dir: Path, prompt: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "system_prompt.txt").write_text(prompt, encoding="utf-8")


def load_initial_lora_weights(peft_model, lora_dir: Optional[Path]) -> None:
    if lora_dir is None:
        return

    weights_path = lora_dir / "adapter_model.safetensors"
    if not weights_path.exists():
        return

    state_dict = load_file(str(weights_path))
    missing_keys, unexpected_keys = peft_model.load_state_dict(state_dict, strict=False)
    if missing_keys:
        print(f"Warning: missing keys when loading LoRA weights: {len(missing_keys)} entries")
    if unexpected_keys:
        print(f"Warning: unexpected keys when loading LoRA weights: {len(unexpected_keys)} entries")
    print(f"Loaded existing LoRA weights from {weights_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Supervised fine-tuning for Qwen.")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--data-dir", default=DEFAULT_DATA_DIR)

    # Default output to the existing LoRA directory to update in place
    default_lora_dir = _default_lora_dir()
    default_output = str(default_lora_dir) if default_lora_dir else "synthetic-qwen-lora"

    parser.add_argument("--output-dir", default=default_output)
    parser.add_argument(
        "--lora-weights",
        default=str(default_lora_dir) if default_lora_dir else None,
        help="Optional path to existing LoRA adapter weights to initialize from.",
    )
    parser.add_argument("--epochs", type=float, default=2.0)
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument("--learning-rate", type=float, default=2e-4)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    # Hard-coded overrides (per user request)
    repo_root = Path(__file__).resolve().parents[3]
    args.model = "Qwen/Qwen3-4B-Thinking-2507"
    # Build absolute dataset path from repo root so CWD does not matter
    hard_data_path = repo_root / "training" / "synthetic_trajectories" / "synthetic_traj_gpt"
    args.data_dir = str(hard_data_path)
    args.output_dir = "qwen3-4b-thinking-openthoughts-lora"
    args.lora_weights = args.output_dir  # reuse same dir for updating existing LoRA

    # Fallback: auto-detect a directory with jsonl files if the hard-coded one is empty
    hard_data = Path(args.data_dir)
    if (not hard_data.exists()) or (not list(hard_data.glob('*.jsonl'))):
        root_scan = repo_root / "training" / "synthetic_trajectories"
        jsonl_candidates = list(root_scan.rglob('*.jsonl')) if root_scan.exists() else []
        if jsonl_candidates:
            detected = jsonl_candidates[0].parent
            print(f"[AutoDetect] Overriding data_dir to {detected} (found JSONL)")
            args.data_dir = str(detected)
        else:
            print(f"[Warning] No JSONL found under {root_scan}.")

    print(f"Using hard-coded settings:\n  Model: {args.model}\n  Data dir: {args.data_dir}\n  Output/LoRA dir: {args.output_dir}\n  Epochs: {args.epochs}")

    print(f"Training for {args.epochs} epoch(s)")

    torch.manual_seed(args.seed)

    data_dir = Path(args.data_dir)
    dataset, system_prompt = load_trajectories(data_dir)
    train_dataset, eval_dataset = maybe_split(dataset, seed=args.seed)

    use_cuda = torch.cuda.is_available()
    bnb_config = build_bnb_config(use_cuda)

    peft_config = build_peft_config()
    sft_config = build_sft_config(
        output_dir=args.output_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.learning_rate,
        max_seq_length=args.max_seq_length,
        eval_dataset=eval_dataset,
        use_cuda=use_cuda,
        bnb_config=bnb_config,
    )

    trainer = SFTTrainer(
        model=args.model,
        args=sft_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        peft_config=peft_config,
    )

    # Disable KV cache for training to save memory
    if hasattr(trainer.model, "config"):
        trainer.model.config.use_cache = False

    lora_dir = Path(args.lora_weights) if args.lora_weights else None
    if lora_dir and lora_dir.exists():
        load_initial_lora_weights(trainer.model, lora_dir)

    print(f"\nStarting training... Will save to: {args.output_dir}")
    trainer.train()
    trainer.save_model(args.output_dir)
    
    # Save tokenizer separately 
    tokenizer = build_tokenizer(args.model)
    tokenizer.save_pretrained(args.output_dir)

    save_system_prompt(Path(args.output_dir), system_prompt)

    print(f"\n{'='*60}")
    print(f"✅ Training complete!")
    print(f"📁 LoRA adapters updated in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
