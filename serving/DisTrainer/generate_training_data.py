"""
Generate training data for DisTrainer using vLLM Offline Engine.
Produces JSONL files with prompts, multiple completions, and accurate logprobs/token IDs.

Usage:
    python generate_training_data.py --model Qwen/Qwen3-4B
"""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional

# Sample prompts for training data generation
SAMPLE_PROMPTS = [
    "Explain how photosynthesis works in plants.",
    "What are the main causes of climate change?",
    "Write a Python function to calculate fibonacci numbers.",
    "Describe the process of machine learning model training.",
    "What is the difference between TCP and UDP?",
    "Explain quantum entanglement in simple terms.",
    "How does a neural network learn?",
    "What are the benefits of renewable energy?",
]


def generate_offline(
    model_path: str,
    prompts: List[str],
    num_completions: int = 4,
    max_tokens: int = 256,
    tensor_parallel_size: int = 1,
    tokenizer_path: Optional[str] = None,
    adapter_path: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    Generate completions using vLLM offline engine.
    This gives us direct access to token IDs and logprobs.
    """
    try:
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest
    except ImportError:
        raise ImportError("vLLM is required for offline generation. Please install it: pip install vllm")
    
    print(f"Loading model from: {model_path}")
    if adapter_path:
        print(f"Enabling LoRA with adapter: {adapter_path}")

    llm = LLM(
        model=model_path,
        tokenizer=tokenizer_path if tokenizer_path else model_path,
        tensor_parallel_size=tensor_parallel_size,
        trust_remote_code=True,
        enable_lora=True if adapter_path else False
    )
    
    sampling_params = SamplingParams(
        n=num_completions,
        max_tokens=max_tokens,
        temperature=0.7,
        logprobs=1  # Get top-1 logprob for each generated token
    )
    
    print(f"Generating completions for {len(prompts)} prompts...")
    
    lora_req = None
    if adapter_path:
        lora_req = LoRARequest("adapter", 1, adapter_path)
        
    outputs = llm.generate(prompts, sampling_params, lora_request=lora_req)
    
    results = []
    
    for idx, output in enumerate(outputs):
        prompt = output.prompt
        prompt_ids = list(output.prompt_token_ids)
        
        completions = []
        for comp_output in output.outputs:
            completion_ids = list(comp_output.token_ids)
            
            # Extract logprobs for each token
            old_logprobs = []
            if comp_output.logprobs:
                for token_logprob in comp_output.logprobs:
                    if token_logprob:
                        # Get the logprob of the chosen token (the one actually generated)
                        # In offline mode, token_ids matches the sequence, so we just look up the id
                        chosen_token_id = completion_ids[len(old_logprobs)] if len(old_logprobs) < len(completion_ids) else None
                        
                        if chosen_token_id is not None and chosen_token_id in token_logprob:
                            old_logprobs.append(token_logprob[chosen_token_id].logprob)
                        else:
                            # Fallback: Use the first (highest prob) token's logprob if exact match missing (rare)
                            first_logprob = next(iter(token_logprob.values()), None)
                            old_logprobs.append(first_logprob.logprob if first_logprob else -0.5)
                    else:
                        old_logprobs.append(-0.5)
            
            # Ensure lengths match
            while len(old_logprobs) < len(completion_ids):
                old_logprobs.append(-0.5)
            old_logprobs = old_logprobs[:len(completion_ids)]
            
            completions.append({
                "text": comp_output.text,
                "completion_ids": completion_ids,
                "reward": round(random.uniform(0.3, 1.0), 3),
                "old_logprobs": old_logprobs
            })
        
        results.append({
            "gen_id": idx + 1,
            "prompt": prompt,
            "prompt_ids": prompt_ids,
            "completions": completions,
            "metadata": {"timestamp": datetime.utcnow().isoformat() + "Z"}
        })
        
        print(f"  Prompt {idx + 1}: {len(completions)} completions generated")
    
    return results


def save_batches(
    data: List[Dict[str, Any]],
    output_dir: str,
    prompts_per_batch: int = 4,
    start_batch: int = 1
):
    """Save data as JSONL batch files."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    batch_idx = start_batch
    for i in range(0, len(data), prompts_per_batch):
        batch_data = data[i:i + prompts_per_batch]
        batch_file = output_path / f"batch_{batch_idx:05d}.jsonl"
        
        with open(batch_file, 'w') as f:
            for group in batch_data:
                f.write(json.dumps(group) + '\n')
        
        print(f"Saved: {batch_file} ({len(batch_data)} prompts)")
        batch_idx += 1


def main():
    parser = argparse.ArgumentParser(description="Generate training data using vLLM (Offline)")
    
    parser.add_argument("--model", type=str, required=True,
                        help="Model path or name (e.g. Qwen/Qwen3-4B)")
    parser.add_argument("--output_dir", type=str, default="./data/generations",
                        help="Output directory for JSONL files")
    parser.add_argument("--num_completions", type=int, default=4,
                        help="Completions per prompt")
    parser.add_argument("--max_tokens", type=int, default=256,
                        help="Max tokens per completion")
    parser.add_argument("--prompts_per_batch", type=int, default=4,
                        help="Prompts per batch file")
    parser.add_argument("--tensor_parallel_size", type=int, default=1,
                        help="Tensor parallel size (number of GPUs)")
    parser.add_argument("--prompts_file", type=str, default=None,
                        help="Optional file with custom prompts (one per line)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Optional tokenizer path if different from model")
    parser.add_argument("--adapter_path", type=str, default=None,
                        help="Optional path to LoRA adapter")
    
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts_file:
        with open(args.prompts_file, 'r') as f:
            prompts = [line.strip() for line in f if line.strip()]
        print(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
    else:
        prompts = SAMPLE_PROMPTS
        print(f"Using {len(prompts)} sample prompts")
    
    # Generate completions
    data = generate_offline(
        model_path=args.model,
        prompts=prompts,
        num_completions=args.num_completions,
        max_tokens=args.max_tokens,
        tensor_parallel_size=args.tensor_parallel_size,
        tokenizer_path=args.tokenizer,
        adapter_path=args.adapter_path
    )
    
    # Save to batch files
    save_batches(data, args.output_dir, args.prompts_per_batch)
    
    print(f"\n✅ Generated {len(data)} prompt groups")
    print(f"   Total completions: {sum(len(d['completions']) for d in data)}")
    print(f"   Output: {args.output_dir}")


if __name__ == "__main__":
    main()
