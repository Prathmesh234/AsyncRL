#!/usr/bin/env python3
"""
DisGenerator CLI Runner

Entry point for running the trajectory generator.
Supports both batch processing from files and interactive mode.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import List

# Add parent directory for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SERVING_DIR not in sys.path:
    sys.path.insert(0, SERVING_DIR)

from config import config
from generator import DisGenerator

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def load_prompts(prompts_file: str) -> List[str]:
    """Load prompts from file."""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#'):
                prompts.append(line)
    return prompts


def load_prompts_jsonl(prompts_file: str) -> List[str]:
    """Load prompts from JSONL file."""
    prompts = []
    with open(prompts_file, 'r') as f:
        for line in f:
            if line.strip():
                data = json.loads(line)
                # Support various formats
                if isinstance(data, dict):
                    prompt = data.get('prompt') or data.get('query') or data.get('text', '')
                else:
                    prompt = str(data)
                if prompt:
                    prompts.append(prompt)
    return prompts


def get_default_prompts() -> List[str]:
    """Return default test prompts."""
    return [
        "What is the capital of France? Search the web to verify.",
        "Write a Python function to calculate the nth Fibonacci number. Test it using code execution.",
        "How do I list all resource groups in Azure? Use the azure tool.",
        "What is machine learning? Explain with examples from web search.",
        "Create a simple hello world script in Python and execute it.",
    ]


async def run_batch_generation(
    prompts: List[str],
    output_dir: str,
    batch_size: int = 10
) -> None:
    """Run batch generation on prompts."""
    
    os.makedirs(output_dir, exist_ok=True)
    
    total_prompts = len(prompts)
    total_batches = (total_prompts + batch_size - 1) // batch_size
    
    logger.info(f"Processing {total_prompts} prompts in {total_batches} batches")
    logger.info(f"Output directory: {output_dir}")
    
    async with DisGenerator() as generator:
        for batch_idx in range(total_batches):
            start_idx = batch_idx * batch_size
            end_idx = min(start_idx + batch_size, total_prompts)
            batch_prompts = prompts[start_idx:end_idx]
            
            logger.info(f"Processing batch {batch_idx + 1}/{total_batches} "
                       f"(prompts {start_idx + 1}-{end_idx})")
            
            # Generate output filename
            timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
            output_file = os.path.join(
                output_dir,
                f"batch_{batch_idx + 1:05d}_{timestamp}.jsonl"
            )
            
            # Generate and save
            await generator.generate_and_save(batch_prompts, output_file)
            
            logger.info(f"Batch {batch_idx + 1} complete: {output_file}")
    
    logger.info("All batches complete!")


async def run_interactive() -> None:
    """Run in interactive mode."""
    print("\n" + "=" * 60)
    print("  DisGenerator Interactive Mode")
    print("=" * 60)
    print("Enter prompts one per line. Type 'generate' to process.")
    print("Type 'quit' to exit.\n")
    
    prompts = []
    
    while True:
        try:
            line = input("> ").strip()
        except EOFError:
            break
        
        if line.lower() == 'quit':
            break
        elif line.lower() == 'generate':
            if prompts:
                async with DisGenerator() as generator:
                    batches = await generator.generate_batch(prompts)
                    
                    print("\n" + "-" * 40)
                    for batch in batches:
                        print(f"\nPrompt: {batch.prompt[:50]}...")
                        for i, comp in enumerate(batch.completions):
                            print(f"  Completion {i+1}: reward={comp.reward:.3f}, "
                                  f"turns={comp.num_turns}, "
                                  f"tools={len(comp.tool_calls)}")
                    print("-" * 40 + "\n")
                
                prompts = []
            else:
                print("No prompts entered. Enter prompts first.")
        elif line:
            prompts.append(line)
            print(f"  Added prompt #{len(prompts)}")


def main():
    parser = argparse.ArgumentParser(
        description="DisGenerator - Distributed Trajectory Generator for GRPO Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default test prompts
  python run.py
  
  # Process prompts from file
  python run.py --prompts-file prompts.txt --output-dir ./data/gen
  
  # Process JSONL with custom batch size
  python run.py --prompts-file data.jsonl --batch-size 20
  
  # Interactive mode
  python run.py --interactive
"""
    )
    
    parser.add_argument(
        "--prompts-file",
        type=str,
        help="Path to prompts file (.txt or .jsonl)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=config.output_dir,
        help=f"Output directory (default: {config.output_dir})"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=config.prompts_per_batch,
        help=f"Number of prompts per batch (default: {config.prompts_per_batch})"
    )
    parser.add_argument(
        "--interactive",
        action="store_true",
        help="Run in interactive mode"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Run in appropriate mode
    if args.interactive:
        asyncio.run(run_interactive())
    else:
        # Load prompts
        if args.prompts_file:
            if not os.path.exists(args.prompts_file):
                logger.error(f"Prompts file not found: {args.prompts_file}")
                sys.exit(1)
            
            if args.prompts_file.endswith('.jsonl'):
                prompts = load_prompts_jsonl(args.prompts_file)
            else:
                prompts = load_prompts(args.prompts_file)
            
            logger.info(f"Loaded {len(prompts)} prompts from {args.prompts_file}")
        else:
            prompts = get_default_prompts()
            logger.info(f"Using {len(prompts)} default test prompts")
        
        # Run generation
        asyncio.run(run_batch_generation(
            prompts,
            args.output_dir,
            args.batch_size
        ))


if __name__ == "__main__":
    main()
