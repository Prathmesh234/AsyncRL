#!/usr/bin/env python3
"""
Download the base Qwen3-4B-Thinking model from HuggingFace.

This script downloads the base model (without any finetuning) to a local directory
for use with DisGenerator when --use-base-model flag is set.

Usage:
    python download_base_model.py
    python download_base_model.py --model Qwen/Qwen3-4B-Thinking-2507
    python download_base_model.py --output-dir ../models/qwen3-4b-base
"""

import argparse
import os
from pathlib import Path


def download_model(model_name: str, output_dir: str):
    """Download a model from HuggingFace to a local directory."""
    from huggingface_hub import snapshot_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"Downloading model: {model_name}")
    print(f"Output directory: {output_path.absolute()}")
    print()

    # Download the model
    snapshot_download(
        repo_id=model_name,
        local_dir=str(output_path),
        local_dir_use_symlinks=False,  # Copy files instead of symlinking
        resume_download=True,  # Resume if interrupted
    )

    print()
    print(f"Model downloaded successfully to: {output_path.absolute()}")
    print()
    print("To use this model with DisGenerator:")
    print(f"  MODEL={output_path.absolute()} ./scripts/start_all.sh 1p1d --use-base-model")


def main():
    parser = argparse.ArgumentParser(
        description="Download base model from HuggingFace"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="Qwen/Qwen3-4B-Thinking-2507",
        help="HuggingFace model ID (default: Qwen/Qwen3-4B-Thinking-2507)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory (default: ../models/<model-name>)"
    )

    args = parser.parse_args()

    # Default output directory based on model name
    if args.output_dir is None:
        model_dir_name = args.model.replace("/", "_")
        script_dir = Path(__file__).parent
        output_dir = script_dir.parent / "models" / model_dir_name
    else:
        output_dir = Path(args.output_dir)

    download_model(args.model, str(output_dir))


if __name__ == "__main__":
    main()
