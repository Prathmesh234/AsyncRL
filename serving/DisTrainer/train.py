"""
Entry point for DisTrainer.
Use with torchrun for distributed training.

Usage:
    CUDA_VISIBLE_DEVICES=0,1 torchrun --nproc_per_node=2 train.py --config config/train_config.toml
"""

import argparse
import torch.distributed as dist

from .server import run_server


def main():
    parser = argparse.ArgumentParser(description="DisTrainer - Distributed GRPO Training")
    parser.add_argument(
        "--config",
        type=str,
        default="config/train_config.toml",
        help="Path to TOML configuration file"
    )
    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host to bind the server to"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port to run the server on"
    )
    parser.add_argument(
        "--use-base-model",
        action="store_true",
        help="Use base model without LoRA adapter (creates fresh LoRA instead)"
    )

    args = parser.parse_args()

    # Run the server (handles distributed init internally)
    run_server(
        config_path=args.config,
        host=args.host,
        port=args.port,
        use_base_model=args.use_base_model
    )


if __name__ == "__main__":
    main()
