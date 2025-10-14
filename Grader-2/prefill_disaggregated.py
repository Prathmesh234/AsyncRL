"""Prefill script for disaggregated inference with LMCache using the NIXL backend.

This script mirrors the configuration of the original grader but drops the
ZeroMQ sockets in favour of the built-in NIXL transport supported by the
`LMCacheConnectorV1` integration inside vLLM. The goal is to run the prefill
stage on the grading host and publish the KV cache so that downstream decode
workers can attach with the same cache namespace.
"""

import argparse
import os
import socket
import uuid

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

PROMPT = os.environ.get(
    "GRADER_PROMPT",
    "You are grading a reinforcement learning assignment. Provide ONLY a numerical score from 1 to 5 representing the quality of"
    " the completion (1=poor, 2=below average, 3=average, 4=good, 5=excellent). Respond with ONLY the number, no additional text or explanation.",
)


def build_llm(args: argparse.Namespace) -> tuple[LLM, KVTransferConfig]:
    """Initialise the vLLM engine together with the LMCache connector.

    The configuration matches the original grader (2-way tensor parallelism
    against the `openai/gpt-oss-120b` checkpoint) but relies exclusively on the
    NIXL backend that ships with LMCache. No ZeroMQ endpoints are required – the
    transport is managed internally by LMCache once both prefill and decode
    workers use the same cache namespace.
    """

    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "name": args.cache_namespace,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "transfer_backend": "nixl",
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "compression": args.compression,
        },
    )

    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tensor_parallel_size,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_lora=False,
        enable_chunked_prefill=True,
        kv_transfer_config=kv_transfer_config,
    )

    return llm, kv_transfer_config


def run_prefill(llm: LLM, args: argparse.Namespace) -> str:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    cache_id = f"prefill-{uuid.uuid4()}"

    print(
        f"[Prefill] Running on host={socket.gethostname()} | prompt length={len(PROMPT)} | cache namespace={args.cache_namespace}"
    )
    print(f"[Prefill] Sampling parameters: {sampling_params}.")
    print(f"[Prefill] Launching prefill pass for cache_id={cache_id} using LMCache+NIXL...")

    llm.generate(
        prompts=[PROMPT],
        sampling_params=sampling_params,
    )

    print("[Prefill] Prefill completed successfully.")
    print(f"[Prefill] Cache URI for downstream decode: {cache_id}")
    return cache_id


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LMCache prefill stage")
    parser.add_argument(
        "--model",
        default="openai/gpt-oss-120b",
        help="Model identifier to load with vLLM",
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=2,
        help="Tensor parallel degree used for the model",
    )
    parser.add_argument(
        "--cache-namespace",
        default="grader/prefill",
        help="Shared LMCache namespace for the KV tensors",
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7, help="Sampling temperature",
    )
    parser.add_argument(
        "--top-p", type=float, default=0.95, help="Top-p nucleus sampling",
    )
    parser.add_argument(
        "--max-tokens", type=int, default=256, help="Maximum decode tokens",
    )
    parser.add_argument(
        "--repetition-penalty",
        type=float,
        default=1.0,
        help="Repetition penalty for sampling",
    )
    parser.add_argument(
        "--compression",
        default="lz4",
        help="Compression algorithm used by LMCache (e.g. lz4, none, zstd)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm, _ = build_llm(args)
    cache_uri = run_prefill(llm, args)
    print("[Prefill] Prefill stage finished – share the cache URI with decode workers.")
    print(f"[Prefill] Final cache URI: {cache_uri}")


if __name__ == "__main__":
    main()
