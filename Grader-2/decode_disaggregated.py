"""Decode script for disaggregated inference with LMCache using the NIXL backend.

The decoder attaches to the cache namespace populated by the prefill worker and
completes generation using the cached KV tensors. This mirrors the original
grader configuration while avoiding any ZeroMQ plumbing – LMCache's NIXL
transport handles the rendezvous between prefill and decode workers as long as
both share the same namespace.
"""

import argparse
import os
import socket

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

PROMPT = os.environ.get(
    "GRADER_PROMPT",
    "You are grading a reinforcement learning assignment. Provide ONLY a numerical score from 1 to 5 representing the quality of"
    " the completion (1=poor, 2=below average, 3=average, 4=good, 5=excellent). Respond with ONLY the number, no additional text or explanation.",
)


def build_llm(args: argparse.Namespace) -> LLM:
    kv_transfer_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_consumer",
        kv_connector_extra_config={
            "name": args.cache_namespace,
            "tensor_parallel_size": args.tensor_parallel_size,
            "pipeline_parallel_size": 1,
            "transfer_backend": "nixl",
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
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
    return llm


def run_decode(llm: LLM, args: argparse.Namespace) -> None:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )

    print(
        f"[Decode] Running on host={socket.gethostname()} | prompt length={len(PROMPT)} | cache namespace={args.cache_namespace}"
    )
    print(f"[Decode] Using sampling params: {sampling_params}.")
    print(f"[Decode] Fetching KV cache with cache URI: {args.cache_uri}")

    outputs = llm.generate(
        prompts=[PROMPT],
        sampling_params=sampling_params,
    )

    for output in outputs:
        print(f"[Decode] cache_uri={args.cache_uri} -> {output.outputs[0].text}")

    print("[Decode] Finished decoding using LMCache + NIXL transfer.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LMCache decode stage")
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
        help="LMCache namespace populated by the prefill worker",
    )
    parser.add_argument(
        "--cache-uri",
        required=True,
        help="Cache URI returned by the prefill stage",
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
        help="Repetition penalty for decoding",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    llm = build_llm(args)
    run_decode(llm, args)


if __name__ == "__main__":
    main()
