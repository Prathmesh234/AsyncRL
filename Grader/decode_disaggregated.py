"""Decode script for disaggregated inference with vLLM LMCache.

This script attaches to an LMCache server that previously stored KV tensors via
`prefill_disaggregated.py`. The decode stage can run on another machine (or GPU)
and consumes the cached prompt to finish the generation. Rich logging is
provided to visualise NIXL transfers and ZeroMQ socket usage.
"""

import argparse
import os
import socket

from vllm import LLM, SamplingParams
from vllm.lmc import LMCacheClient

PROMPT = os.environ.get(
    "GRADER_PROMPT",
    "You are grading a reinforcement learning assignment. Explain the policy "
    "updates and provide actionable feedback for the student.",
)


def build_llm(args: argparse.Namespace) -> LLM:
    print("[Decode] Initialising LLM with GPT-120B-OSS for decoding...")
    llm = LLM(
        model="gpt-120b-oss",
        tensor_parallel_size=2,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_lora=False,
        chunked_prefill_enabled=True,
    )
    print("[Decode] LLM engine warmed up with prefix caching enabled.")
    return llm


def build_cache_client(args: argparse.Namespace) -> LMCacheClient:
    print("[Decode] Connecting to LMCache server via NIXL transfer backend...")
    client = LMCacheClient(
        cache_uri=args.cache_uri,
        zmq_control_endpoint=args.control_endpoint,
        zmq_data_endpoint=args.data_endpoint,
        transfer_backend="nixl",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
    )
    print(
        f"[Decode] Subscribed to ZeroMQ control={args.control_endpoint} "
        f"data={args.data_endpoint}."
    )
    return client


def run_decode(llm: LLM, client: LMCacheClient, args: argparse.Namespace) -> None:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    print(f"[Decode] Using sampling params: {sampling_params}.")
    print(f"[Decode] Fetching KV cache from: {args.cache_uri} ...")
    cache_state = client.materialise()
    print("[Decode] KV cache materialised. Beginning decode phase...")

    outputs = llm.decode(
        cache_state=cache_state,
        sampling_params=sampling_params,
        prompt=PROMPT,
        enable_kv_cache=True,
    )
    print("[Decode] Decode completed. Generated text:")
    for output in outputs:
        print(f"[Decode] request_id={output.request_id} -> {output.outputs[0].text}")
    print("[Decode] Finished decoding using LMCache + NIXL transfer.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LMCache decode stage")
    parser.add_argument(
        "--control-endpoint",
        default="tcp://127.0.0.1:5555",
        help="ZeroMQ control endpoint published by the prefill host",
    )
    parser.add_argument(
        "--data-endpoint",
        default="tcp://127.0.0.1:6000",
        help="ZeroMQ data endpoint published by the prefill host",
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
    args = parser.parse_args()

    print(
        f"[Decode] Hostname: {socket.gethostname()} | Prompt length: {len(PROMPT)} characters"
    )
    llm = build_llm(args)
    client = build_cache_client(args)
    run_decode(llm, client, args)


if __name__ == "__main__":
    main()
