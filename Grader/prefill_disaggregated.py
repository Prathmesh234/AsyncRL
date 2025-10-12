"""Prefill script for disaggregated inference with vLLM LMCache.

This script performs the prompt prefill stage on a machine equipped with 2xA100
GPUs. It bootstraps the LMCache server that uses the NIXL transport backend to
stream key/value tensors to remote decoding workers over ZeroMQ. Extensive
print statements are included to highlight the LMCache lifecycle.
"""

import argparse
import os
import socket
import uuid

from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

PROMPT = os.environ.get(
    "GRADER_PROMPT",
    "You are grading a reinforcement learning assignment. Provide ONLY a numerical score from 1 to 5 representing the quality of the completion (1=poor, 2=below average, 3=average, 4=good, 5=excellent). Respond with ONLY the number, no additional text or explanation.",
)


def build_llm(args: argparse.Namespace) -> LLM:
    print("[Prefill] Initialising LLM with the GPT-120B-OSS checkpoint...")
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "name": args.cache_namespace,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "transfer_backend": "nixl",
            "zmq_control_endpoint": args.control_endpoint,
            "zmq_data_endpoint": args.data_endpoint,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "compression": args.compression,
        },
    )
    llm = LLM(
        model="gpt-120b-oss",
        tensor_parallel_size=2,
        trust_remote_code=True,
        enable_prefix_caching=True,
        enable_lora=False,
        chunked_prefill_enabled=True,
        kv_transfer_config=ktc,
    )
    print("[Prefill] LLM engine ready with tensor_parallel_size=2 (2xA100 GPUs).")
    return llm


def build_cache_server(llm: LLM, args: argparse.Namespace) -> KVTransferConfig:
    print("[Prefill] Creating LMCache configuration with NIXL transfer backend...")
    cache_config = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_producer",
        kv_connector_extra_config={
            "name": args.cache_namespace,
            "tensor_parallel_size": 2,
            "pipeline_parallel_size": 1,
            "transfer_backend": "nixl",
            "zmq_control_endpoint": args.control_endpoint,
            "zmq_data_endpoint": args.data_endpoint,
            "enable_prefix_caching": True,
            "enable_chunked_prefill": True,
            "compression": args.compression,
        },
    )
    print(
        f"[Prefill] ZeroMQ control endpoint: {args.control_endpoint}; data endpoint: {args.data_endpoint}"
    )
    print("[Prefill] Launching LMCache server... This binds to the ZeroMQ sockets.")
    server = cache_config  # In vLLM v1, LMCache is configured via KVTransferConfig
    print("[Prefill] LMCache server ready to accept prefill requests.")
    return server


def run_prefill(llm: LLM, server: KVTransferConfig, args: argparse.Namespace) -> str:
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    )
    request_id = f"prefill-{uuid.uuid4()}"
    print(f"[Prefill] Prepared sampling params: {sampling_params}.")
    print(f"[Prefill] Starting prefill for request_id={request_id}...")

    outputs = llm.generate(
        prompts=[PROMPT],
        sampling_params=sampling_params,
        request_id=request_id,
        use_cached_kv=False,
    )
    print("[Prefill] Prefill completed.")
    print(f"[Prefill] Cache URI for downstream decode: {request_id}")
    print(
        "[Prefill] KV cache shards stored using NIXL transport. Watch ZeroMQ logs "
        "for transfer acknowledgements."
    )
    return request_id


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LMCache prefill stage")
    parser.add_argument(
        "--control-endpoint",
        default="tcp://0.0.0.0:5555",
        help="ZeroMQ control endpoint for LMCache",
    )
    parser.add_argument(
        "--data-endpoint",
        default="tcp://0.0.0.0:6000",
        help="ZeroMQ data endpoint for LMCache",
    )
    parser.add_argument(
        "--cache-namespace",
        default="grader/prefill",
        help="LMCache namespace used to store KV entries",
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
        help="LMCache tensor compression (lz4, none, zstd)",
    )
    args = parser.parse_args()

    print(
        f"[Prefill] Hostname: {socket.gethostname()} | Prompt length: {len(PROMPT)} characters"
    )
    llm = build_llm(args)
    server = build_cache_server(llm, args)
    cache_uri = run_prefill(llm, server, args)
    print(
        "[Prefill] Prefill stage finished. Share the cache URI with the decode "
        "worker to continue generation."
    )
    print(f"[Prefill] Final cache URI: {cache_uri}")


if __name__ == "__main__":
    main()
