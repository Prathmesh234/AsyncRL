"""
Configuration for DisGenerator disaggregated serving.

This module provides configuration for the P2P NCCL disaggregated
prefill-decode architecture.
"""

import os
import re
import glob
from dataclasses import dataclass, field
from typing import Literal, Optional
from pathlib import Path


# Path to DisTrainer models directory (where policies are stored)
DISTRAINER_MODELS_DIR = os.getenv(
    "DISTRAINER_MODELS_DIR",
    os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "DisTrainer", "models"))
)


def get_latest_policy_path() -> Optional[str]:
    """
    Get the path to the latest policy checkpoint from DisTrainer/models.
    
    Returns:
        Path to the latest policy directory, or None if not found
    """
    if not os.path.exists(DISTRAINER_MODELS_DIR):
        return None
    
    # FIRST: Check for 'latest_adapter' symlink/directory
    latest_symlink = os.path.join(DISTRAINER_MODELS_DIR, "latest_adapter")
    if os.path.exists(latest_symlink):
        return latest_symlink

    # FALLBACK: Scan for explicit policy folders
    policies = []
    pattern = re.compile(r"policy-(\d+)")
    
    for entry in os.listdir(DISTRAINER_MODELS_DIR):
        full_path = os.path.join(DISTRAINER_MODELS_DIR, entry)
        if os.path.isdir(full_path):
            match = pattern.match(entry)
            if match:
                version = int(match.group(1))
                policies.append((version, full_path))
    
    if not policies:
        return None
    
    # Return path with highest version
    policies.sort(key=lambda x: x[0])
    return policies[-1][1]


@dataclass
class ServerConfig:
    """Configuration for a single vLLM server instance."""
    
    gpu_id: int
    http_port: int
    kv_port: int
    role: Literal["prefill", "decode"]
    
    @property
    def kv_role(self) -> str:
        return "kv_producer" if self.role == "prefill" else "kv_consumer"
    
    @property
    def kv_buffer_size(self) -> str:
        # Prefill needs less buffer (produces), decode needs more (consumes)
        return "1e9" if self.role == "prefill" else "8e9"
    
    @property
    def gpu_memory_utilization(self) -> float:
        # Decode needs headroom for incoming KV cache
        return 0.9 if self.role == "prefill" else 0.7


@dataclass
class DisGeneratorConfig:
    """Main configuration for DisGenerator."""
    
    # Model configuration (base model - matches DisTrainer)
    model: str = os.getenv("MODEL", "Qwen/Qwen3-4B-Thinking-2507")
    
    # LoRA adapter path - auto-detect latest policy from DisTrainer/models
    # Set to None to not use any adapter
    adapter_path: Optional[str] = field(default_factory=get_latest_policy_path)
    
    dtype: str = "float16"
    max_model_len: int = 32768
    max_num_batched_tokens: int = 32768
    max_num_seqs: int = 128
    tensor_parallel_size: int = 1
    seed: int = 1024
    trust_remote_code: bool = True
    enforce_eager: bool = True
    
    # GPU allocation (2P2D by default: 2 Prefill + 2 Decode for 4 GPUs)
    # For 2 GPUs, use 1P1D
    prefill_gpus: list[int] = field(default_factory=lambda: [0])
    decode_gpus: list[int] = field(default_factory=lambda: [1])
    
    # Port configuration
    prefill_base_http_port: int = 20001
    decode_base_http_port: int = 20002
    prefill_base_kv_port: int = 21001
    decode_base_kv_port: int = 22001
    
    # Proxy configuration
    proxy_ip: str = "0.0.0.0"
    proxy_port: int = 30001
    proxy_http_port: int = 10001
    
    # Timeouts
    server_timeout_seconds: int = 600
    request_timeout_hours: int = 6
    
    # NCCL configuration
    nccl_num_channels: int = 16
    send_type: str = "PUT_ASYNC"
    
    @property
    def prefill_servers(self) -> list[ServerConfig]:
        """Generate prefill server configurations."""
        return [
            ServerConfig(
                gpu_id=gpu_id,
                http_port=self.prefill_base_http_port + i * 2,
                kv_port=self.prefill_base_kv_port + i,
                role="prefill",
            )
            for i, gpu_id in enumerate(self.prefill_gpus)
        ]
    
    @property
    def decode_servers(self) -> list[ServerConfig]:
        """Generate decode server configurations."""
        return [
            ServerConfig(
                gpu_id=gpu_id,
                http_port=self.decode_base_http_port + i * 2,
                kv_port=self.decode_base_kv_port + i,
                role="decode",
            )
            for i, gpu_id in enumerate(self.decode_gpus)
        ]
    
    def get_kv_transfer_config(self, server: ServerConfig) -> dict:
        """Generate KV transfer config JSON for a server."""
        return {
            "kv_connector": "P2pNcclConnector",
            "kv_role": server.kv_role,
            "kv_buffer_size": server.kv_buffer_size,
            "kv_port": str(server.kv_port),
            "kv_connector_extra_config": {
                "proxy_ip": self.proxy_ip,
                "proxy_port": str(self.proxy_port),
                "http_port": str(server.http_port),
                "send_type": self.send_type,
                "nccl_num_channels": str(self.nccl_num_channels),
            },
        }


# Default configuration
default_config = DisGeneratorConfig()


# Preset configurations for different GPU setups
def get_config_1p1d() -> DisGeneratorConfig:
    """1 Prefill + 1 Decode (2 GPUs)"""
    return DisGeneratorConfig(
        prefill_gpus=[0],
        decode_gpus=[1],
    )


def get_config_2p2d() -> DisGeneratorConfig:
    """2 Prefill + 2 Decode (4 GPUs) - balanced"""
    return DisGeneratorConfig(
        prefill_gpus=[0, 1],
        decode_gpus=[2, 3],
    )


def get_config_1p3d() -> DisGeneratorConfig:
    """1 Prefill + 3 Decode (4 GPUs) - decode heavy"""
    return DisGeneratorConfig(
        prefill_gpus=[0],
        decode_gpus=[1, 2, 3],
    )


def get_config_3p1d() -> DisGeneratorConfig:
    """3 Prefill + 1 Decode (4 GPUs) - prefill heavy"""
    return DisGeneratorConfig(
        prefill_gpus=[0, 1, 2],
        decode_gpus=[3],
    )
