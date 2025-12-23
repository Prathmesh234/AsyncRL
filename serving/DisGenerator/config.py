"""
DisGenerator Configuration

Central configuration for the Disaggregated Prefill-Decode Trajectory Generator.
Uses 2P/2D architecture:
- 2 Prefill GPUs (0, 1) for processing prompts/history
- 2 Decode GPUs (2, 3) for token generation
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ServerConfig:
    """vLLM server configuration."""
    host: str = "localhost"
    port: int = 8100
    
    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"
    
    @property
    def completions_url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"


@dataclass
class DisGeneratorConfig:
    """Main configuration for DisGenerator."""
    
    # Model Configuration
    # Base model for vLLM engine (from DisTrainer train_config.toml)
    model_name: str = os.getenv("MODEL_NAME", "Qwen/Qwen3-4B-Thinking-2507")
    
    # LoRA adapter from DisTrainer (set to empty string to disable)
    lora_adapter_path: str = os.getenv("LORA_ADAPTER_PATH", "serving/DisTrainer/checkpoints/Qwen3-4B-Thinking-Policy-v1/latest_adapter")
    lora_adapter_name: str = os.getenv("LORA_ADAPTER_NAME", "grpo-adapter")
    
    # Server Configuration (Disaggregated)
    prefill_server: ServerConfig = field(default_factory=lambda: ServerConfig(
        host=os.getenv("PREFILL_HOST", "localhost"),
        port=int(os.getenv("PREFILL_PORT", "8100"))
    ))
    decode_server: ServerConfig = field(default_factory=lambda: ServerConfig(
        host=os.getenv("DECODE_HOST", "localhost"),
        port=int(os.getenv("DECODE_PORT", "8200"))
    ))
    
    # Generation Parameters
    prompts_per_batch: int = int(os.getenv("PROMPTS_PER_BATCH", "10"))
    completions_per_prompt: int = int(os.getenv("COMPLETIONS_PER_PROMPT", "8"))
    max_tokens_per_completion: int = int(os.getenv("MAX_TOKENS_PER_COMPLETION", "4096"))
    max_turns: int = int(os.getenv("MAX_TURNS", "8"))  # Max tool call turns
    
    # Sampling Parameters
    temperature: float = float(os.getenv("TEMPERATURE", "0.7"))
    top_p: float = float(os.getenv("TOP_P", "0.9"))
    
    # Tool Calling Configuration
    # Note: We intercept tool tags in the streaming loop manually to ensure
    # we capture the full closing tag. We only stop the server on </solution>.
    stop_tokens: List[str] = field(default_factory=lambda: [
        "</solution>"
    ])
    
    intercept_tags: List[str] = field(default_factory=lambda: [
        "</web>", "</code>", "</azure>"
    ])
    
    # Timeouts
    tool_timeout_s: int = int(os.getenv("TOOL_TIMEOUT_S", "30"))
    grader_timeout_s: int = int(os.getenv("GRADER_TIMEOUT_S", "60"))
    request_timeout_s: int = int(os.getenv("REQUEST_TIMEOUT_S", "120"))
    
    # Output Configuration
    output_dir: str = os.getenv("OUTPUT_DIR", "./data/generations")
    batches_per_file: int = int(os.getenv("BATCHES_PER_FILE", "1"))
    
    # Service Bus Configuration
    service_bus_connection_string: Optional[str] = os.getenv("SERVICE_BUS_CONNECTION_STRING")
    command_topic_name: str = os.getenv("COMMAND_TOPIC_NAME", "commandtopic")
    reward_topic_name: str = os.getenv("REWARD_TOPIC_NAME", "rewardtopic")
    web_subscription_name: str = os.getenv("WEB_SUBSCRIPTION_NAME", "websubscription")
    code_subscription_name: str = os.getenv("CODE_SUBSCRIPTION_NAME", "codesubscription")
    grader_subscription_name: str = os.getenv("GRADER_SUBSCRIPTION_NAME", "gradersubscription")
    grader_reward_subscription_name: str = os.getenv("GRADER_REWARD_SUBSCRIPTION_NAME", "webrewardsubscription")
    
    # System Prompt
    system_prompt: str = os.getenv("SYSTEM_PROMPT", """You are an orchestrator agent. Decide when to use tools and when to answer directly.
Use private reasoning between <think> and </think>; never reveal it.
You may call only these tools (schemas are provided separately):
• web.search(q, k) – retrieve docs/snippets. Use <web>{"type": "web", "q": "search terms", "k": INTEGER}</web>. 
• code.exec(cmd, cwd, timeout_s) – run commands in /workspace. <code>{"type": "code", "code_command": "shell command"}</code>
• azure.run(args[]) – whitelisted az subcommands; prefer idempotent flags. <azure>{"type": "azure", "azure_command": "az subcommand"}</azure>

When you have the final answer, wrap it in <solution>your answer</solution> tags.""")
    
    # Logging
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    debug: bool = os.getenv("DEBUG", "0") == "1"
    
    def validate(self) -> bool:
        """Validate configuration."""
        if not self.service_bus_connection_string:
            print("[WARNING] SERVICE_BUS_CONNECTION_STRING not set. Tool calls will fail.")
        return True
    
    @property
    def total_trajectories_per_batch(self) -> int:
        """Total number of trajectories generated per batch."""
        return self.prompts_per_batch * self.completions_per_prompt


# Singleton instance
config = DisGeneratorConfig()


# GPU Configuration for vLLM servers
VLLM_SERVER_CONFIG = {
    "prefill": {
        "cuda_visible_devices": "0,1",
        "tensor_parallel_size": 2,
        "port": 8100,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": True,
        "max_num_seqs": 128,
    },
    "decode": {
        "cuda_visible_devices": "2,3",
        "tensor_parallel_size": 2,
        "port": 8200,
        "max_model_len": 32768,
        "gpu_memory_utilization": 0.85,
        "enable_prefix_caching": True,
        "max_num_seqs": 128,
    }
}
