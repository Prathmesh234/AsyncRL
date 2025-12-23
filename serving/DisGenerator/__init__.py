"""
DisGenerator: Distributed Trajectory Generator

A high-throughput trajectory generation service for GRPO training data collection.
Uses disaggregated prefill-decode architecture for efficient RL data generation.

Components:
- generator.py: Main async orchestrator
- config.py: Configuration management
- tool_executor.py: Async tool execution via Service Bus

Usage:
    async with DisGenerator() as gen:
        batches = await gen.generate_batch(prompts)
        await gen.generate_and_save(prompts, "output.jsonl")
"""

from .config import config, DisGeneratorConfig, VLLM_SERVER_CONFIG
from .generator import DisGenerator, CompletionResult, TrajectoryBatch
from .tool_executor import AsyncToolExecutor, execute_tool, grade_completion

__all__ = [
    # Config
    "config",
    "DisGeneratorConfig", 
    "VLLM_SERVER_CONFIG",
    # Generator
    "DisGenerator",
    "CompletionResult",
    "TrajectoryBatch",
    # Tool Executor
    "AsyncToolExecutor",
    "execute_tool",
    "grade_completion",
]

__version__ = "0.1.0"
