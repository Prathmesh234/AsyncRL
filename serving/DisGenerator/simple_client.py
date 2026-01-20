#!/usr/bin/env python3
"""
Orchestrator Client for DisGenerator.

This script initializes the AsyncBatchOrchestrator, loads prompts from a JSONL file,
and processes them through the disaggregated serving system with tool support.

Usage:
    python simple_client.py
"""

import argparse
import asyncio
import json
import os
import sys
import logging
import uuid
import time
from typing import List, Iterable

# Add parent directory to path to locate batch_orchestrator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batch_orchestrator import AsyncBatchOrchestrator, Trajectory
from policy_manager import PolicyManager
from config import DISTRAINER_MODELS_DIR
from parser import TOOL_TAGS
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:10001/v1/chat/completions")
# Use LoRA adapter name for vLLM routing (must match --lora-modules in vLLM)
MODEL = os.getenv("LORA_ADAPTER_NAME", os.getenv("MODEL", "grpo-adapter"))
TOKENIZER = os.getenv("TOKENIZER", "Qwen/Qwen3-4B-Thinking-2507")  # Tokenizer for token ID extraction
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.jsonl")
# Default output to DisTrainer's data/generations folder
OUTPUT_DIR = os.getenv("OUTPUT_DIR", os.path.abspath(os.path.join(os.path.dirname(__file__), "../DisTrainer/data/generations")))
NUM_GPU_WORKERS = int(os.getenv("NUM_GPU_WORKERS", "4"))
NUM_TOOL_WORKERS = int(os.getenv("NUM_TOOL_WORKERS", "32"))

# Orchestration Hyperparameters
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "10"))  # Number of prompts per batch file
NUM_COMPLETIONS = int(os.getenv("NUM_COMPLETIONS", "4")) # Number of completions per prompt (GRPO group size)
GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.9"))

# Hot-swap LoRA Policy Configuration
ENABLE_HOTSWAP = os.getenv("ENABLE_HOTSWAP", "true").lower() in ("true", "1", "yes")
POLICY_POLL_INTERVAL = float(os.getenv("POLICY_POLL_INTERVAL", "5.0"))  # Seconds between policy checks

# System prompt from .env (matches ToolGRPOTrainer)
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [Client] %(message)s')
logger = logging.getLogger("Client")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="DisGenerator Orchestrator Client")
    parser.add_argument(
        "--tool",
        action="append",
        default=[],
        help="Enable tool interception (repeatable or comma-separated): azure, code, web.",
    )
    return parser.parse_args()

def normalize_tools(raw_tools: Iterable[str]) -> List[str]:
    tools: List[str] = []
    for entry in raw_tools:
        for tool in entry.split(","):
            tool = tool.strip().lower()
            if tool:
                tools.append(tool)
    if not tools:
        return list(TOOL_TAGS)
    invalid = sorted(set(tools) - set(TOOL_TAGS))
    if invalid:
        raise ValueError(f"Unknown tool(s): {', '.join(invalid)}")
    return sorted(set(tools), key=tools.index)

async def load_prompts(file_path: str, system_prompt: str = "") -> List[Trajectory]:
    """Load prompts from a JSONL file and create Trajectory objects.
    
    If system_prompt is provided, it will be prepended as a system message.
    Reads expected_answer if present for solution verification.
    """
    trajectories = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                prompt_id = data.get("prompt_id", str(uuid.uuid4()))
                messages = data.get("messages", [])
                expected_answer = data.get("expected_answer", "")
                
                # Basic validation
                if not messages:
                    logger.warning(f"Skipping empty message for ID {prompt_id}")
                    continue
                
                # Prepend system prompt if provided and not already present
                if system_prompt:
                    has_system = any(msg.get("role") == "system" for msg in messages)
                    if not has_system:
                        messages = [{"role": "system", "content": system_prompt}] + messages

                traj = Trajectory(
                    id=prompt_id,
                    messages=messages,
                    completions=[],
                    status="QUEUED",
                    expected_answer=expected_answer
                )
                trajectories.append(traj)
    except FileNotFoundError:
        logger.error(f"Prompts file not found: {file_path}")
    except json.JSONDecodeError as e:
         logger.error(f"Error decoding JSONL: {e}")

    return trajectories

async def monitor_progress(orchestrator: AsyncBatchOrchestrator, total_tasks: int):
    """Monitor and print the progress of the orchestrator."""
    # This is a placeholder. Real implementation would track completed tasks 
    # via a shared counter or checking orchestrator state. 
    # For now, we run indefinitely or until interrupted.
    logger.info("Monitoring started... (Press Ctrl+C to stop)")
    while True:
        # In a real system, checking queue sizes gives an idea of progress
        q_task = orchestrator.task_queue.qsize()
        q_tool = orchestrator.tool_queue.qsize()
        logger.info(f"Queue Status -> Task: {q_task} | Tool: {q_tool}")
        if q_task == 0 and q_tool == 0:
             # Very naive completion check - waits for queues to drain. 
             # Does not account for active workers.
             # Ideally orchestrator exposes an active_count.
             pass
        await asyncio.sleep(5)

async def main():
    args = parse_args()
    enabled_tools = normalize_tools(args.tool)

    print(f"\n{'='*60}")
    print("DisGenerator Orchestrator Client")
    print(f"{'='*60}")
    print(f"  Proxy URL     : {PROXY_URL}")
    print(f"  Model         : {MODEL}")
    print(f"  Tokenizer     : {TOKENIZER}")
    print(f"  Prompts File  : {PROMPTS_FILE}")
    print(f"  Output Dir    : {OUTPUT_DIR}")
    print(f"  System Prompt : {'Yes (' + str(len(SYSTEM_PROMPT)) + ' chars)' if SYSTEM_PROMPT else 'No'}")
    print(f"  GPU Workers   : {NUM_GPU_WORKERS}")
    print(f"  Tool Workers  : {NUM_TOOL_WORKERS}")
    print(f"  Enabled Tools : {', '.join(enabled_tools)}")
    print(f"  ─────────────────────────────")
    print(f"  GRPO Hyperparameters:")
    print(f"  Batch Size    : {BATCH_SIZE} prompts ({BATCH_SIZE * NUM_COMPLETIONS} trajectories)")
    print(f"  Completions   : {NUM_COMPLETIONS} per prompt")
    print(f"  Temperature   : {GENERATION_TEMPERATURE}")
    print(f"  ─────────────────────────────")
    print(f"  Hot-Swap LoRA:")
    print(f"  Enabled       : {ENABLE_HOTSWAP}")
    if ENABLE_HOTSWAP:
        print(f"  Poll Interval : {POLICY_POLL_INTERVAL}s")
        print(f"  Models Dir    : {DISTRAINER_MODELS_DIR}")
    print(f"{'='*60}\n")

    # 1. Initialize PolicyManager for hot-swapping LoRA adapters
    policy_manager = None
    if ENABLE_HOTSWAP:
        logger.info("Initializing PolicyManager for hot-swap LoRA support...")
        policy_manager = PolicyManager(
            models_dir=DISTRAINER_MODELS_DIR,
            lora_name="grpo-adapter",
            poll_interval=POLICY_POLL_INTERVAL,
            enable_hotswap=True
        )
        policy_manager.start_watching()
        initial_policy = policy_manager.get_current_policy_info()
        logger.info(f"Initial policy: {initial_policy}")

    # 2. Initialize Orchestrator
    # batch_size in orchestrator is number of TRAJECTORIES (prompts * completions)
    orchestrator = AsyncBatchOrchestrator(
        proxy_url=PROXY_URL,
        model=MODEL,
        tokenizer_name=TOKENIZER,
        num_gpu_workers=NUM_GPU_WORKERS,
        num_tool_workers=NUM_TOOL_WORKERS,
        output_dir=OUTPUT_DIR,
        batch_size=BATCH_SIZE,  # Number of complete GROUPS (prompts) per batch
        num_completions_per_prompt=NUM_COMPLETIONS,
        generation_temperature=GENERATION_TEMPERATURE,
        enabled_tools=enabled_tools,
        policy_manager=policy_manager  # Enable hot-swap
    )

    # 3. Start Workers
    await orchestrator.start()

    # 3. Load Prompts (with system prompt from .env)
    trajectories = await load_prompts(PROMPTS_FILE, system_prompt=SYSTEM_PROMPT)
    logger.info(f"Loaded {len(trajectories)} trajectories.")
    if SYSTEM_PROMPT:
        logger.info(f"System prompt injected ({len(SYSTEM_PROMPT)} chars)")

    # 4. Enqueue Tasks
    for traj in trajectories:
        await orchestrator.add_trajectory(traj)

    # 5. Monitor execution
    try:
        # For this simple client, we just wait for user interrupt or until logic finishes
        # A more robust client would collect results and save to file.
        await monitor_progress(orchestrator, len(trajectories))
    except asyncio.CancelledError:
        logger.info("Client cancelled.")
    finally:
        # Flush any remaining trajectories to disk
        await orchestrator.flush_pending()
        await orchestrator.stop()

        # Stop policy watcher
        if policy_manager is not None:
            policy_manager.stop_watching()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
