#!/usr/bin/env python3
"""
Orchestrator Client for DisGenerator.

This script initializes the AsyncBatchOrchestrator, loads prompts from a JSONL file,
and processes them through the disaggregated serving system with tool support.

Usage:
    python simple_client.py
"""

import asyncio
import json
import os
import sys
import logging
import uuid
import time
from typing import List

# Add parent directory to path to locate batch_orchestrator
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from batch_orchestrator import AsyncBatchOrchestrator, Trajectory
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:10001/v1/chat/completions")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-4B-Thinking-2507")  # Can be LoRA adapter name
PROMPTS_FILE = os.path.join(os.path.dirname(__file__), "prompts.jsonl")
NUM_GPU_WORKERS = int(os.getenv("NUM_GPU_WORKERS", "4"))
NUM_TOOL_WORKERS = int(os.getenv("NUM_TOOL_WORKERS", "32"))

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] [Client] %(message)s')
logger = logging.getLogger("Client")

async def load_prompts(file_path: str) -> List[Trajectory]:
    """Load prompts from a JSONL file and create Trajectory objects."""
    trajectories = []
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                prompt_id = data.get("prompt_id", str(uuid.uuid4()))
                messages = data.get("messages", [])
                
                # Basic validation
                if not messages:
                    logger.warning(f"Skipping empty message for ID {prompt_id}")
                    continue

                traj = Trajectory(
                    id=prompt_id,
                    messages=messages,
                    completions=[],
                    status="QUEUED"
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
    print(f"\n{'='*60}")
    print("DisGenerator Orchestrator Client")
    print(f"{'='*60}")
    print(f"  Proxy URL     : {PROXY_URL}")
    print(f"  Model         : {MODEL}")
    print(f"  Prompts File  : {PROMPTS_FILE}")
    print(f"  GPU Workers   : {NUM_GPU_WORKERS}")
    print(f"  Tool Workers  : {NUM_TOOL_WORKERS}")
    print(f"{'='*60}\n")

    # 1. Initialize Orchestrator
    orchestrator = AsyncBatchOrchestrator(
        proxy_url=PROXY_URL,
        model=MODEL,
        num_gpu_workers=NUM_GPU_WORKERS,
        num_tool_workers=NUM_TOOL_WORKERS
    )

    # 2. Start Workers
    await orchestrator.start()

    # 3. Load Prompts
    trajectories = await load_prompts(PROMPTS_FILE)
    logger.info(f"Loaded {len(trajectories)} trajectories.")

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
        await orchestrator.stop()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
