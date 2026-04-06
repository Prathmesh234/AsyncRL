import asyncio
import copy
import json
import logging
import time
import os
import sys
import uuid
import aiohttp
from typing import List, Dict, Any, Optional, Iterable
from dataclasses import dataclass, field

# Add parent directory to path to allow imports from serving root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import tool senders
from communication.command_sender import send_web_command
from communication.code_command_sender import send_code_command
from communication.azure_command_sender import send_azure_command
from parser import stream_parser, TOOL_TAGS

# Import helper modules
from utilities import get_next_batch_number, write_batch_file, ensure_output_dir
from data_processing import (
    messages_to_prompt_string,
    build_trajectory_record,
    build_completion_text,
    extract_prompt_messages
)
from reward_functions import compute_reward

# PolicyManager for hot-swapping LoRA adapters
from policy_manager import PolicyManager

# BufferQueue: in-memory async deque shared with the trainer
from buffer_queue import BufferQueue

# Tokenizer for proper token ID extraction
from transformers import AutoTokenizer

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Orchestrator")

@dataclass
class Trajectory:
    id: str
    messages: List[Dict[str, str]]
    completions: List[str] = field(default_factory=list)
    # Accumulator for streaming logprobs (token IDs are computed via local tokenizer)
    accumulated_logprobs: List[float] = field(default_factory=list)
    # Action mask: 1 = model-generated token, 0 = tool result token (skip in loss)
    action_mask: List[int] = field(default_factory=list)
    status: str = "QUEUED"
    created_at: float = field(default_factory=time.time)
    # GRPO: Group ID for grouping multiple completions per prompt
    group_id: str = ""
    # Expected answer for verification (used by solution_verifier)
    expected_answer: str = ""

class AsyncBatchOrchestrator:
    def __init__(
        self,
        proxy_url: str,
        model: str = "Qwen/Qwen3-4B-Thinking-2507",
        tokenizer_name: str = "Qwen/Qwen3-4B-Thinking-2507",
        num_gpu_workers: int = 4,
        num_tool_workers: int = 32,
        output_dir: str = None,
        batch_size: int = 10,
        num_completions_per_prompt: int = 4,
        generation_temperature: float = 1.0,
        enabled_tools: Optional[Iterable[str]] = None,
        policy_manager: Optional[PolicyManager] = None,
        buffer_queue: Optional[BufferQueue] = None,
    ):
        self.proxy_url = proxy_url
        self.model = model  # Can be base model or LoRA adapter name
        self.task_queue = asyncio.Queue()  # For GPU tasks
        self.tool_queue = asyncio.Queue()  # For Tool execution tasks
        self.enabled_tools = tuple(enabled_tools) if enabled_tools is not None else TOOL_TAGS

        # PolicyManager for hot-swapping LoRA adapters (optional)
        self.policy_manager = policy_manager

        # BufferQueue: when provided, completed groups are pushed here instead of
        # being written to disk.  Falls back to file-based output when None.
        self.buffer_queue = buffer_queue

        # File-based output (used only when buffer_queue is None)
        if buffer_queue is None:
            if output_dir is None:
                output_dir = os.path.abspath(
                    os.path.join(os.path.dirname(__file__), "../DisTrainer/data/generations")
                )
            self.output_dir = ensure_output_dir(output_dir)
            self.batch_counter = get_next_batch_number(self.output_dir)
        else:
            # Keep output_dir around in case flush_pending is called without a queue
            self.output_dir = output_dir
            self.batch_counter = 0

        self._batch_lock = asyncio.Lock()

        # Batch accumulation – collect N complete GROUPS (prompts) before flushing.
        # With buffer_queue: each complete group is pushed immediately (batch_size
        # is still used by flush_pending for leftover groups written to disk).
        # Without buffer_queue: batch_size groups are written to one JSONL file.
        self.batch_size = batch_size  # Number of prompts per batch file (file mode)
        self._pending_groups: Dict[str, List[Dict]] = {}  # group_id -> list of records

        # GRPO configuration
        self.num_completions_per_prompt = num_completions_per_prompt
        self.generation_temperature = generation_temperature

        self.num_gpu_workers = num_gpu_workers
        self.num_tool_workers = num_tool_workers
        self.workers: List[asyncio.Task] = []

        # Initialize tokenizer for proper token ID extraction
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")

        # Stop strings for vLLM to pause generation immediately on tool call
        self.tool_stop_tokens = [f"</{tool}>" for tool in self.enabled_tools]
        self.stop_tokens = [*self.tool_stop_tokens, "</solution>"]

        # Context window configuration
        self.max_model_len = 65536  # Model's max context length (64K)
        self.min_completion_tokens = 256  # Minimum tokens to reserve for completion
    
    async def start(self):
        """Start all worker tasks."""
        logger.info(f"Starting Orchestrator with {self.num_gpu_workers} GPU workers and {self.num_tool_workers} Tool workers.")
        
        # Spawn GPU workers
        for i in range(self.num_gpu_workers):
            task = asyncio.create_task(self.gpu_worker(i))
            self.workers.append(task)
            
        # Spawn Tool workers
        for i in range(self.num_tool_workers):
            task = asyncio.create_task(self.tool_worker(i))
            self.workers.append(task)
            
    async def stop(self):
        """Cancel all workers."""
        for task in self.workers:
            task.cancel()
        await asyncio.gather(*self.workers, return_exceptions=True)
        logger.info("Orchestrator stopped.")

    async def add_trajectory(self, traj: Trajectory):
        """
        Entry point: Add a new trajectory to the system.
        Spawns num_completions_per_prompt clones for GRPO group-relative training.
        Each clone shares the same group_id for grouping in the trainer.
        """
        # Generate a group_id if not already set
        group_id = traj.group_id or traj.id or str(uuid.uuid4())
        
        # Spawn N clones of this trajectory
        for i in range(self.num_completions_per_prompt):
            clone = Trajectory(
                id=f"{group_id}-{i}",
                messages=copy.deepcopy(traj.messages),
                completions=[],
                accumulated_logprobs=[],
                action_mask=[],
                status="QUEUED",
                group_id=group_id,
                expected_answer=traj.expected_answer  # Propagate for verification
            )
            await self.task_queue.put(clone)
        
        logger.info(f"[Group {group_id}] Spawned {self.num_completions_per_prompt} completions for GRPO")

    async def save_trajectory(self, traj: Trajectory):
        """
        Saves the completed trajectory to a JSONL file.
        
        Each trajectory is saved as an individual record with group_id.
        The DataLoader groups records by group_id into the format expected by compute_grpo_loss.
        
        Computes reward before saving using tool_reward, char_reward, and format_reward.
        """
        # Build completion text for reward computation
        completion_text = build_completion_text(traj.messages, traj.completions)
        
        # Extract prompt for reward computation
        prompt_messages = extract_prompt_messages(traj.messages)
        prompt_text = messages_to_prompt_string(prompt_messages, self.tokenizer)
        
        # Compute reward (async-safe as it's CPU-bound, run in thread pool)
        # Pass expected_answer for solution verification
        loop = asyncio.get_running_loop()
        reward = await loop.run_in_executor(
            None, 
            compute_reward, 
            completion_text, 
            prompt_text,
            traj.expected_answer  # For Prime Intellect solution verifier
        )
        
        logger.info(f"[Traj {traj.id}] Computed reward: {reward:.3f}")
        
        # Build the record using data_processing helper
        record = build_trajectory_record(
            traj_id=traj.id,
            group_id=traj.group_id,
            messages=traj.messages,
            completions=traj.completions,
            accumulated_logprobs=traj.accumulated_logprobs,
            action_mask=traj.action_mask,
            tokenizer=self.tokenizer,
            reward=reward
        )
        # Accumulate record by group_id for GRPO grouping
        async with self._batch_lock:
            group_id = record.get("group_id", traj.id)

            # Add to pending group
            if group_id not in self._pending_groups:
                self._pending_groups[group_id] = []
            self._pending_groups[group_id].append(record)

            group_size = len(self._pending_groups[group_id])
            logger.info(f"[Group {group_id}] Completion {group_size}/{self.num_completions_per_prompt}")

            # Once all completions for this group have arrived, assemble and dispatch.
            if group_size >= self.num_completions_per_prompt:
                group_records = self._pending_groups.pop(group_id)
                first = group_records[0]
                grouped_record = {
                    "group_id": group_id,
                    "prompt": first.get("prompt"),
                    "prompt_ids": first.get("prompt_ids"),
                    "completions": [r.get("completion") for r in group_records],
                    "metadata": {
                        "num_completions": len(group_records),
                        "timestamp": first.get("metadata", {}).get("timestamp"),
                    },
                }

                if self.buffer_queue is not None:
                    # ── Queue mode: push directly into the in-memory deque ──
                    added = await self.buffer_queue.put(grouped_record)
                    q_stats = self.buffer_queue.stats()
                    logger.info(
                        f"[Group {group_id}] → BufferQueue "
                        f"(size={q_stats['size']}/{q_stats['maxlen']}, "
                        f"evicted={'yes' if not added else 'no'})"
                    )
                else:
                    # ── File mode (legacy): batch up and write to JSONL ──
                    # Re-insert so the file-batch accumulator can see it.
                    # We reuse _pending_groups with a sentinel key to hold
                    # complete groups waiting for the file batch threshold.
                    if "_ready" not in self._pending_groups:
                        self._pending_groups["_ready"] = []
                    self._pending_groups["_ready"].append(grouped_record)

                    ready = self._pending_groups["_ready"]
                    if len(ready) >= self.batch_size:
                        batch_num = self.batch_counter
                        self.batch_counter += 1
                        grouped_records = self._pending_groups.pop("_ready")

                        batch_file = os.path.join(self.output_dir, f"batch_{batch_num:05d}.jsonl")
                        loop = asyncio.get_running_loop()
                        await loop.run_in_executor(None, write_batch_file, batch_file, grouped_records)
                        logger.info(
                            f"[Batch {batch_num}] Saved {len(grouped_records)} groups "
                            f"({self.num_completions_per_prompt} completions each) to {batch_file}"
                        )
                    else:
                        logger.info(
                            f"[Group {group_id}] Buffered for file batch "
                            f"({len(ready)}/{self.batch_size} groups ready)"
                        )
            else:
                logger.info(
                    f"[Traj {traj.id}] Added to group {group_id}. "
                    f"Pending completions: {group_size}/{self.num_completions_per_prompt}"
                )

    async def flush_pending(self):
        """
        Flush any remaining records (including incomplete groups).

        Queue mode:  partial groups are pushed into the BufferQueue as-is so
                     the trainer can still learn from them.
        File mode:   writes a final JSONL batch to disk.
        """
        async with self._batch_lock:
            if not self._pending_groups:
                return

            grouped_records = []

            # Collect the "_ready" file-batch list (file mode only)
            ready = self._pending_groups.pop("_ready", [])
            grouped_records.extend(ready)

            # Collect any partially-complete groups (all modes)
            for g_id, group_records in self._pending_groups.items():
                if group_records:
                    first = group_records[0]
                    grouped_records.append({
                        "group_id": g_id,
                        "prompt": first.get("prompt"),
                        "prompt_ids": first.get("prompt_ids"),
                        "completions": [r.get("completion") for r in group_records],
                        "metadata": {
                            "num_completions": len(group_records),
                            "timestamp": first.get("metadata", {}).get("timestamp"),
                        },
                    })
            self._pending_groups = {}

            if not grouped_records:
                return

            if self.buffer_queue is not None:
                for gr in grouped_records:
                    await self.buffer_queue.put(gr)
                logger.info(
                    f"[flush_pending] Pushed {len(grouped_records)} remaining groups "
                    f"into BufferQueue. {self.buffer_queue}"
                )
            else:
                batch_num = self.batch_counter
                self.batch_counter += 1
                batch_file = os.path.join(self.output_dir, f"batch_{batch_num:05d}.jsonl")
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, write_batch_file, batch_file, grouped_records)
                logger.info(
                    f"[Batch {batch_num}] Flushed {len(grouped_records)} remaining groups to {batch_file}"
                )

    # Per-request timeout for vLLM proxy calls (seconds).
    # Long enough for multi-tool trajectories; prevents workers hanging forever.
    _VLLM_TIMEOUT_S: int = 600  # 10 minutes

    async def gpu_worker(self, worker_id: int):
        """
        Consumes tasks from task_queue.
        Sends HTTP requests to vLLM Proxy.
        Streams tokens and checks for tool tags.
        """
        timeout = aiohttp.ClientTimeout(total=self._VLLM_TIMEOUT_S)
        async with aiohttp.ClientSession(timeout=timeout) as session:
            while True:
                traj = await self.task_queue.get()

                try:
                    logger.debug(f"[GPU-{worker_id}] Processing Traj {traj.id}")
                    
                    # Calculate input tokens to set dynamic max_tokens
                    # This prevents "max_tokens too large" errors as conversation grows
                    prompt_text = messages_to_prompt_string(traj.messages, self.tokenizer)
                    input_tokens = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                    available_tokens = self.max_model_len - input_tokens - 100  # 100 token buffer
                    max_tokens = max(self.min_completion_tokens, min(available_tokens, 16000))
                    
                    logger.debug(f"[GPU-{worker_id}] Input: {input_tokens} tokens, max_tokens: {max_tokens}")

                    # Prepare request payload
                    payload = {
                        "model": self.model,  # Use configured model (base or LoRA adapter name)
                        "messages": traj.messages,
                        "max_tokens": max_tokens,
                        "temperature": self.generation_temperature,  # GRPO: Use higher temp (1.0) for diverse completions
                        "stream": True,
                        "logprobs": 1, # Request logprobs
                        "stop": self.stop_tokens
                    }

                    # Add LoRA request if PolicyManager is configured (hot-swap support)
                    if self.policy_manager is not None:
                        lora_request = self.policy_manager.get_current_lora_request()
                        if lora_request is not None:
                            # vLLM uses extra_body for LoRA requests
                            payload["extra_body"] = {
                                "lora_request": lora_request.to_dict()
                            }
                            logger.debug(
                                f"[GPU-{worker_id}] Using LoRA: lora_int_id={lora_request.lora_int_id}"
                            )

                    buffer = ""
                    # Reset logprob accumulator for this turn (NOT completions - those accumulate)
                    traj.accumulated_logprobs = []
                    
                    tool_detected = False
                    
                    async with session.post(self.proxy_url, json=payload) as response:
                        if response.status != 200:
                            error_text = await response.text()
                            logger.error(f"[GPU-{worker_id}] Error from Proxy: {response.status} - {error_text[:200]}")
                            # Re-queue for retry or skip - for now skip
                            continue

                        # Stream handling
                        finish_reason = None
                        stop_reason = None
                        
                        async for line in response.content:
                            line = line.decode('utf-8').strip()
                            if not line or line == "data: [DONE]":
                                continue
                            if line.startswith("data: "):
                                try:
                                    # Parse vLLM chunk
                                    chunk = json.loads(line[6:])
                                    if "choices" in chunk and chunk["choices"]:
                                        choice = chunk["choices"][0]
                                        delta = choice.get("delta", {})
                                        content = delta.get("content", "")
                                        
                                        # Track finish/stop reason
                                        if choice.get("finish_reason"):
                                            finish_reason = choice["finish_reason"]
                                            stop_reason = choice.get("stop_reason")
                                        
                                        # Extract Logprobs from streaming response
                                        # vLLM format: {"logprobs": {"content": [{"token": "...", "logprob": -0.1, ...}]}}
                                        if "logprobs" in choice and choice["logprobs"]:
                                            lps = choice["logprobs"]
                                            if "content" in lps and lps["content"]:
                                                for token_info in lps["content"]:
                                                    if isinstance(token_info, dict):
                                                        # Extract logprob value
                                                        logprob = token_info.get("logprob", 0.0)
                                                        traj.accumulated_logprobs.append(logprob)
                                                        # Mark as model-generated token (include in loss)
                                                        traj.action_mask.append(1)
                                                    
                                        if content:
                                            buffer += content
                                        
                                        # Check for tool tags in the accumulated buffer
                                        # stream_parser returns dict if complete tag found: {'type': 'web', 'content': '...'}
                                        tool_call = stream_parser(buffer, allowed_tools=self.enabled_tools)
                                        
                                        if tool_call:
                                            logger.info(f"[GPU-{worker_id}] Tool Detected: {tool_call['type']}")
                                            
                                            # Accumulate this turn's content
                                            traj.completions.append(buffer)
                                            
                                            # Update Trajectory with the partial generation (Assistant content)
                                            tool_str = f"<{tool_call['type']}>{tool_call['content']}</{tool_call['type']}>"
                                            traj.messages.append({"role": "assistant", "content": tool_str})
                                            
                                            # Push to Tool Queue
                                            await self.tool_queue.put((traj, tool_call))
                                            
                                            tool_detected = True
                                            break # Stop streaming, free GPU

                                except json.JSONDecodeError:
                                    continue
                        
                        # After stream ends, check if we stopped on a tool closing tag
                        # vLLM stop_reason contains the actual stop string that triggered the stop
                        if not tool_detected and finish_reason == "stop" and stop_reason:
                            # Append the stop token to buffer if it's a tool tag
                            if stop_reason in self.tool_stop_tokens:
                                buffer += stop_reason
                                logger.debug(f"[GPU-{worker_id}] Appended stop_reason: {stop_reason}")
                                
                                # Re-check for tool after appending stop token
                                tool_call = stream_parser(buffer, allowed_tools=self.enabled_tools)
                                if tool_call:
                                    logger.info(f"[GPU-{worker_id}] Tool Detected (via stop_reason): {tool_call['type']}")
                                    
                                    # Accumulate this turn's content  
                                    traj.completions.append(buffer)
                                    
                                    tool_str = f"<{tool_call['type']}>{tool_call['content']}</{tool_call['type']}>"
                                    traj.messages.append({"role": "assistant", "content": tool_str})
                                    
                                    await self.tool_queue.put((traj, tool_call))
                                    tool_detected = True
                            elif stop_reason == "</solution>":
                                if stop_reason not in buffer:
                                    buffer += stop_reason
                                    logger.debug(f"[GPU-{worker_id}] Appended stop_reason: {stop_reason}")
                    
                    # If stream finished without tool, task is done (or solution found)
                    if not tool_detected:
                        # Accumulate final turn's content
                        traj.completions.append(buffer)
                        
                        # Append to message history
                        traj.messages.append({"role": "assistant", "content": buffer})
                        logger.info(f"[GPU-{worker_id}] Traj {traj.id} Completed.")
                        
                        # Save the completed trajectory
                        await self.save_trajectory(traj)
                        
                        # Mark as done in final logic
                        
                except asyncio.TimeoutError:
                    logger.error(
                        f"[GPU-{worker_id}] Request timed out after {self._VLLM_TIMEOUT_S}s "
                        f"for Traj {traj.id}. Dropping trajectory."
                    )
                except Exception as e:
                    logger.error(f"[GPU-{worker_id}] Exception processing Traj {traj.id}: {e}")
                finally:
                    self.task_queue.task_done()

    async def tool_worker(self, worker_id: int):
        """
        Consumes tasks from tool_queue.
        Executes external tools (I/O bound).
        Pushes result back to task_queue.
        """
        while True:
            item = await self.tool_queue.get()
            traj, tool_call = item
            
            try:
                tool_type = tool_call.get("type")
                content = tool_call.get("content")
                
                logger.info(f"[Tool-{worker_id}] Executing {tool_type} for Traj {traj.id}")
                
                # For now, running in executor to be safe and non-blocking
                loop = asyncio.get_running_loop()
                result = None

                # Execute Tool (Pseudo-async default to blocking call wrapped in thread if needed)
                if tool_type == "web":
                     result = await loop.run_in_executor(None, send_web_command, {"q": content})
                elif tool_type == "code":
                     result = await loop.run_in_executor(None, send_code_command, {"code_command": content})
                elif tool_type == "azure":
                     result = await loop.run_in_executor(None, send_azure_command, {"azure_command": content})
                else:
                    result = f"[Error] Unknown tool type: {tool_type}"

                # Format Result
                # Format Result
                prefix = "<tool_result>"
                suffix = "</tool_result>\n"
                result_str = f"{prefix}{result}{suffix}"
                
                # Tokenize tool result to get token count for action masking
                # These tokens should NOT contribute to loss (mask = 0)
                full_tokens = self.tokenizer(result_str, add_special_tokens=False)["input_ids"]
                num_tokens = len(full_tokens)
                
                # Calculate lengths of prefix and suffix tokens
                prefix_ids = self.tokenizer(prefix, add_special_tokens=False)["input_ids"]
                suffix_ids = self.tokenizer(suffix, add_special_tokens=False)["input_ids"]
                len_prefix = len(prefix_ids)
                len_suffix = len(suffix_ids)

                # mask: 1 = model-generated token (include in loss)
                #       0 = tool result content (exclude from loss)
                # We unmask the surrounding <tool_result> / </tool_result>\n tags so the
                # model learns to emit them, but mask the inner content (injected by the tool).
                mask = [0] * num_tokens

                if num_tokens >= len_prefix + len_suffix:
                    # Unmask <tool_result> prefix
                    for i in range(len_prefix):
                        mask[i] = 1
                    # Unmask </tool_result>\n suffix
                    for i in range(len_suffix):
                        mask[num_tokens - len_suffix + i] = 1
                # else: result shorter than tags (should never happen); leave all masked (0).

                # Append mask and placeholder logprobs
                traj.action_mask.extend(mask)
                traj.accumulated_logprobs.extend([0.0] * num_tokens)
                
                # Update Trajectory
                traj.messages.append({"role": "tool", "content": result_str})
                
                # Re-queue for GPU Processing (Next Turn)
                logger.info(f"[Tool-{worker_id}] Finished {tool_type}. Added {num_tokens} masked tokens. Re-queueing Traj {traj.id}")
                await self.task_queue.put(traj)
                
            except Exception as e:
                logger.error(f"[Tool-{worker_id}] Exception: {e}")
            finally:
                self.tool_queue.task_done()

# Utility to run standalone
if __name__ == "__main__":
    # Example usage
    async def main():
        orchestrator = AsyncBatchOrchestrator(proxy_url="http://localhost:10001/v1/chat/completions")
        await orchestrator.start()
        
        # Test Trajectory
        test_traj = Trajectory(id="test-1", messages=[{"role": "user", "content": "Search for the latest news on AI."}], completions=[])
        await orchestrator.add_trajectory(test_traj)
        
        # Keep running
        while True:
            await asyncio.sleep(1)

    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
