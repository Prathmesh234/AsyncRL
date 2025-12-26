import asyncio
import json
import logging
import time
import os
import sys
import aiohttp
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field

# Add parent directory to path to allow imports from serving root
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Import tool senders
from communication.command_sender import send_web_command
from communication.code_command_sender import send_code_command
from communication.azure_command_sender import send_azure_command
from parser import stream_parser

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
    status: str = "QUEUED"
    created_at: float = field(default_factory=time.time)

class AsyncBatchOrchestrator:
    def __init__(self, proxy_url: str, model: str = "Qwen/Qwen3-4B-Thinking-2507", tokenizer_name: str = "Qwen/Qwen3-4B-Thinking-2507", num_gpu_workers: int = 4, num_tool_workers: int = 32, output_dir: str = None, batch_size: int = 10):
        self.proxy_url = proxy_url
        self.model = model  # Can be base model or LoRA adapter name
        self.task_queue = asyncio.Queue()  # For GPU tasks
        self.tool_queue = asyncio.Queue()  # For Tool execution tasks
        
        # Output configuration - save as batch files to DisTrainer
        if output_dir is None:
            # Default to DisTrainer's data/generations folder
            output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../DisTrainer/data/generations"))
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)
        self.batch_counter = self._get_next_batch_number()
        self._batch_lock = asyncio.Lock()
        
        # Batch accumulation - collect N trajectories before writing
        self.batch_size = batch_size
        self._pending_records: List[Dict] = []
        
        self.num_gpu_workers = num_gpu_workers
        self.num_tool_workers = num_tool_workers
        self.workers: List[asyncio.Task] = []
        
        # Initialize tokenizer for proper token ID extraction
        logger.info(f"Loading tokenizer: {tokenizer_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name, trust_remote_code=True)
        logger.info(f"Tokenizer loaded. Vocab size: {len(self.tokenizer)}")
        
        # Stop strings for vLLM to pause generation immediately on tool call
        self.stop_tokens = ["</web>", "</code>", "</azure>", "<solution>"]
        
        # Context window configuration
        self.max_model_len = 65536  # Model's max context length (64K)
        self.min_completion_tokens = 256  # Minimum tokens to reserve for completion
    
    def _get_next_batch_number(self) -> int:
        """Find the next available batch number by checking existing files."""
        import glob
        pattern = os.path.join(self.output_dir, "batch_*.jsonl")
        existing = glob.glob(pattern)
        if not existing:
            return 1
        # Extract numbers and find max
        numbers = []
        for f in existing:
            try:
                num = int(os.path.basename(f).replace("batch_", "").replace(".jsonl", ""))
                numbers.append(num)
            except ValueError:
                continue
        return max(numbers) + 1 if numbers else 1

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
        """Entry point: Add a new trajectory to the system."""
        await self.task_queue.put(traj)
        logger.info(f"[Traj {traj.id}] Added to Task Queue")

    def _messages_to_prompt_string(self, messages: List[Dict[str, str]]) -> str:
        """
        Convert messages list to a plain prompt string.
        Uses the chat template if available, otherwise concatenates content.
        """
        try:
            # Use chat template for proper formatting (Qwen uses this)
            prompt_str = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            return prompt_str
        except Exception:
            # Fallback: simple concatenation
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"{role}: {content}")
            return "\n".join(parts)

    async def save_trajectory(self, traj: Trajectory):
        """
        Saves the completed trajectory to a JSONL file in the format expected by DisTrainer.
        Format:
        {
            "gen_id": str,
            "prompt": str,
            "prompt_ids": List[int],
            "completions": [
                {
                    "text": str,
                    "completion_ids": List[int],
                    "old_logprobs": List[float],
                    "reward": float (placeholder)
                }
            ],
            ...
        }
        """
        # Find original prompt (system + user messages before any assistant response)
        prompt_messages = []
        for msg in traj.messages:
            role = msg.get("role", "")
            if role in ["system", "user"]:
                prompt_messages.append(msg)
            else:
                break  # Stop at first non-prompt message
        
        # Convert prompt to string using chat template
        prompt_str = self._messages_to_prompt_string(prompt_messages)
        
        # Build full completion by interleaving accumulated completions with tool results
        # traj.completions contains assistant outputs from each turn
        # traj.messages contains tool results after each assistant message
        completion_parts = []
        completion_idx = 0
        
        for msg in traj.messages:
            role = msg.get("role", "")
            if role in ["system", "user"]:
                continue  # Skip prompt messages
            elif role == "assistant":
                # Use accumulated completion content
                if completion_idx < len(traj.completions):
                    completion_parts.append(traj.completions[completion_idx])
                    completion_idx += 1
            elif role == "tool":
                # Tool result - add it to completion
                completion_parts.append(msg.get("content", ""))
        
        full_completion = "\n".join(completion_parts)
        
        # Tokenize prompt to get prompt_ids
        prompt_encoding = self.tokenizer(prompt_str, add_special_tokens=False, return_tensors=None)
        prompt_ids = prompt_encoding["input_ids"]
        
        # Tokenize completion to get completion_ids
        completion_encoding = self.tokenizer(full_completion, add_special_tokens=False, return_tensors=None)
        completion_ids = completion_encoding["input_ids"]
        
        # Get logprobs - use accumulated if available, otherwise create placeholder
        # Note: streaming logprobs from vLLM may not align perfectly with tokenizer output
        # We use the accumulated logprobs and pad/truncate to match completion_ids length
        old_logprobs = traj.accumulated_logprobs.copy()
        
        # Ensure logprobs length matches completion_ids
        if len(old_logprobs) < len(completion_ids):
            # Pad with placeholder values
            old_logprobs.extend([-1.0] * (len(completion_ids) - len(old_logprobs)))
        elif len(old_logprobs) > len(completion_ids):
            # Truncate to match
            old_logprobs = old_logprobs[:len(completion_ids)]
        
        record = {
            "gen_id": traj.id,
            "prompt": prompt_str,
            "prompt_ids": prompt_ids,
            "completions": [{
                "text": full_completion,
                "completion_ids": completion_ids,
                "old_logprobs": old_logprobs,
                "reward": 0.0  # Placeholder for reward model
            }],
            "metadata": {
                "timestamp": time.time(),
                "status": "COMPLETED",
                "num_turns": len(traj.completions)
            }
        }
        # Accumulate record in pending batch
        async with self._batch_lock:
            self._pending_records.append(record)
            
            # When we have batch_size records, write them to a file
            if len(self._pending_records) >= self.batch_size:
                batch_num = self.batch_counter
                self.batch_counter += 1
                records_to_write = self._pending_records.copy()
                self._pending_records = []
                
                batch_file = os.path.join(self.output_dir, f"batch_{batch_num:05d}.jsonl")
                
                # Write batch asynchronously
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._write_batch_file, batch_file, records_to_write)
                logger.info(f"[Batch {batch_num}] Saved {len(records_to_write)} trajectories to {batch_file}")
            else:
                logger.info(f"[Traj {traj.id}] Added to pending batch ({len(self._pending_records)}/{self.batch_size})")

    async def flush_pending(self):
        """Flush any remaining records to a final batch file."""
        async with self._batch_lock:
            if self._pending_records:
                batch_num = self.batch_counter
                self.batch_counter += 1
                records_to_write = self._pending_records.copy()
                self._pending_records = []
                
                batch_file = os.path.join(self.output_dir, f"batch_{batch_num:05d}.jsonl")
                
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, self._write_batch_file, batch_file, records_to_write)
                logger.info(f"[Batch {batch_num}] Flushed {len(records_to_write)} remaining trajectories to {batch_file}")

    def _write_batch_file(self, filepath: str, records: List[Dict]):
        """Write multiple trajectories to a batch file (one per line)."""
        with open(filepath, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")

    async def gpu_worker(self, worker_id: int):
        """
        Consumes tasks from task_queue.
        Sends HTTP requests to vLLM Proxy.
        Streams tokens and checks for tool tags.
        """
        async with aiohttp.ClientSession() as session:
            while True:
                traj = await self.task_queue.get()
                
                try:
                    logger.debug(f"[GPU-{worker_id}] Processing Traj {traj.id}")
                    
                    # Calculate input tokens to set dynamic max_tokens
                    # This prevents "max_tokens too large" errors as conversation grows
                    prompt_text = self._messages_to_prompt_string(traj.messages)
                    input_tokens = len(self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"])
                    available_tokens = self.max_model_len - input_tokens - 100  # 100 token buffer
                    max_tokens = max(self.min_completion_tokens, min(available_tokens, 16000))
                    
                    logger.debug(f"[GPU-{worker_id}] Input: {input_tokens} tokens, max_tokens: {max_tokens}")
                    
                    # Prepare request payload
                    payload = {
                        "model": self.model,  # Use configured model (base or LoRA adapter name)
                        "messages": traj.messages,
                        "max_tokens": max_tokens,
                        "temperature": 0.7,
                        "stream": True,
                        "logprobs": 1, # Request logprobs
                        "stop": self.stop_tokens
                    }

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
                                                    
                                        if content:
                                            buffer += content
                                        
                                        # Check for tool tags in the accumulated buffer
                                        # stream_parser returns dict if complete tag found: {'type': 'web', 'content': '...'}
                                        tool_call = stream_parser(buffer)
                                        
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
                            if stop_reason in ["</web>", "</code>", "</azure>"]:
                                buffer += stop_reason
                                logger.debug(f"[GPU-{worker_id}] Appended stop_reason: {stop_reason}")
                                
                                # Re-check for tool after appending stop token
                                tool_call = stream_parser(buffer)
                                if tool_call:
                                    logger.info(f"[GPU-{worker_id}] Tool Detected (via stop_reason): {tool_call['type']}")
                                    
                                    # Accumulate this turn's content  
                                    traj.completions.append(buffer)
                                    
                                    tool_str = f"<{tool_call['type']}>{tool_call['content']}</{tool_call['type']}>"
                                    traj.messages.append({"role": "assistant", "content": tool_str})
                                    
                                    await self.tool_queue.put((traj, tool_call))
                                    tool_detected = True
                    
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
                        
                except Exception as e:
                    logger.error(f"[GPU-{worker_id}] Exception: {e}")
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
                # DUMMY IMPLEMENTATION FOR TESTING
                # Commenting out actual execution logic
                # if tool_type == "web":
                #      result = await loop.run_in_executor(None, send_web_command, {"q": content})
                # elif tool_type == "code":
                #      result = await loop.run_in_executor(None, send_code_command, {"code_command": content})
                # elif tool_type == "azure":
                #      result = await loop.run_in_executor(None, send_azure_command, {"azure_command": content})
                # else:
                #     result = f"[Error] Unknown tool type: {tool_type}"

                # Mock Response
                await asyncio.sleep(0.5) # Simulate latency
                if tool_type == "web":
                    result = f"Mock Web Search Result for: {content}"
                elif tool_type == "code":
                    result = f"Mock Code Execution Result for: {content}\nOutput: Success"
                elif tool_type == "azure":
                    result = f"Mock Azure Command Result for: {content}\nStatus: OK"
                else:
                    result = f"Mock Unknown Tool: {content}"

                # Format Result
                result_str = f"<tool_result>{result}</tool_result>\n"
                
                # Update Trajectory
                traj.messages.append({"role": "tool", "content": result_str})
                
                # Re-queue for GPU Processing (Next Turn)
                logger.info(f"[Tool-{worker_id}] Finished {tool_type}. Re-queueing Traj {traj.id}")
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
