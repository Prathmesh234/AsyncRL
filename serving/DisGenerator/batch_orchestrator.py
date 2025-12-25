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

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
logger = logging.getLogger("Orchestrator")

@dataclass
class Trajectory:
    id: str
    messages: List[Dict[str, str]]
    completions: List[str] = field(default_factory=list)
    # Accumulators for the current turn/generation
    accumulated_ids: List[int] = field(default_factory=list)
    accumulated_logprobs: List[float] = field(default_factory=list)
    status: str = "QUEUED"
    created_at: float = field(default_factory=time.time)

class AsyncBatchOrchestrator:
    def __init__(self, proxy_url: str, model: str = "Qwen/Qwen3-4B-Thinking-2507", num_gpu_workers: int = 4, num_tool_workers: int = 32, output_file: str = "completed_trajectories.jsonl"):
        self.proxy_url = proxy_url
        self.model = model  # Can be base model or LoRA adapter name
        self.task_queue = asyncio.Queue()  # For GPU tasks
        self.tool_queue = asyncio.Queue()  # For Tool execution tasks
        self.output_file = output_file
        
        self.num_gpu_workers = num_gpu_workers
        self.num_tool_workers = num_tool_workers
        self.workers: List[asyncio.Task] = []
        
        # Stop strings for vLLM to pause generation immediately on tool call
        self.stop_tokens = ["</web>", "</code>", "</azure>", "<solution>"]

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

    async def save_trajectory(self, traj: Trajectory, final_text: str):
        """
        Saves the completed trajectory to a JSONL file in the format expected by DisTrainer.
        Format:
        {
            "gen_id": str,
            "prompt": str (or messages),
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
        # Construct the record
        # Note: In a real RLHF loop, 'prompt' would be the initial instruction.
        # Here we dump the full conversation state or just the last turn depending on training needs.
        # For simplicity, we'll assume standard SFT/RL format where we save the response.
        
        record = {
            "gen_id": traj.id,
            "prompt": json.dumps(traj.messages[:-1]), # All messages prior to the last assistant response
            "prompt_ids": [], # We don't have prompt IDs from valid streaming response easily without tokenizing locally
            "completions": [{
                "text": final_text,
                "completion_ids": traj.accumulated_ids,
                "old_logprobs": traj.accumulated_logprobs,
                "reward": 0.0 # Placeholder for reward model
            }],
            "metadata": {
                "timestamp": time.time(),
                "status": "COMPLETED"
            }
        }
        
        # Append to file asynchronously (using thread executor to avoid blocking event loop on file I/O)
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, self._append_to_file, record)
        logger.info(f"[Traj {traj.id}] Saved to {self.output_file}")

    def _append_to_file(self, record: Dict):
        with open(self.output_file, "a") as f:
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
                    
                    # Prepare request payload
                    payload = {
                        "model": self.model,  # Use configured model (base or LoRA adapter name)
                        "messages": traj.messages,
                        "max_tokens": 32768,
                        "temperature": 0.7,
                        "stream": True,
                        "logprobs": 1, # Request logprobs
                        "stop": self.stop_tokens
                    }

                    buffer = ""
                    # Reset accumulators for this turn
                    traj.accumulated_ids = []
                    traj.accumulated_logprobs = []
                    
                    tool_detected = False
                    
                    async with session.post(self.proxy_url, json=payload) as response:
                        if response.status != 200:
                            logger.error(f"[GPU-{worker_id}] Error from Proxy: {response.status}")
                            # Let finally handle task_done(). Just continue to next task.
                            continue

                        # Stream handling
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
                                        delta = choice["delta"].get("content", "")
                                        
                                        # Extract Logprobs
                                        if "logprobs" in choice and choice["logprobs"]:
                                            # vLLM streaming logprobs format:
                                            # "logprobs": {"content": [{"token": "...", "logprob": -0.1, "bytes": [...]}]} 
                                            # OR standard OpenAI format depending on version.
                                            # Let's assume standard OpenAI for now, but vLLM might nest it in 'content'.
                                            # Checking structure is safer.
                                            lps = choice["logprobs"]
                                            if "content" in lps: # vLLM specific
                                                for token_info in lps["content"]:
                                                    # Depending on vLLM version, token_info might be dict
                                                    if isinstance(token_info, dict):
                                                        # Note: Using a dummy ID if not provided, streaming usually doesn't send IDs unless requested
                                                        # We might need to rely on 'text' if IDs aren't there.
                                                        # But 'logprobs' usually implies we get prob data.
                                                        # Let's try to get what we can.
                                                        traj.accumulated_logprobs.append(token_info.get("logprob", 0.0))
                                                        # Token ID might not be directly available in standard OpenAI stream without 'echo'
                                                        # but some vLLM versions send it.
                                                        # If unavailable, we might need a tokenizer locally.
                                                        # For now, append 0 or try to find ID.
                                                        traj.accumulated_ids.append(token_info.get("token_id", 0)) 
                                                    
                                        if delta:
                                            buffer += delta
                                        
                                    # Check for tool tags in the accumulated buffer
                                    # stream_parser returns dict if complete tag found: {'type': 'web', 'content': '...'}
                                    tool_call = stream_parser(buffer)
                                    
                                    if tool_call:
                                        logger.info(f"[GPU-{worker_id}] Tool Detected: {tool_call['type']}")
                                        
                                        # 1. Update Trajectory with the partial generation (Assistant content)
                                        # Note: buffer might contain closing tag which we want to keep
                                        tool_str = f"<{tool_call['type']}>{tool_call['content']}</{tool_call['type']}>"
                                        traj.messages.append({"role": "assistant", "content": tool_str})
                                        
                                        # 2. Push to Tool Queue
                                        await self.tool_queue.put((traj, tool_call))
                                        
                                        tool_detected = True
                                        break # Stop streaming, free GPU

                                except json.JSONDecodeError:
                                    continue
                    
                    # If stream finished without tool, task is done (or solution found)
                    if not tool_detected:
                        # Append final content
                        traj.messages.append({"role": "assistant", "content": buffer})
                        logger.info(f"[GPU-{worker_id}] Traj {traj.id} Completed.")
                        
                        # Save the completed trajectory
                        await self.save_trajectory(traj, buffer)
                        
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
