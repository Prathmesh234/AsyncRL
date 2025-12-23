"""
DisGenerator: Distributed Trajectory Generator

Main async orchestrator for generating trajectories with tool calling support.
Uses disaggregated prefill-decode architecture for high throughput.

Architecture:
- Prefill Server (GPUs 0,1): Processes prompts and conversation history
- Decode Server (GPUs 2,3): Token-by-token generation

Flow per completion:
1. Send to Prefill server to warm KV cache
2. Stream from Decode server until stop token
3. If tool tag detected, execute tool and resume
4. Collect logprobs throughout
5. Send to grader for reward scoring
"""

import asyncio
import json
import logging
import os
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

import aiohttp
from transformers import AutoTokenizer

# Add parent directory for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
if SERVING_DIR not in sys.path:
    sys.path.insert(0, SERVING_DIR)

from config import config, DisGeneratorConfig
from tool_executor import AsyncToolExecutor

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class CompletionResult:
    """Result of a single completion generation."""
    text: str
    completion_ids: List[int] = field(default_factory=list)
    logprobs: List[float] = field(default_factory=list)
    reward: float = 0.0
    num_turns: int = 0
    tool_calls: List[Dict] = field(default_factory=list)
    error: Optional[str] = None


@dataclass
class TrajectoryBatch:
    """A batch of trajectories for a single prompt."""
    prompt: str
    prompt_ids: List[int] = field(default_factory=list)
    completions: List[CompletionResult] = field(default_factory=list)
    metadata: Dict = field(default_factory=dict)


class SSEParser:
    """Parse Server-Sent Events from vLLM streaming responses."""
    
    @staticmethod
    def parse_line(line: str) -> Optional[Dict]:
        """Parse a single SSE line."""
        line = line.strip()
        if not line or line.startswith(':'):
            return None
        if line.startswith('data: '):
            data = line[6:]
            if data == '[DONE]':
                return {'done': True}
            try:
                return json.loads(data)
            except json.JSONDecodeError:
                return None
        return None
    
    @staticmethod
    def extract_content(chunk: Dict) -> Tuple[str, Optional[List[float]], Optional[List[int]]]:
        """Extract content, logprobs, and token IDs from a chunk."""
        content = ""
        logprobs = None
        token_ids = None
        
        if 'choices' in chunk and chunk['choices']:
            choice = chunk['choices'][0]
            delta = choice.get('delta', {})
            content = delta.get('content', '')
            
            # Extract logprobs if available
            if 'logprobs' in choice and choice['logprobs']:
                lp_content = choice['logprobs'].get('content', [])
                if lp_content:
                    logprobs = [item.get('logprob', 0.0) for item in lp_content]
                    token_ids = [item.get('token_id') for item in lp_content if 'token_id' in item]
        
        return content, logprobs, token_ids


class ToolParser:
    """Parse and extract tool calls from generated text."""
    
    TOOL_PATTERNS = {
        'web': re.compile(r'<web>\s*(\{[^}]+\})\s*</web>', re.DOTALL),
        'code': re.compile(r'<code>\s*(\{[^}]+\})\s*</code>', re.DOTALL),
        'azure': re.compile(r'<azure>\s*(\{[^}]+\})\s*</azure>', re.DOTALL),
    }
    
    @staticmethod
    def has_complete_tool_call(text: str) -> Optional[Tuple[str, str, Dict]]:
        """
        Check if text contains a complete tool call.
        
        Returns (tool_type, raw_match, parsed_payload) or None.
        """
        for tool_type, pattern in ToolParser.TOOL_PATTERNS.items():
            match = pattern.search(text)
            if match:
                try:
                    payload = json.loads(match.group(1))
                    return (tool_type, match.group(0), payload)
                except json.JSONDecodeError:
                    # Malformed JSON, still return for logging
                    return (tool_type, match.group(0), {})
        return None
    
    @staticmethod
    def has_solution(text: str) -> bool:
        """Check if text contains solution tag."""
        return '<solution>' in text


class DisGenerator:
    """
    Main trajectory generator using disaggregated serving.
    
    Generates multiple completions per prompt with tool calling support,
    collecting logprobs and rewards for GRPO training.
    """
    
    def __init__(self, cfg: Optional[DisGeneratorConfig] = None):
        self.config = cfg or config
        self.tool_executor = AsyncToolExecutor(
            service_bus_connection_string=self.config.service_bus_connection_string,
            command_topic_name=self.config.command_topic_name,
            reward_topic_name=self.config.reward_topic_name,
            web_subscription_name=self.config.web_subscription_name,
            code_subscription_name=self.config.code_subscription_name,
            grader_subscription_name=self.config.grader_subscription_name,
            grader_reward_subscription_name=self.config.grader_reward_subscription_name,
        )
        self.sse_parser = SSEParser()
        self.tool_parser = ToolParser()
        
        # Initialize tokenizer for prompt_ids
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        except Exception as e:
            logger.warning(f"Failed to load tokenizer from {self.config.model_name}: {e}. Local folder may be needed.")
            self.tokenizer = None
        
        # Session will be created per-batch
        self._session: Optional[aiohttp.ClientSession] = None
    
    @property
    def model_name_for_api(self) -> str:
        """
        Get the model name to use in API requests.
        
        If LoRA adapter is configured, use the adapter name.
        Otherwise, use the base model name.
        """
        if self.config.lora_adapter_path and self.config.lora_adapter_name:
            return self.config.lora_adapter_name
        return self.config.model_name
    
    async def __aenter__(self):
        """Async context manager entry."""
        self._session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.request_timeout_s)
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit."""
        if self._session:
            await self._session.close()
    
    async def generate_batch(self, prompts: List[str]) -> List[TrajectoryBatch]:
        """
        Generate trajectories for a batch of prompts.
        
        Args:
            prompts: List of user prompts
            
        Returns:
            List of TrajectoryBatch objects
        """
        logger.info(f"Generating batch of {len(prompts)} prompts "
                   f"({self.config.completions_per_prompt} completions each)")
        
        # Generate all trajectories in parallel
        tasks = []
        for prompt in prompts:
            for completion_idx in range(self.config.completions_per_prompt):
                tasks.append(self._generate_single_trajectory(prompt, completion_idx))
        
        # Run all tasks
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Group results by prompt
        batches = []
        result_idx = 0
        for prompt in prompts:
            batch = TrajectoryBatch(
                prompt=prompt,
                prompt_ids=self.tokenizer.encode(prompt) if self.tokenizer else [],
                metadata={
                    "timestamp": datetime.utcnow().isoformat(),
                    "base_model": self.config.model_name,
                    "lora_adapter": self.config.lora_adapter_name if self.config.lora_adapter_path else None,
                    "model": self.model_name_for_api,
                    "completions_requested": self.config.completions_per_prompt,
                }
            )
            
            for _ in range(self.config.completions_per_prompt):
                result = results[result_idx]
                if isinstance(result, Exception):
                    logger.error(f"Completion failed: {result}")
                    batch.completions.append(CompletionResult(
                        text="",
                        error=str(result)
                    ))
                else:
                    batch.completions.append(result)
                result_idx += 1
            
            batches.append(batch)
        
        return batches
    
    async def _generate_single_trajectory(
        self,
        prompt: str,
        completion_idx: int
    ) -> CompletionResult:
        """
        Generate a single trajectory with tool calling support.
        
        Uses the disaggregated prefill-decode pattern:
        1. Prefill server warms the KV cache
        2. Decode server generates tokens
        3. Tool interception and execution
        4. Resume generation after tool result
        """
        messages = [
            {"role": "system", "content": self.config.system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        full_text = ""
        all_logprobs = []
        all_token_ids = []
        tool_calls = []
        
        for turn in range(self.config.max_turns):
            logger.debug(f"Turn {turn + 1} for completion {completion_idx}")
            
            # Step 1: Prefill (warm KV cache)
            try:
                await self._prefill_warmup(messages)
            except Exception as e:
                logger.warning(f"Prefill warmup failed (continuing): {e}")
            
            # Step 2: Decode (stream until stop token)
            try:
                text, logprobs, token_ids, stop_reason = await self._decode_stream(messages)
            except Exception as e:
                logger.error(f"Decode failed: {e}")
                return CompletionResult(
                    text=full_text,
                    logprobs=all_logprobs,
                    completion_ids=all_token_ids,
                    tool_calls=tool_calls,
                    num_turns=turn + 1,
                    error=str(e)
                )
            
            full_text += text
            all_logprobs.extend(logprobs)
            all_token_ids.extend(token_ids)
            
            # Check for solution tag
            if self.tool_parser.has_solution(text):
                logger.debug(f"Solution found at turn {turn + 1}")
                break
            
            # Check for tool call
            tool_call = self.tool_parser.has_complete_tool_call(text)
            if tool_call:
                tool_type, raw_match, payload = tool_call
                logger.info(f"Tool call detected: {tool_type}")
                
                # Execute tool
                result = await self.tool_executor.execute_tool(
                    tool_type,
                    payload,
                    self.config.tool_timeout_s
                )
                
                tool_calls.append({
                    "type": tool_type,
                    "payload": payload,
                    "result": result,
                    "turn": turn + 1
                })
                
                # Append tool result to text and messages
                tool_result = f"<tool_result>{result}</tool_result>"
                full_text += tool_result
                
                # Update messages for next turn
                messages.append({"role": "assistant", "content": text + tool_result})
            else:
                # No tool call, add as-is to messages
                messages.append({"role": "assistant", "content": text})
                
                # If no tool call and no solution, might be stuck
                if stop_reason == "length":
                    logger.debug(f"Hit max tokens at turn {turn + 1}")
                    break
        
        # Get completion text (excluding prompt/system)
        completion_text = full_text
        
        # Grade the completion
        reward = await self.tool_executor.grade_completion(
            prompt,
            completion_text,
            self.config.grader_timeout_s
        )
        
        return CompletionResult(
            text=completion_text,
            completion_ids=all_token_ids,
            logprobs=all_logprobs,
            reward=reward,
            num_turns=len(tool_calls) + 1,
            tool_calls=tool_calls
        )
    
    async def _prefill_warmup(self, messages: List[Dict]) -> None:
        """Send request to prefill server to warm up KV cache."""
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Use LoRA adapter name if configured, otherwise base model
        payload = {
            "model": self.model_name_for_api,
            "messages": messages,
            "max_tokens": 1,  # Just warm the cache
            "stream": False
        }
        
        async with self._session.post(
            self.config.prefill_server.completions_url,
            json=payload
        ) as response:
            if response.status != 200:
                text = await response.text()
                raise RuntimeError(f"Prefill failed: {response.status} - {text}")
    
    async def _decode_stream(
        self,
        messages: List[Dict]
    ) -> Tuple[str, List[float], List[int], str]:
        """
        Stream tokens from decode server until stop token.
        
        Returns:
            (generated_text, logprobs, token_ids, stop_reason)
        """
        if not self._session:
            raise RuntimeError("Session not initialized. Use async context manager.")
        
        # Use LoRA adapter name if configured, otherwise base model
        payload = {
            "model": self.model_name_for_api,
            "messages": messages,
            "max_tokens": self.config.max_tokens_per_completion,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "stream": True,
            "logprobs": True,
            "top_logprobs": 1,
            "stop": self.config.stop_tokens
        }
        
        text = ""
        logprobs = []
        token_ids = []
        stop_reason = "stop"
        
        async with self._session.post(
            self.config.decode_server.completions_url,
            json=payload
        ) as response:
            if response.status != 200:
                error_text = await response.text()
                raise RuntimeError(f"Decode failed: {response.status} - {error_text}")
            
            async for line in response.content:
                line_str = line.decode('utf-8').strip()
                if not line_str:
                    continue
                
                chunk = self.sse_parser.parse_line(line_str)
                if chunk is None:
                    continue
                
                if chunk.get('done'):
                    break
                
                content, chunk_logprobs, chunk_token_ids = self.sse_parser.extract_content(chunk)
                text += content
                
                if chunk_logprobs:
                    logprobs.extend(chunk_logprobs)
                if chunk_token_ids:
                    token_ids.extend(chunk_token_ids)
                
                # Check for tool call interception
                if any(tag in text for tag in self.config.intercept_tags):
                    stop_reason = "tool_call"
                    break
                
                # Check finish reason
                if 'choices' in chunk and chunk['choices']:
                    finish_reason = chunk['choices'][0].get('finish_reason')
                    if finish_reason:
                        stop_reason = finish_reason
        
        return text, logprobs, token_ids, stop_reason
    
    async def generate_and_save(
        self,
        prompts: List[str],
        output_path: str
    ) -> None:
        """
        Generate trajectories and save to JSONL file.
        
        Args:
            prompts: List of prompts to process
            output_path: Path to output JSONL file
        """
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        
        batches = await self.generate_batch(prompts)
        
        with open(output_path, 'w') as f:
            for batch in batches:
                record = self._batch_to_record(batch)
                f.write(json.dumps(record, ensure_ascii=False) + '\n')
        
        logger.info(f"Saved {len(batches)} trajectories to {output_path}")
    
    def _batch_to_record(self, batch: TrajectoryBatch) -> Dict:
        """Convert batch to JSONL record format for DisTrainer."""
        return {
            "prompt": batch.prompt,
            "prompt_ids": batch.prompt_ids,
            "completions": [
                {
                    "text": c.text,
                    "completion_ids": c.completion_ids,
                    "old_logprobs": c.logprobs,
                    "reward": c.reward,
                    "num_turns": c.num_turns,
                    "tool_calls": c.tool_calls,
                    "error": c.error
                }
                for c in batch.completions
            ],
            "metadata": batch.metadata
        }


async def main():
    """Example usage of DisGenerator."""
    import argparse
    
    parser = argparse.ArgumentParser(description="DisGenerator - Trajectory Generator")
    parser.add_argument("--prompts-file", type=str, help="Path to file with prompts (one per line)")
    parser.add_argument("--output", type=str, default="./data/generations/batch.jsonl",
                       help="Output JSONL file path")
    parser.add_argument("--num-prompts", type=int, default=10,
                       help="Number of prompts to process (if not using file)")
    args = parser.parse_args()
    
    # Load prompts
    if args.prompts_file and os.path.exists(args.prompts_file):
        with open(args.prompts_file) as f:
            prompts = [line.strip() for line in f if line.strip()]
    else:
        # Default test prompts
        prompts = [
            "What is the capital of France?",
            "Write a Python function to calculate factorial.",
            "How do I deploy an Azure Function?",
        ][:args.num_prompts]
    
    logger.info(f"Processing {len(prompts)} prompts")
    
    async with DisGenerator() as generator:
        await generator.generate_and_save(prompts, args.output)


if __name__ == "__main__":
    asyncio.run(main())
