"""
Tool Executor for DisGenerator

Async wrapper around existing communication modules for tool execution.
Supports Web, Code, Azure, and Grader tools via Azure Service Bus.
"""

import asyncio
import json
import logging
import sys
import os
from typing import Any, Dict, Optional
from uuid import uuid4

# Add parent directories to path for imports
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
SERVING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, '..'))
COMMUNICATION_DIR = os.path.join(SERVING_DIR, 'communication')

if SERVING_DIR not in sys.path:
    sys.path.insert(0, SERVING_DIR)
if COMMUNICATION_DIR not in sys.path:
    sys.path.insert(0, COMMUNICATION_DIR)

from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Import Service Bus library
try:
    from servicebus_web import ServiceBusTopic
except ImportError:
    logger.warning("servicebus_web not available. Using mock implementation.")
    ServiceBusTopic = None


class AsyncToolExecutor:
    """
    Async tool executor that communicates via Azure Service Bus.
    
    Provides async versions of the tool command senders for use in
    high-concurrency trajectory generation.
    """
    
    def __init__(
        self,
        service_bus_connection_string: Optional[str] = None,
        command_topic_name: str = "commandtopic",
        reward_topic_name: str = "rewardtopic",
        web_subscription_name: str = "websubscription",
        code_subscription_name: str = "codesubscription",
        grader_subscription_name: str = "gradersubscription",
        grader_reward_subscription_name: str = "webrewardsubscription",
        default_timeout_s: int = 30,
    ):
        self.connection_string = service_bus_connection_string or os.getenv("SERVICE_BUS_CONNECTION_STRING")
        self.command_topic = command_topic_name
        self.reward_topic = reward_topic_name
        self.web_subscription = web_subscription_name
        self.code_subscription = code_subscription_name
        self.grader_subscription = grader_subscription_name
        self.grader_reward_subscription = grader_reward_subscription_name
        self.default_timeout = default_timeout_s
        
        if not self.connection_string:
            logger.warning("No SERVICE_BUS_CONNECTION_STRING. Tools will return mock responses.")
    
    async def execute_tool(self, tool_type: str, payload: Dict[str, Any], timeout_s: Optional[int] = None) -> str:
        """
        Execute a tool call asynchronously.
        
        Args:
            tool_type: One of "web", "code", "azure"
            payload: Tool-specific payload dict
            timeout_s: Optional timeout override
            
        Returns:
            Tool result string
        """
        timeout = timeout_s or self.default_timeout
        
        if tool_type == "web":
            return await self._execute_web(payload, timeout)
        elif tool_type == "code":
            return await self._execute_code(payload, timeout)
        elif tool_type == "azure":
            return await self._execute_azure(payload, timeout)
        else:
            return f"[error] Unknown tool type: {tool_type}"
    
    async def grade_completion(self, query: str, completion: str, timeout_s: Optional[int] = None) -> float:
        """
        Send completion to grader and get numeric reward score.
        
        Args:
            query: The original prompt
            completion: The model's completion
            timeout_s: Optional timeout override
            
        Returns:
            Reward score (0.0 to 1.0)
        """
        timeout = timeout_s or self.default_timeout
        
        if not self.connection_string or ServiceBusTopic is None:
            logger.warning("Grader unavailable. Returning mock score.")
            return 0.5
        
        message_id = str(uuid4())
        message = {
            "type": "grader_command",
            "query": query,
            "completion": completion
        }
        
        try:
            # Send grader command
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_to_topic,
                message,
                message_id,
                self.grader_subscription
            )
            
            # Wait for response
            response = await self._wait_for_response(
                self.grader_reward_subscription,
                timeout
            )
            
            # Extract score
            return self._extract_score(response)
            
        except Exception as e:
            logger.error(f"Grader error: {e}")
            return 0.0
    
    async def _execute_web(self, payload: Dict[str, Any], timeout_s: int) -> str:
        """Execute web search tool."""
        q = str(payload.get("q", "")).strip()
        if not q:
            return "[web-error] Empty search query"
        
        k = int(payload.get("k", 3))
        
        if not self.connection_string or ServiceBusTopic is None:
            return f'[web-result] {{"mock": true, "query": "{q}", "results": []}}'
        
        message = {"type": "web", "q": q, "k": k}
        message_id = str(uuid4())
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_to_topic,
                message,
                message_id,
                self.web_subscription
            )
            
            response = await self._wait_for_response(self.web_subscription, timeout_s)
            return f"[web-result] {json.dumps(response, ensure_ascii=False)}"
            
        except Exception as e:
            logger.error(f"Web tool error: {e}")
            return f"[web-error] {e}"
    
    async def _execute_code(self, payload: Dict[str, Any], timeout_s: int) -> str:
        """Execute code execution tool."""
        command = str(payload.get("code_command", "")).strip()
        if not command:
            return "[code-error] Empty command"
        
        if not self.connection_string or ServiceBusTopic is None:
            return f'[code-result] {{"mock": true, "command": "{command}", "output": "Mock output"}}'
        
        message = {"type": "code", "code_command": command}
        message_id = str(uuid4())
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_to_topic,
                message,
                message_id,
                self.code_subscription
            )
            
            response = await self._wait_for_response(self.code_subscription, timeout_s)
            return f"[code-result] {json.dumps(response, ensure_ascii=False)}"
            
        except Exception as e:
            logger.error(f"Code tool error: {e}")
            return f"[code-error] {e}"
    
    async def _execute_azure(self, payload: Dict[str, Any], timeout_s: int) -> str:
        """Execute Azure CLI tool."""
        command = str(payload.get("azure_command", "")).strip()
        if not command:
            return "[azure-error] Empty command"
        
        if not self.connection_string or ServiceBusTopic is None:
            return f'[azure-result] {{"mock": true, "command": "{command}", "output": "Mock output"}}'
        
        message = {"type": "azure", "azure_command": command}
        message_id = str(uuid4())
        
        try:
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_to_topic,
                message,
                message_id,
                self.web_subscription  # Azure uses web subscription
            )
            
            response = await self._wait_for_response(self.web_subscription, timeout_s)
            return f"[azure-result] {json.dumps(response, ensure_ascii=False)}"
            
        except Exception as e:
            logger.error(f"Azure tool error: {e}")
            return f"[azure-error] {e}"
    
    def _send_to_topic(self, message: Dict, message_id: str, subscription: str) -> bool:
        """Synchronous send to Service Bus topic."""
        if ServiceBusTopic is None:
            return False
            
        try:
            with ServiceBusTopic(
                self.connection_string,
                topic_name=self.command_topic,
                subscription_name=subscription
            ) as topic:
                return topic.send_web_result(
                    message,
                    message_id=message_id,
                    wrap=False
                )
        except Exception as e:
            logger.error(f"Failed to send to topic: {e}")
            return False
    
    async def _wait_for_response(self, subscription: str, timeout_s: int) -> Dict:
        """Wait for response from reward topic."""
        if ServiceBusTopic is None:
            return {"message": "Service Bus not available"}
        
        try:
            reward_topic = ServiceBusTopic(
                self.connection_string,
                topic_name=self.reward_topic,
                subscription_name=subscription
            )
            
            for _ in range(timeout_s):
                try:
                    resp = await reward_topic.receive_web_reward_async()
                    if resp and resp.get("message") not in {"No rewards received", "No messages received"}:
                        return resp
                except Exception as e:
                    logger.error(f"Error polling reward topic: {e}")
                await asyncio.sleep(1)
                
        except Exception as e:
            logger.error(f"Failed to connect to reward topic: {e}")
        
        return {"message": "No response within timeout"}
    
    def _extract_score(self, response: Any) -> float:
        """Extract and normalize score from grader response."""
        if isinstance(response, dict):
            score = response.get("message")
            if isinstance(score, (int, float)):
                # Normalize from 1-5 to 0-1
                normalized = (score - 1.0) / 4.0
                return max(0.0, min(1.0, normalized))
        
        # Try to parse as number
        try:
            score = float(str(response).strip())
            normalized = (score - 1.0) / 4.0
            return max(0.0, min(1.0, normalized))
        except (TypeError, ValueError):
            pass
        
        return 0.0


# Convenience function for quick tool execution
async def execute_tool(tool_type: str, payload: Dict[str, Any], timeout_s: int = 30) -> str:
    """Quick async tool execution."""
    executor = AsyncToolExecutor()
    return await executor.execute_tool(tool_type, payload, timeout_s)


async def grade_completion(query: str, completion: str, timeout_s: int = 60) -> float:
    """Quick async grading."""
    executor = AsyncToolExecutor()
    return await executor.grade_completion(query, completion, timeout_s)
