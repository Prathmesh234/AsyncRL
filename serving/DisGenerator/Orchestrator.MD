# Async Batch Orchestrator - Design Document

## Objective
To orchestrate large-scale, multi-turn generation tasks that involve:
1.  **Async Concurrency**: Processing hundreds of prompts without blocking.
2.  **Tool Interception**: Detecting tool calls in the stream, pausing generation, executing the tool, and resuming.
3.  **High GPU Utilization**: Decoupling the slow "Tool Execution" phase from the fast "GPU Generation" phase.

## Architecture

The Orchestrator acts as the "State Manager" while the GPU acts as a stateless "Token Generator".

```mermaid
graph TD
    Q[Task Queue] -->|Pop| W[Worker Thread]
    W -->|HTTP Request| Proxy[DisAGG Proxy]
    Proxy -->|Stream| W
    W -->|Check| Check{Tool Tag?}
    
    Check -- Yes (<web>, <code>, <azure>) --> Parser[Extract JSON]
    Parser --> Tool[Execute Tool (Async)]
    Tool -->|Result| Update[Append <tool_result>]
    Update -->|Re-Queue| Q
    
    Check -- No (EOS / <solution>) --> Save[Save JSONL]
```

## Core Components

### 1. The Trajectory Object
A simple data class that holds the state of a single conversation.

```python
@dataclass
class Trajectory:
    id: str                  # Unique ID
    messages: List[Dict]     # Full OpenAI-format history
    completions: List[str]   # Saved completions for training data
    status: str = "QUEUED"   # QUEUED, GENERATING, TOOL_WAITING, COMPLETED
```

### 2. The Worker Loop
We spawn `N` concurrent workers (e.g., 50). Each worker runs an infinite loop processing tasks from the queue.

**The Algorithm:**
1.  **Fetch**: Get a `Trajectory` from the Queue.
2.  **Generate**: Send `trajectory.messages` to the Proxy.
3.  **Stream & Detect**:
    *   Stream response chunks.
    *   Use `stream_parser` logic to detect complete tags: `<web>`, `<code>`, `<azure>`.
    *   If tag detected: **ABORT STREAM**.
4.  **Branch**:
    *   **If Tool Detected**:
        1.  Parse the inner content (JSON) -> `{"q": "...", "k": 3}`.
        2.  `await` the Tool Execution (e.g., `send_web_command`). **(GPU is free now!)**
        3.  Format result: `<tool_result>{result}</tool_result>\n`.
        4.  Append Tool Result to `trajectory.messages`.
        5.  Put Trajectory back in **Queue**.
    *   **If EOS or `<solution>`**:
        1.  Save final trajectory to JSONL.
        2.  Mark as COMPLETED.

## Implementation Details

### File: `batch_orchestrator.py`

This file will be implemented in Phase 2.

**Dependencies:**
- `aiohttp`: For async HTTP requests to Proxy.
- `asyncio`: For managing the Queue and Workers.
- `serving.communication` modules: For executing actual tools.
- `serving.parser`: Reuse `stream_parser` logic.

### Example Logic (Simplified)

```python
async def worker_loop(self):
    while True:
        traj = await self.queue.get()
        
        # 1. Call GPU
        response = await self.call_gpu(traj.messages)
        
        # 2. Check Interception (reuse serving/parser.py logic)
        tool_call = stream_parser(response)
        
        if tool_call:
            tool_type = tool_call["type"]
            content = tool_call["content"]
            
            # 3. Execute Tool (Async/Slow)
            if tool_type == "web":
                result = await self.execute_web_tool(content)
            elif tool_type == "code":
                result = await self.execute_code_tool(content)
            elif tool_type == "azure":
                result = await self.execute_azure_tool(content)
            
            # 4. Update & Re-Queue
            # Mimic ToolGRPOTrainer format
            tool_str = f"<{tool_type}>{content}</{tool_type}>"
            result_str = f"<tool_result>{result}</tool_result>\n"
            
            traj.messages.append({"role": "assistant", "content": tool_str})
            traj.messages.append({"role": "tool", "content": result_str}) # or user role
            
            await self.queue.put(traj)
            
        else:
            # 5. Finish
            if "<solution>" in response or "<eos>" in response:
                self.save_jsonl(traj)
```

## Why This Design?

1.  **Max GPU Efficiency**: The expensive H100s never wait for Azure. If a tool takes 5 seconds, the GPU serves 10 other requests during that time.
2.  **Prefix Caching**: When a trajectory is re-queued, vLLM automatically reuses the KV cache from the previous turn, making the "resume" step extremely fast.
3.  **Consistency**: By strictly following `ToolGRPOTrainer` tag logic (`<web>`, `<tool_result>`, `<solution>`), generated data is instantly training-ready.
