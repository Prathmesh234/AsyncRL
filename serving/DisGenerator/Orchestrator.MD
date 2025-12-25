# Async Batch Orchestrator - Design Document

## Objective
To orchestrate large-scale, multi-turn generation tasks that involve:
1.  **Async Concurrency**: Processing hundreds of prompts without blocking.
2.  **Tool Interception**: Detecting tool calls in the stream, pausing generation, executing the tool, and resuming.
3.  **High GPU Utilization**: Decoupling the slow "Tool Execution" phase from the fast "GPU Generation" phase.

## Architecture

The system uses a **Dual-Queue Event Loop** to decouple expensive GPU inference from slow I/O tool execution.

```mermaid
graph TD
    subgraph "Queue System"
        TQ[Task Queue\n(Ready for GPU)]
        ToolQ[Tool Queue\n(Needs Execution)]
    end

    subgraph "Inference Loop (Fast)"
        TQ -->|Pop| GW[GPU Worker]
        GW -->|Generate| Proxy[vLLM Proxy]
        Proxy -->|Stream Tokens| GW
        GW -->|Detect Tool?| Decision{Tag Found?}
        Decision -- Yes -->|Abort & Push| ToolQ
        Decision -- No -->|Finish| Save[Save Result]
    end

    subgraph "Tool Loop (Slow/Async)"
        ToolQ -->|Pop| TW[Tool Worker]
        TW -->|Extract JSON| Parser
        Parser -->|IO Wait| Ext[Execute Tool\n(Azure/Web/Code)]
        Ext -->|Result| Appender[Append <tool_result>]
        Appender -->|Push Back| TQ
    end
```

## Core Components

### 1. The Queues
*   **`TaskQueue`**: Holds `Trajectory` objects that need text generation. These are either new prompts or conversations waiting for the model to generate the next response after a tool result.
*   **`ToolQueue`**: Holds `Trajectory` objects where the model has just emitted a tool call (e.g., `<web>`). These need to be parsed and executed.

### 2. The Workers

#### A. GPU Workers (Inference)
*   **Role**: Manage the connection to vLLM.
*   **Concurrency**: High (e.g., 50+ workers) to keep the GPU batch full.
*   **Logic**:
    1.  Pop `Trajectory` from `TaskQueue`.
    2.  Send prompt/history to Proxy.
    3.  Stream tokens using `stream_parser`.
    4.  **CRITICAL**: If a tool tag (e.g., `<web>`) is detected:
        *   **Abort** the HTTP request immediately (frees the GPU slot).
        *   Push the `Trajectory` (with the partial tool call) to `ToolQueue`.
    5.  If `<solution>` or EOS is reached, save and finish.

#### B. Tool Workers (Execution)
*   **Role**: Handle external I/O (Azure Service Bus, etc.).
*   **Concurrency**: Configurable (e.g., `MAX_TOOL_WORKERS = 32`). Uses `asyncio` to handle thousands of waiting states without blocking.
*   **Logic**:
    1.  Pop `Trajectory` from `ToolQueue`.
    2.  Extract the JSON payload from the last message.
    3.  Call the appropriate async tool function (`send_web_command`, etc.).
    4.  **Wait** for the result (non-blocking).
    5.  Format the result as `<tool_result>...</tool_result>`.
    6.  Append to `Trajectory` history.
    7.  Push back to `TaskQueue`.

## Implementation Details

### File: `batch_orchestrator.py`

**Dependencies:**
- `aiohttp`: For async HTTP requests to Proxy.
- `asyncio`: For managing the Queues and Worker tasks.
- `serving.communication` modules: For executing actual tools.
- `serving.parser`: Reuse `stream_parser` logic.

### Pseudo-Code

```python
class AsyncBatchOrchestrator:
    def __init__(self):
        self.task_queue = asyncio.Queue()  # For GPU
        self.tool_queue = asyncio.Queue()  # For Tools
        
        # Concurrency Limits
        self.num_gpu_workers = 64
        self.num_tool_workers = 32
        
    async def start(self):
        # Start pure async consumers
        gpu_tasks = [asyncio.create_task(self.gpu_worker()) 
                     for _ in range(self.num_gpu_workers)]
        tool_tasks = [asyncio.create_task(self.tool_worker()) 
                      for _ in range(self.num_tool_workers)]
        
        await asyncio.gather(*gpu_tasks, *tool_tasks)

    async def gpu_worker(self):
        while True:
            traj = await self.task_queue.get()
            # Send to vLLM...
            # If tool detected -> self.tool_queue.put(traj)
            # Else -> Save

    async def tool_worker(self):
        while True:
            traj = await self.tool_queue.get()
            # Parse tool tag...
            # result = await run_tool(payload)
            # traj.history.append(result)
            # self.task_queue.put(traj)
```

## Why This Design?

1.  **Zero GPU Blocking**: The GPU never waits for a network request. It processes a token, sees a tool, and immediately moves to the full queue of other tasks.
2.  **Scalability**: You can independently scale `num_gpu_workers` (to saturate GPU batch size) and `num_tool_workers` (to respect API rate limits).
3.  **Prefix Caching**: When the `Tool Worker` pushes back to `Task Queue`, the prompt sent to vLLM is `[History + Tool Call + Tool Result]`. The *History + Tool Call* part is identical to the previous run, so vLLM's RadixAttention will skip computation for 99% of the sequence.

## Running DisGenerator

To run the full DisGenerator system on a 4-GPU node (2 Prefill, 2 Decode), follow these steps:

### 1. Start the Disaggregated Logic
You need to launch the Proxy and the 4 vLLM instances. Assuming you have the existing launch script `scripts/start_all.sh`:

```bash
# Launch Proxy and vLLM Engines
./scripts/start_all.sh
```

*Ensure that your config defaults to `get_config_2p2d()` or explicitly sets it in the launch arguments.*

**LoRA Adapter Configuration:**
The system uses a LoRA checkpoint from GRPO training. The scripts already include:

```bash
# These flags are already in start_prefill.sh and start_decode.sh
--enable-lora \
--lora-modules grpo-adapter=/path/to/serving/ToolGRPOTrainer/grpo-streamed/checkpoint-10
```

Set the `LORA_MODULE_PATH` environment variable if your path differs from the default.

### 2. Start the Orchestrator Client
Run the `simple_client.py` which wraps the `AsyncBatchOrchestrator`. This will load prompts from `prompts.jsonl`, submit them to the proxy, handle tool calls, and save completed trajectories.

```bash
# Set environment variables for Workers
export PROXY_URL="http://localhost:10001/v1/chat/completions"
export NUM_GPU_WORKERS=64
export NUM_TOOL_WORKERS=32

# Set the LoRA adapter name (must match the name in --lora-modules)
export MODEL="grpo-adapter"

# Run the Client
python simple_client.py
```

### 3. Monitor Output
*   **Logs**: Check the console output of `simple_client.py` for queue status.
*   **Data**: Completed trajectories will be appended to `completed_trajectories.jsonl`.
