# DisGenerator: Disaggregated GPU Serving for Trajectory Generation

## Overview

DisGenerator generates training trajectories using **vLLM's disaggregated prefill-decode architecture**. It separates prompt processing (prefill) from token generation (decode) across different GPUs, enabling efficient batch processing and KV cache reuse.

**Key Innovation**: Uses vLLM's P2P NCCL communication to transfer KV cache tensors directly between GPUs, avoiding CPU bottlenecks and enabling true disaggregation.

---

## Requirements

### vLLM Version

This implementation requires **vLLM v0.6.4 or later** for P2P NCCL disaggregated serving support.

**Recommended**: Install the **latest stable version** of vLLM to access the most recent features and bug fixes.

#### Installation

```bash
# Install latest vLLM with CUDA support
pip install vllm

# Or install specific version (if needed)
pip install vllm==0.6.4.post1
```

#### System Requirements

- **Python**: 3.8 - 3.11
- **CUDA**: 12.1 or later (for NVIDIA GPUs)
- **PyTorch**: 2.0 or later
- **GPU**: NVIDIA GPUs with Compute Capability 7.0+ (V100, T4, A100, H100, etc.)
- **NCCL**: 2.18+ (auto-installed with CUDA toolkit)

#### Dependencies for DisGenerator

```bash
# Core dependencies
pip install aiohttp asyncio
pip install msgpack pyzmq quart  # For proxy server
pip install azure-servicebus      # For tool execution (if using Azure)

# Full vLLM installation
pip install vllm[all]
```

**Verify Installation**:
```bash
python -c "import vllm; print(vllm.__version__)"
# Should output: 0.6.4.post1 or later
```

**Reference**: [vLLM Installation Guide](https://docs.vllm.ai/en/latest/getting_started/installation.html)

---

## Architecture: P2P NCCL Disaggregated Serving

Based on vLLM's official [disaggregated serving documentation](https://docs.vllm.ai/en/latest/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/), we implement the following architecture:

### Components

1. **Prefill Instances (KV Producers)**: Process prompts and generate KV cache
2. **Decode Instances (KV Consumers)**: Receive KV cache and generate tokens
3. **Proxy Server**: Coordinates KV transfer routing between prefill and decode instances
4. **Python Orchestrator**: Manages request flow and implements tool calling logic

### Architecture Diagram (2P2D Configuration)

```
┌─────────────────────────────────────────────────────────────┐
│                    Proxy Server (CPU)                       │
│              Coordinates KV Transfer Routing                │
│              Port: 30001 (service discovery)                │
│              Port: 10001 (HTTP routing)                     │
└──────────────┬────────────────────┬─────────────────────────┘
               │                    │
    ┌──────────▼──────────┐  ┌─────▼──────────────┐
    │  Prefill Instance 1 │  │ Decode Instance 1  │
    │  (KV Producer)      │  │ (KV Consumer)      │
    │  GPU 0              │  │ GPU 2              │
    │  HTTP: 20001        │  │ HTTP: 20002        │
    │  KV Port: 21001     │  │ KV Port: 22001     │
    └─────────────────────┘  └────────────────────┘
    
    ┌─────────────────────┐  ┌────────────────────┐
    │  Prefill Instance 2 │  │ Decode Instance 2  │
    │  (KV Producer)      │  │ (KV Consumer)      │
    │  GPU 1              │  │ GPU 3              │
    │  HTTP: 20003        │  │ HTTP: 20004        │
    │  KV Port: 21002     │  │ KV Port: 22002     │
    └─────────────────────┘  └────────────────────┘
```

### How It Works: The KV Transfer Flow

**Step 1: Client sends request to Proxy**
```
Client → Proxy (http://localhost:10001/v1/chat/completions)
```

**Step 2: Proxy routes to Prefill instance with max_tokens=1**
```python
# Proxy modifies request
prefill_request = {
    "messages": [...],
    "max_tokens": 1,  # Only do prefill, no decode
    "X-Request-Id": "___prefill_addr_0.0.0.0:21001___decode_addr_0.0.0.0:22001_abc123"
}
# Sends to Prefill Instance 1 (http://localhost:20001)
```

**Step 3: Prefill instance processes and sends KV cache via NCCL**
- Prefill instance computes KV cache for the prompt
- Extracts decode address from `X-Request-Id` header
- Uses P2P NCCL to transfer KV cache directly to Decode instance's GPU
- Returns empty response (only used to warm up KV)

**Step 4: Proxy routes original request to Decode instance**
```python
# Proxy sends full request to Decode
decode_request = {
    "messages": [...],
    "max_tokens": 4096,  # Full generation
    "stream": True,
    "X-Request-Id": "___prefill_addr_0.0.0.0:21001___decode_addr_0.0.0.0:22001_abc123"
}
# Sends to Decode Instance 1 (http://localhost:20002)
```

**Step 5: Decode instance receives KV and generates tokens**
- Decode instance looks up KV cache using `X-Request-Id`
- KV cache already transferred via NCCL (available in GPU memory)
- Generates tokens and streams back to Proxy → Client

---

## Phase 1: Simple Serving (Current Focus)

**Goal**: Receive one prompt, compute prefill, transfer KV cache via NCCL, compute decode token by token.

### Implementation Components

#### 1. Prefill Server Launch

```bash
CUDA_VISIBLE_DEVICES=0 vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 20001 \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.9 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_producer","kv_buffer_size":"1e9","kv_port":"21001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20001","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'
```

**Key Parameters**:
- `kv_connector`: `P2pNcclConnector` - Uses NVIDIA NCCL for GPU-to-GPU transfer
- `kv_role`: `kv_producer` - This instance produces KV cache
- `kv_port`: `21001` - NCCL communication port
- `kv_buffer_size`: `1e9` - 1GB buffer for KV cache (small for prefill)
- `proxy_ip`, `proxy_port`: Coordinates with proxy for routing

#### 2. Decode Server Launch

```bash
CUDA_VISIBLE_DEVICES=2 vllm serve Qwen/Qwen3-4B-Thinking-2507 \
    --enforce-eager \
    --host 0.0.0.0 \
    --port 20002 \
    --tensor-parallel-size 1 \
    --dtype float16 \
    --max-model-len 8192 \
    --max-num-batched-tokens 8192 \
    --max-num-seqs 128 \
    --gpu-memory-utilization 0.7 \
    --kv-transfer-config \
    '{"kv_connector":"P2pNcclConnector","kv_role":"kv_consumer","kv_buffer_size":"8e9","kv_port":"22001","kv_connector_extra_config":{"proxy_ip":"0.0.0.0","proxy_port":"30001","http_port":"20002","send_type":"PUT_ASYNC","nccl_num_channels":"16"}}'
```

**Key Parameters**:
- `kv_role`: `kv_consumer` - This instance consumes KV cache
- `kv_buffer_size`: `8e9` - 8GB buffer for KV cache (larger for decode)
- `gpu_memory_utilization`: `0.7` - Lower to leave room for incoming KV cache

#### 3. Proxy Server (`disagg_proxy.py`)

The proxy server handles:
1. **Service Discovery**: Prefill/Decode instances register via ZMQ
2. **Request Routing**: Routes prefill → decode based on `X-Request-Id`
3. **Two-Phase Execution**: First prefill (max_tokens=1), then decode (full request)

**Key Code Snippets**:

```python
# Service discovery via ZMQ
context = zmq.Context()
router_socket = context.socket(zmq.ROUTER)
router_socket.bind(f"tcp://0.0.0.0:30001")

# Instances register themselves
# Prefill: {"type": "P", "http_address": "0.0.0.0:20001", "zmq_address": "0.0.0.0:21001"}
# Decode:  {"type": "D", "http_address": "0.0.0.0:20002", "zmq_address": "0.0.0.0:22001"}
```

```python
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    original_request = await request.get_json()
    
    # Select prefill and decode instances (round-robin)
    prefill_http, prefill_zmq = select_prefill_instance()
    decode_http, decode_zmq = select_decode_instance()
    
    # Create request ID encoding routing info
    request_id = f"___prefill_addr_{prefill_zmq}___decode_addr_{decode_zmq}_{uuid.uuid4().hex}"
    
    # Step 1: Prefill (max_tokens=1)
    prefill_request = original_request.copy()
    prefill_request["max_tokens"] = 1
    async for _ in forward_request(f"http://{prefill_http}/v1/chat/completions", 
                                   prefill_request, request_id):
        pass  # Discard prefill response
    
    # Step 2: Decode (full request)
    generator = forward_request(f"http://{decode_http}/v1/chat/completions",
                                original_request, request_id)
    return make_response(generator)
```

#### 4. Python Client (`simple_client.py`)

```python
import aiohttp
import asyncio

async def generate_simple(prompt: str):
    """Send a single prompt through disagg serving"""
    async with aiohttp.ClientSession() as session:
        payload = {
            "model": "Qwen/Qwen3-4B-Thinking-2507",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 100,
            "stream": True
        }
        
        # Send to proxy
        async with session.post("http://localhost:10001/v1/chat/completions", 
                               json=payload) as resp:
            async for line in resp.content:
                # Parse SSE stream
                if line.startswith(b"data: "):
                    data = json.loads(line[6:])
                    if "choices" in data and len(data["choices"]) > 0:
                        delta = data["choices"][0].get("delta", {})
                        if "content" in delta:
                            print(delta["content"], end="", flush=True)

# Usage
asyncio.run(generate_simple("What is the capital of France?"))
```

---

## Configuration

### GPU Allocation

**Option 1: 2P2D (2 Prefill + 2 Decode)**
- Prefill: GPU 0, GPU 1
- Decode: GPU 2, GPU 3
- Best for: Balanced workload, moderate concurrency

**Option 2: 1P3D (1 Prefill + 3 Decode)**
- Prefill: GPU 0
- Decode: GPU 1, GPU 2, GPU 3
- Best for: Long generation, high decode throughput

**Option 3: 3P1D (3 Prefill + 1 Decode)**
- Prefill: GPU 0, GPU 1, GPU 2
- Decode: GPU 3
- Best for: Short generation, many concurrent prompts

### Model Configuration

```bash
MODEL="Qwen/Qwen3-4B-Thinking-2507"  # Matches DisTrainer
MAX_MODEL_LEN=8192                    # Context window
MAX_NUM_SEQS=128                      # Max concurrent requests
DTYPE="float16"                       # Use FP16 for efficiency
```

---

## Implementation Roadmap

### Phase 1: Simple Serving ✅ **IMPLEMENTED**

**Objective**: Prove disaggregated serving works for a single prompt.

**Tasks**:
- [x] Write `scripts/start_prefill.sh` - Launch prefill instance with P2P NCCL config
- [x] Write `scripts/start_decode.sh` - Launch decode instance with P2P NCCL config  
- [x] Write `scripts/start_proxy.sh` - Launch proxy server for routing
- [x] Write `scripts/start_all.sh` - Launch full stack (1P1D, 2P2D, 1P3D, 3P1D configs)
- [x] Write `scripts/stop_all.sh` - Graceful shutdown
- [x] Write `disagg_proxy.py` - Implement proxy with service discovery and two-phase routing
- [x] Write `simple_client.py` - Test client that sends one prompt through proxy
- [x] Write `config.py` - Configuration dataclasses for GPU/port allocation
- [x] Write `pyproject.toml` - Dependencies managed with `uv`
- [ ] Test end-to-end: Prompt → Prefill → KV Transfer → Decode → Response (on H100s)

**Success Criteria**:
- Single prompt generates response correctly
- Logs show KV transfer happening via NCCL
- Decode latency is low (KV already in GPU memory)

### Phase 2: Batched Serving

**Objective**: Handle multiple concurrent prompts efficiently.

**Tasks**:
- [ ] Implement async batch orchestrator (10+ prompts simultaneously)
- [ ] Add request queuing and rate limiting
- [ ] Benchmark throughput vs. single-instance vLLM

**Success Criteria**:
- 10+ concurrent prompts processed efficiently
- Prefill and Decode GPUs show balanced utilization
- Throughput exceeds single-instance baseline

### Phase 3: Tool Calling Integration

**Objective**: Support multi-turn trajectories with tool execution.

**Tasks**:
- [ ] Add tool call detection (stop sequences: `<tool_call>`, `<solution>`)
- [ ] Integrate Azure Service Bus execution (reuse `serving/communication/`)
- [ ] Implement multi-turn loop (prefill → decode → tool → prefill → decode)

**Success Criteria**:
- Tools execute correctly mid-generation
- KV cache properly resumes after tool results appended
- Tool latency doesn't block other prompts

### Phase 4: JSONL Output for DisTrainer

**Objective**: Generate training data in DisTrainer-compatible format.

**Tasks**:
- [ ] Collect logprobs during decode (`logprobs=True`)
- [ ] Call reward model for each completion
- [ ] Write JSONL with format: `{prompt, completions: [{text, ids, logprobs, reward}]}`

**Success Criteria**:
- JSONL files load correctly in DisTrainer
- Logprobs match token IDs
- Rewards correlate with completion quality

---

## File Structure (Implemented)

```
serving/DisGenerator/
├── DisGenerator.MD              # This document
├── pyproject.toml               # Dependencies (managed with uv)
├── config.py                    # Configuration dataclasses
├── disagg_proxy.py              # Proxy server (service discovery + routing)
├── simple_client.py             # Phase 1: Single prompt test client
├── scripts/
│   ├── start_all.sh             # 🚀 Launch full stack (main entry point)
│   ├── start_prefill.sh         # Launch prefill instance(s)
│   ├── start_decode.sh          # Launch decode instance(s)
│   ├── start_proxy.sh           # Launch proxy server only
│   └── stop_all.sh              # Graceful shutdown all services
├── logs/                        # Runtime logs (auto-created)
│   ├── proxy.log
│   ├── prefill_gpu0_port20001.log
│   └── decode_gpu1_port20002.log
└── (Phase 2-4 files - TODO)
    ├── batch_orchestrator.py    # Phase 2: Multi-prompt async orchestrator
    ├── tool_integration.py      # Phase 3: Tool calling logic
    └── jsonl_writer.py          # Phase 4: DisTrainer output formatter
```

---

## Quick Start (Phase 1)

### Prerequisites

1. **Install uv** (Python package manager):
   ```bash
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **GPU Requirements**: At least 2 NVIDIA GPUs (for 1P1D configuration)

3. **Optional**: Set HuggingFace token for gated models:
   ```bash
   export HF_TOKEN=hf_your_token_here
   ```

### Usage

```bash
# Navigate to DisGenerator folder
cd serving/DisGenerator

# Install dependencies (first time only)
uv sync

# Start the full stack (1P1D: 1 Prefill + 1 Decode)
./scripts/start_all.sh 1p1d

# Or for 4 GPUs (2P2D: 2 Prefill + 2 Decode)
./scripts/start_all.sh 2p2d

# Test with the simple client
uv run python simple_client.py "What is the capital of France?"

# Or use curl directly
curl -X POST http://localhost:10001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model": "Qwen/Qwen3-4B-Thinking-2507", "messages": [{"role": "user", "content": "Hello!"}]}'

# Stop all services
./scripts/stop_all.sh
```

### Configuration Options

| Config | GPUs Required | Description |
|--------|---------------|-------------|
| `1p1d` | 2 | 1 Prefill + 1 Decode (default) |
| `2p2d` | 4 | 2 Prefill + 2 Decode (balanced) |
| `1p3d` | 4 | 1 Prefill + 3 Decode (decode heavy) |
| `3p1d` | 4 | 3 Prefill + 1 Decode (prefill heavy) |

### Environment Variables

```bash
# Model configuration (matches DisTrainer)
export MODEL="Qwen/Qwen3-4B-Thinking-2507"  # Model to serve
export MAX_MODEL_LEN=8192                    # Max sequence length
export DTYPE=float16                         # Data type

# Port configuration  
export PROXY_PORT=30001                   # ZMQ service discovery port
export PROXY_HTTP_PORT=10001              # API endpoint port

# Timeouts
export TIMEOUT_SECONDS=600                # Server startup timeout
```

## Key Differences from Previous Approach

| Aspect | Old Approach | New Approach (P2P NCCL) |
|--------|-------------|-------------------------|
| **KV Transfer** | Application-level via HTTP | GPU-to-GPU via NCCL |
| **Prefill/Decode** | Manual orchestration | vLLM built-in disagg mode |
| **Request Routing** | Custom Python logic | Proxy + `X-Request-Id` header |
| **Service Discovery** | Hardcoded URLs | ZMQ-based dynamic registration |
| **Scalability** | 1P + 1D only | XpYd (flexible X prefill, Y decode) |

---

## Testing Strategy

### Test 1: Minimal Viability (Phase 1)

```bash
# Terminal 1: Start proxy
python disagg_proxy.py

# Terminal 2: Start prefill
bash scripts/start_prefill.sh

# Terminal 3: Start decode
bash scripts/start_decode.sh

# Terminal 4: Run test
python simple_client.py
```

**Expected Output**:
```
🔵Add [HTTP:0.0.0.0:20001, ZMQ:0.0.0.0:21001]  # Prefill registered
🔵Add [HTTP:0.0.0.0:20002, ZMQ:0.0.0.0:22001]  # Decode registered
handle_request count: 0, [HTTP:0.0.0.0:20001, ZMQ:0.0.0.0:21001] 👉 [HTTP:0.0.0.0:20002, ZMQ:0.0.0.0:22001]
The capital of France is Paris.
```

### Test 2: KV Transfer Verification

Check vLLM logs for NCCL activity:
```bash
tail -f logs/prefill.log | grep -i nccl
tail -f logs/decode.log | grep -i "kv"
```

**Expected**:
- Prefill logs show NCCL send operations
- Decode logs show KV cache lookup by request_id

---

## FAQ

### Q: Why P2P NCCL instead of HTTP for KV transfer?

**A**: HTTP requires:
1. Serialize KV cache tensors to CPU memory
2. Send via network stack
3. Deserialize on receiving GPU

NCCL directly transfers GPU→GPU, bypassing CPU entirely. **10-100x faster** for large KV caches.

### Q: Can I run this without a proxy?

**A**: No. The proxy is essential for:
1. Service discovery (which prefill/decode instances are alive)
2. Two-phase routing (prefill first, then decode)
3. Encoding routing info in `X-Request-Id` headers

### Q: What if prefill and decode are on different machines?

**A**: NCCL supports RDMA for cross-machine transfers. Requires:
- InfiniBand or RoCE network
- NCCL configured with `NCCL_IB_DISABLE=0`
- Same NCCL version on all nodes

### Q: How does this scale to 8 GPUs?

**A**: Configure 4P4D (4 prefill + 4 decode):
```bash
PREFILL_GPUS=0,1,2,3
DECODE_GPUS=4,5,6,7
PREFILL_PORTS=20001,20002,20003,20004
DECODE_PORTS=20005,20006,20007,20008
```

Proxy will round-robin requests across all instances.

---

## Next Steps

1. **Read vLLM disagg serving code**:
   - `vllm/entrypoints/openai/api_server.py` - OpenAI API compatibility layer
   - `vllm/engine/async_llm_engine.py` - Request handling
   - `vllm/worker/kv_connector.py` - P2P NCCL connector

2. **Implement Phase 1** (Simple Serving):
   - Start with provided bash scripts
   - Copy proxy code from vLLM docs
   - Test single prompt end-to-end

3. **Iterate to Phase 2** (Batching):
   - Once Phase 1 works, add concurrent requests
   - Measure throughput vs. baseline

---

## References

- **vLLM P2P NCCL Docs**: https://docs.vllm.ai/en/latest/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
- **vLLM GitHub**: https://github.com/vllm-project/vllm
- **NCCL Documentation**: https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/
- **Prime Intellect INTELLECT-3**: https://www.primeintellect.ai/blog/intellect-3
