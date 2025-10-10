# Grader Environment

This directory provides a grading environment for evaluating RL completions using disaggregated inference with the `gpt-120b-oss` model via [vLLM](https://docs.vllm.ai) with LMCache and the NIXL transfer layer.

## Architecture

The grader follows the same pattern as `azure-env` and `code-env`:

- **Command Queue**: Listens to Azure Service Bus for messages with `type: grader_command`, containing `query` and `completion` fields
- **Reward Queue**: Sends grading results back to Azure Service Bus
- **Disaggregated Inference**: Uses `prefill_disaggregated.py` and `decode_disaggregated.py` for efficient LLM-based grading

## Workflow

1. The ToolGRPOTrainer sends a message to the command topic with:
   - `type: grader_command`
   - `query`: The user's original query
   - `completion`: The full trajectory from the agent

2. The grader receives the command via the command queue

3. The grader runs:
   - `prefill_disaggregated.py` on the prefill GPU (2xA100) with query + completion
   - `decode_disaggregated.py` on the decode GPU to generate grading feedback

4. The grading result is sent back via the reward queue to Azure Service Bus

## Scripts

- `prefill_disaggregated.py`: Performs the prefill phase on a GPU host and stores the KV cache inside LMCache using the high-throughput NIXL transport (ZeroMQ based)
- `decode_disaggregated.py`: Attaches to the cached KV tensors from a different GPU host (or process) and performs decoding using the cached context

Both scripts expose the same environment variable `GRADER_PROMPT` so that you can override the prompt without modifying the code.

## Setup

1. **Configure environment**

   ```bash
   cp .env.example .env
   # Edit .env with your Azure Service Bus connection string
   ```

2. **Build and run with Docker**

   ```bash
   docker-compose up --build
   ```

3. **The grader will:**
   - Listen on port 8003 (mapped from container port 8001)
   - Subscribe to the command topic with `gradersubscription`
   - Filter for messages with `type: grader_command`
   - Send results to the reward topic

## API Endpoints

- `GET /` - Service status
- `GET /health` - Health check
- `GET /read-command` - View current command
- `POST /receive-command` - Manually send command (for testing)
- `POST /send-reward` - Manually send reward (for testing)

## Message Format

### Input (Command Topic)
```json
{
  "type": "grader_command",
  "query": "User's original query",
  "completion": "Full trajectory from the agent"
}
```

### Output (Reward Topic)
```json
{
  "type": "grader_results",
  "query": "User's original query",
  "completion": "Full trajectory from the agent",
  "status": "ok",
  "result": {
    "grading": "LLM-generated grading feedback",
    "cache_uri": "lmcache://grader/prefill/..."
  }
}
```

## Directory Structure

```
grader/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with command/reward queue integration
│   ├── command_queue.py     # Azure Service Bus command queue listener
│   └── reward_queue.py      # Azure Service Bus reward queue sender
├── prefill_disaggregated.py # LMCache prefill stage
├── decode_disaggregated.py  # LMCache decode stage
├── scripts/
│   └── start.sh             # Container startup script
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose configuration
├── pyproject.toml           # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```
