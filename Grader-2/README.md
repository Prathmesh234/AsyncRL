# Grader-2 Environment

This directory provides a second grading environment that mirrors the original
setup but relies solely on the LMCache + vLLM NIXL transport (no ZeroMQ
endpoints). The service still loads the `openai/gpt-oss-120b` checkpoint and
uses the same FastAPI control plane as the original grader.

## Architecture

The grader follows the same pattern as `azure-env` and `code-env`:

- **Command Queue**: Listens to Azure Service Bus for messages with `type: grader_command`, containing `query` and `completion` fields
- **Reward Queue**: Sends grading results back to Azure Service Bus
- **Disaggregated Inference**: Uses `prefill_disaggregated.py` and `decode_disaggregated.py` for efficient LLM-based grading via LMCache + NIXL

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

- `prefill_disaggregated.py`: Performs the prefill phase on a GPU host and stores the KV cache using LMCache with the NIXL backend.
- `decode_disaggregated.py`: Attaches to the cached KV tensors from a different GPU host (or process) and performs decoding using the cached context.
- `scripts/run_prefill.sh`: Convenience wrapper that launches the prefill worker via `uv run` without any ZeroMQ parameters.
- `scripts/run_decode.sh`: Launches the decode worker and points it at the shared cache namespace.
- `scripts/run_disagg_serving.sh`: Helper for running the disaggregated prefill service inside Docker or on bare metal.

All scripts expose the same environment variable `GRADER_PROMPT` so that you can override the prompt without modifying the code.

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
The grader returns a simple numerical score (1-5) representing the quality of the completion:
```
4
```

The score is sent as a plain integer value (1=poor, 2=below average, 3=average, 4=good, 5=excellent), not wrapped in any JSON structure.

## Directory Structure

```
Grader-2/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with command/reward queue integration
│   ├── command_queue.py     # Azure Service Bus command queue listener
│   └── reward_queue.py      # Azure Service Bus reward queue sender
├── prefill_disaggregated.py # LMCache prefill stage (NIXL backend)
├── decode_disaggregated.py  # LMCache decode stage (NIXL backend)
├── scripts/
│   ├── run_decode.sh        # Helper to launch decode workers
│   ├── run_disagg_serving.sh# Helper to run the disaggregated prefill service
│   └── run_prefill.sh       # Helper to launch the prefill worker
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose configuration
├── pyproject.toml           # Python dependencies
├── .env.example             # Environment variables template
└── README.md                # This file
```
