# Grader-2: LMCache Disaggregated Prefill Environment

Simple grading environment using LMCache for disaggregated prefill/decode with shell scripts.

## Architecture

```
Azure Service Bus (command_queue)
        ↓
FastAPI Application (src/main.py)
        ↓
  Proxy Server (shell script)
        ↓
   ┌────┴────┐
   ↓         ↓
Prefill    Decode
(GPU 0)    (GPU 1)
   └────┬────┘
     (LMCache KV Cache Transfer)
        ↓
Azure Service Bus (reward_queue)
```

## Components

### Shell Scripts
- **`start_prefill.sh`**: Launches vLLM prefill server on GPU 0
- **`start_decode.sh`**: Launches vLLM decode server on GPU 1
- **`start_proxy.sh`**: Launches proxy to coordinate prefill/decode
- **`start_all.sh`**: Master script that launches all services

### Configuration Files
- **`.env`**: Environment variables (ports, GPU assignments, model name, HF token)
- **`lmcache-prefiller-config.yaml`**: LMCache config for prefill server
- **`lmcache-decoder-config.yaml`**: LMCache config for decode server

### Application Code
- **`src/main.py`**: FastAPI app that reads from command_queue, sends to proxy, returns to reward_queue
- **`src/command_queue.py`**: Azure Service Bus command receiver
- **`src/reward_queue.py`**: Azure Service Bus reward sender

## How It Works

1. **Command arrives** from Azure Service Bus command_queue
2. **FastAPI app** receives command with `query` and `completion`
3. **Request sent to proxy** at `http://localhost:9100/v1/completions`
4. **Proxy coordinates**:
   - Sends to prefill server (GPU 0)
   - Prefill processes and shares KV cache via LMCache to decode
   - Decode generates response
5. **Response parsed** for numeric score (1-5)
6. **Score sent** to Azure Service Bus reward_queue

## Configuration

Edit `.env` file:

```bash
# Required: Your Hugging Face token
HF_TOKEN=your_token_here

# Model to use
MODEL_NAME=gpt-120b-oss

# GPU assignments
PREFILL_GPU=0
DECODE_GPU=1

# Ports
PREFILL_PORT=7100
DECODE_PORT=7200
PROXY_PORT=9100

# Azure Service Bus (Required)
SERVICE_BUS_CONNECTION_STRING=your_azure_service_bus_connection_string_here
COMMAND_TOPIC_NAME=commandtopic
COMMAND_SUBSCRIPTION_NAME=gradersubscription
REWARD_TOPIC_NAME=rewardtopic
REWARD_SUBSCRIPTION_NAME=webrewardsubscription
```

## Running

### With Docker Compose

```bash
# Build and start
docker-compose up --build

# The system will:
# 1. Start decode server on GPU 1
# 2. Start prefill server on GPU 0
# 3. Start proxy server
# 4. Start FastAPI application on port 8001
```

### Manually (for development)

```bash
# Terminal 1: Start decode server
./scripts/start_decode.sh

# Terminal 2: Start prefill server
./scripts/start_prefill.sh

# Terminal 3: Start proxy
./scripts/start_proxy.sh

# Terminal 4: Start FastAPI app
uvicorn src.main:app --host 0.0.0.0 --port 8001
```

## Ports

- **8001**: FastAPI application (mapped to 8003 in docker-compose)
- **7100**: Prefill vLLM server
- **7200**: Decode vLLM server
- **7300**: Decode initialization port
- **7400**: Decode allocation port
- **7500**: Proxy internal port
- **9100**: Proxy server (main entry point)

## Requirements

- 2 GPUs with NVLink or RDMA support (recommended)
- Python 3.11+
- CUDA 12.1+
- Dependencies:
  - `vllm` (latest main branch)
  - `lmcache` (0.2.1+)
  - `httpx`, `fastapi`, `uvicorn`
  - `azure-servicebus`

## API Endpoints

- `GET /` - Service status
- `GET /health` - Health check (includes proxy status)
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

The score is sent as a plain integer value (1=poor, 2=below average, 3=average, 4=good, 5=excellent).

## Testing

Send a test command to the FastAPI endpoint:

```bash
curl -X POST http://localhost:8001/receive-command \
  -H "Content-Type: application/json" \
  -d '{
    "type": "grader_command",
    "query": "What is 2+2?",
    "completion": "The answer is 4."
  }'
```

Check health:

```bash
curl http://localhost:8001/health
```

## Monitoring

View logs:

```bash
# Prefill server
tail -f /tmp/prefill.log

# Decode server
tail -f /tmp/decode.log

# Proxy server
tail -f /tmp/proxy.log
```

Check LMCache KV transfer throughput in prefill logs:
```
LMCache INFO: Store 5271 tokens takes: 6.5000 ms, throughput: 98.9889 GB/s
```

## Troubleshooting

1. **Services not starting**: Check that ports are available and HF_TOKEN is set
2. **GPU errors**: Verify CUDA_VISIBLE_DEVICES and GPU availability
3. **Connection refused**: Ensure decode server starts before prefill (handshake required)
4. **Slow performance**: Check that NVLink/RDMA is enabled with `ucx_perftest`

## Directory Structure

```
Grader-2/
├── src/
│   ├── __init__.py
│   ├── main.py              # FastAPI app with LMCache proxy integration
│   ├── command_queue.py     # Azure Service Bus command queue listener
│   └── reward_queue.py      # Azure Service Bus reward queue sender
├── scripts/
│   ├── start_prefill.sh     # Launch prefill vLLM server
│   ├── start_decode.sh      # Launch decode vLLM server
│   ├── start_proxy.sh       # Launch coordination proxy
│   └── start_all.sh         # Master startup script
├── lmcache-prefiller-config.yaml   # LMCache prefill configuration
├── lmcache-decoder-config.yaml     # LMCache decode configuration
├── Dockerfile               # Container definition
├── docker-compose.yml       # Docker Compose configuration
├── pyproject.toml           # Python dependencies
├── .env                     # Environment variables
└── README.md                # This file
```
