# Simple Web RL Environment

A basic web-based reinforcement learning environment API built with FastAPI and Docker.

## Features

- **Health Check**: `/health` endpoint to check API status
- **Receive Command**: `/receive-command` endpoint to receive commands from clients
- **Send Command**: `/send-command` endpoint to send commands to the RL environment

## Quick Start

### Using Docker Compose

1. Build and run the application:
```bash
docker-compose up --build
```

2. The API will be available at `http://localhost:8000`

### Using UV (Python package manager)

1. Install dependencies:
```bash
uv sync
```

2. Run the application:
```bash
uv run python -m uvicorn src.main:app --host 0.0.0.0 --port 8000 --reload
```

## API Endpoints

- `GET /` - Root endpoint with API information
- `GET /health` - Health check endpoint
- `POST /receive-command` - Receive a command from client
- `POST /send-command` - Send a command to RL environment

## Example Usage

### Send a command
```bash
curl -X POST "http://localhost:8000/send-command" \
     -H "Content-Type: application/json" \
     -d '{
       "command_type": "step",
       "action": 1,
       "parameters": {"episode": 1}
     }'
```

### Receive a command
```bash
curl -X POST "http://localhost:8000/receive-command" \
     -H "Content-Type: application/json" \
     -d '{
       "command_type": "reset",
       "parameters": {}
     }'
```

## Project Structure

```
.
├── Dockerfile
├── docker-compose.yml
├── pyproject.toml
├── README.md
└── src/
    ├── __init__.py
    └── main.py
```

## Development

This project uses:
- **FastAPI** for the web framework
- **UV** for Python package management
- **Docker** for containerization
