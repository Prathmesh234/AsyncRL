import os, json
from typing import Optional, Any
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from azure.servicebus import ServiceBusClient, ServiceBusMessage, TransportType
from azure.servicebus.aio import ServiceBusClient as AioServiceBusClient
from .command_queue import CommandQueue
from .reward_queue import RewardQueue
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Simple Web RL Environment", version="0.2.1")

class CommandRequest(BaseModel):
    command_type: str
    action: Optional[Any] = None
    parameters: Optional[dict] = None


class RewardRequest(BaseModel):
    message: Any

# ✅ DO NOT hardcode secrets; use env var
# Use .get so a missing env var does not throw a TypeError
SERVICE_BUS_CONNECTION_STRING = os.environ.get("AZURE_SERVICE_BUS_CONNECTION_STRING")
if not SERVICE_BUS_CONNECTION_STRING:
    raise ValueError("AZURE_SERVICE_BUS_CONNECTION_STRING environment variable is required")
COMMAND_QUEUE_NAME = os.environ.get("COMMAND_QUEUE_NAME", "commandqueue")
REWARD_QUEUE_NAME  = os.environ.get("REWARD_QUEUE_NAME",  "rewardqueue")

# ✅ Use AMQP over WebSockets (works behind firewalls/NAT; port 443)
servicebus_client = ServiceBusClient.from_connection_string(
    conn_str=SERVICE_BUS_CONNECTION_STRING,
    transport_type=TransportType.AmqpOverWebsocket,
    logging_enable=True,  # helpful diagnostics
)

# Async client for push-style receiving
aio_servicebus_client = AioServiceBusClient.from_connection_string(
    conn_str=SERVICE_BUS_CONNECTION_STRING,
    transport_type=TransportType.AmqpOverWebsocket,
    logging_enable=True,
)

@app.on_event("startup")
async def _startup():
    # Start background async receiver (push-style)
    app.state.cmd_queue = CommandQueue(aio_servicebus_client, COMMAND_QUEUE_NAME)
    await app.state.cmd_queue.start()
    # Prepare reward sender
    app.state.reward_queue = RewardQueue(aio_servicebus_client, REWARD_QUEUE_NAME)


@app.on_event("shutdown")
async def _shutdown():
    # Stop background async poller
    cmdq = getattr(app.state, "cmd_queue", None)
    if cmdq:
        await cmdq.stop()

@app.get("/")
async def root():
    return {"message": "Simple Web RL Environment API", "status": "running"}

@app.get("/health")
async def health_check():
    # Try a lightweight management operation to surface connectivity issues
    try:
        with servicebus_client:
            return {"status": "healthy"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/receive-command")
async def receive_command(command: CommandRequest):
    try:
        with servicebus_client:
            with servicebus_client.get_queue_sender(COMMAND_QUEUE_NAME) as sender:
                sender.send_messages(ServiceBusMessage(json.dumps(command.dict())))
        return {"success": True, "message": "Command sent to Service Bus"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/send-reward")
async def send_reward(reward: RewardRequest):
    try:
        rewardq = getattr(app.state, "reward_queue", None)
        if rewardq:
            await rewardq.send(reward.message)
        else:
            # Fallback to sync client if async sender is unavailable
            payload = reward.message if isinstance(reward.message, str) else json.dumps(reward.message)
            with servicebus_client:
                with servicebus_client.get_queue_sender(REWARD_QUEUE_NAME) as sender:
                    sender.send_messages(ServiceBusMessage(payload))
        return {"success": True, "message": "Reward sent to Service Bus"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/read-command")
async def read_command():
    cmdq = getattr(app.state, "cmd_queue", None)
    payload = cmdq.current_command if cmdq else {"message": "Receiver not started"}
    # Return just the `data` object when available; otherwise return payload as-is
    if isinstance(payload, dict) and isinstance(payload.get("data"), dict):
        data_only = payload["data"]
    else:
        data_only = payload
    return Response(content=json.dumps(data_only, indent=2, ensure_ascii=False), media_type="application/json")


@app.get("/receive-command-logs")
async def receive_command_logs():
    cmdq = getattr(app.state, "cmd_queue", None)
    payload = cmdq.current_command if cmdq else {"message": "Receiver not started"}
    return Response(content=json.dumps(payload, indent=2, ensure_ascii=False), media_type="application/json")
