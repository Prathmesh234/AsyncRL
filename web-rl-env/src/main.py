import os, json, asyncio, logging
from logging.handlers import RotatingFileHandler
from typing import Optional, Any
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel
from azure.servicebus import ServiceBusClient, ServiceBusMessage, TransportType
from azure.servicebus.aio import ServiceBusClient as AioServiceBusClient
from .command_queue import CommandQueue
from .reward_queue import RewardQueue
from dotenv import load_dotenv
from uuid import uuid4

from web_env.web_tool import WebTool

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
    # Set up simple file logging for web tool outputs
    def _setup_logger() -> logging.Logger:
        logs_dir = os.environ.get("LOGS_DIR", "/app/logs")
        os.makedirs(logs_dir, exist_ok=True)
        log_path = os.path.join(logs_dir, "web_tool.log")

        logger = logging.getLogger("web_tool")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)

        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

        file_handler = RotatingFileHandler(log_path, maxBytes=5 * 1024 * 1024, backupCount=3)
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)

        return logger

    app.state.logger = _setup_logger()

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
        # 1) Send to Service Bus (keep original behavior)
        msg_id = str(uuid4())
        payload = json.dumps(command.dict())
        message = ServiceBusMessage(payload, message_id=msg_id)
        with servicebus_client:
            with servicebus_client.get_queue_sender(COMMAND_QUEUE_NAME) as sender:
                sender.send_messages(message)

        # 2) Try to extract q,k directly from the posted body
        q = None
        k = None
        # Prefer a top-level `data` shape if present in action/parameters
        # This keeps compatibility with the CommandQueue parser
        body_dict = command.dict()
        # a) Directly in parameters
        if isinstance(body_dict.get("parameters"), dict):
            params = body_dict["parameters"]
            q = params.get("q", q)
            k = params.get("k", k)
        # b) Or inside action as free-form dict
        if q is None and isinstance(body_dict.get("action"), dict):
            q = body_dict["action"].get("q", q)
            k = body_dict["action"].get("k", k)

        # 3) If not found, poll the in-memory `/read-command` view briefly
        #    to leverage the parsed content from CommandQueue
        if q is None or k is None:
            # Give the receiver a moment to process the message
            for _ in range(10):  # ~5s total
                data_resp = await read_command()
                try:
                    data = json.loads(data_resp.body)
                except Exception:
                    data = None
                if isinstance(data, dict):
                    q = data.get("q", q)
                    k = data.get("k", k)
                if q is not None and k is not None:
                    break
                await asyncio.sleep(0.5)

        # 4) Run the WebTool if we have a query; print results only
        if q:
            try:
                tool: WebTool = getattr(app.state, "web_tool", None) or WebTool()
                app.state.web_tool = tool
                logger: logging.Logger = getattr(app.state, "logger", logging.getLogger("web_tool"))
                kk = int(k) if isinstance(k, (int, float, str)) and str(k).isdigit() else 3
                start_msg = f"[web_tool] starting query='{str(q)}' k={kk}"
                print(start_msg)
                logger.info(start_msg)
                results = await tool.run(str(q), kk)
                if not results:
                    nores = f"[web_tool] no results for query='{str(q)}'"
                    print(nores)
                    logger.info(nores)
                for r in results:
                    url = r.get("url", "")
                    title = r.get("title", "")
                    content = r.get("content", "")
                    if r.get("error"):
                        msg = f"[web_tool] ERROR url={url} err={r['error']}"
                        print(msg)
                        logger.info(msg)
                    else:
                        snippet = (content or "")[:200].replace("\n", " ")
                        msg = f"[web_tool] url={url}\n  title={title}\n  content={snippet}..."
                        print(msg)
                        logger.info(msg)
            except Exception as e:
                msg = f"[web_tool] failed to run: {e}"
                print(msg)
                logger = getattr(app.state, "logger", logging.getLogger("web_tool"))
                logger.exception(msg)

        return {"success": True, "message": "Command sent to Service Bus", "message_id": msg_id}
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
