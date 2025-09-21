import os, json, asyncio, logging
from typing import Optional, Any, Dict
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict
from azure.servicebus import ServiceBusClient, ServiceBusMessage, TransportType
from azure.servicebus.aio import ServiceBusClient as AioServiceBusClient
from uuid import uuid4
from dotenv import load_dotenv

from .command_queue import CommandQueue
from .reward_queue import RewardQueue
from azure_env.azure_tool import AzureTool

load_dotenv()

app = FastAPI(title="Azure Tool Environment", version="0.1.0")

class CommandRequest(BaseModel):
    command_type: Optional[str] = None
    azure_command: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class RewardRequest(BaseModel):
    message: Any

SERVICE_BUS_CONNECTION_STRING = os.environ.get("SERVICE_BUS_CONNECTION_STRING")
if not SERVICE_BUS_CONNECTION_STRING:
    raise ValueError("SERVICE_BUS_CONNECTION_STRING env var required")
COMMAND_TOPIC_NAME = os.environ.get("COMMAND_TOPIC_NAME", "commandtopic")
COMMAND_SUBSCRIPTION_NAME = os.environ.get("COMMAND_SUBSCRIPTION_NAME", "azuresubscription")
REWARD_TOPIC_NAME  = os.environ.get("REWARD_TOPIC_NAME",  "rewardtopic")
REWARD_SUBSCRIPTION_NAME = os.environ.get("REWARD_SUBSCRIPTION_NAME", "webrewardsubscription")

servicebus_client = ServiceBusClient.from_connection_string(
    conn_str=SERVICE_BUS_CONNECTION_STRING,
    transport_type=TransportType.AmqpOverWebsocket,
    logging_enable=True,
)

aio_servicebus_client = AioServiceBusClient.from_connection_string(
    conn_str=SERVICE_BUS_CONNECTION_STRING,
    transport_type=TransportType.AmqpOverWebsocket,
    logging_enable=True,
)

@app.on_event("startup")
async def _startup():
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("azure_tool")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        logger.addHandler(h)
        return logger
    app.state.logger = _setup_logger()

    app.state.cmd_queue = CommandQueue(aio_servicebus_client, COMMAND_TOPIC_NAME, COMMAND_SUBSCRIPTION_NAME)
    await app.state.cmd_queue.start()
    app.state.reward_queue = RewardQueue(aio_servicebus_client, REWARD_TOPIC_NAME)
    app.state._processed_ids = set()

    async def _process_loop():
        while True:
            try:
                cmdq = getattr(app.state, "cmd_queue", None)
                payload = cmdq.current_command if cmdq else None
                if isinstance(payload, dict):
                    msg_id = payload.get("message_id")
                    data = payload.get("data")
                    if msg_id and msg_id not in app.state._processed_ids and isinstance(data, dict):
                        tool_type = payload.get("tool_type") or data.get("type")
                        if isinstance(tool_type, str):
                            tool_type = tool_type.strip().lower()
                        if tool_type != "azure":
                            app.state._processed_ids.add(msg_id)
                            continue

                        azure_command = data.get("azure_command")
                        if azure_command:
                            tool: AzureTool = getattr(app.state, "azure_tool", None) or AzureTool()
                            app.state.azure_tool = tool
                            logger: logging.Logger = getattr(app.state, "logger", logging.getLogger("azure_tool"))
                            logger.info(f"[azure_tool][topic] starting azure_command='{azure_command}'")
                            try:
                                result = await tool.run(azure_command)
                                status = result.get("status", "ok")
                            except Exception as e:
                                status = "error"
                                result = {"error": str(e)}
                            reward_payload: Dict[str, Any] = {
                                "type": "azure_tool_results",
                                "azure_command": azure_command,
                                "status": status,
                                "result": result,
                            }
                            try:
                                await app.state.reward_queue.send(reward_payload)
                            except Exception as exc:
                                logger.exception(f"[reward_topic] failed to publish: {exc}")
                            app.state._processed_ids.add(msg_id)
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.getLogger("azure_tool").exception(f"[worker] error: {e}")
                await asyncio.sleep(1.0)

    app.state.worker = asyncio.create_task(_process_loop(), name="azure-tool-worker")

@app.on_event("shutdown")
async def _shutdown():
    cmdq = getattr(app.state, "cmd_queue", None)
    if cmdq:
        await cmdq.stop()
    worker = getattr(app.state, "worker", None)
    if worker:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

@app.get("/")
async def root():
    return {"message": "Azure Tool Environment API", "status": "running"}

@app.get("/health")
async def health():
    try:
        with servicebus_client:
            return {"status": "healthy"}
    except Exception as e:
        return {"status": "degraded", "error": str(e)}

@app.post("/receive-command")
async def receive_command(command: CommandRequest):
    try:
        msg_id = str(uuid4())
        payload_dict = command.model_dump(exclude_none=True)
        payload = json.dumps(payload_dict)
        message = ServiceBusMessage(payload, message_id=msg_id)
        with servicebus_client:
            with servicebus_client.get_topic_sender(COMMAND_TOPIC_NAME) as sender:
                sender.send_messages(message)
        return {"success": True, "message": "Enqueued; background worker will process", "message_id": msg_id}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.post("/send-reward")
async def send_reward(reward: RewardRequest):
    try:
        rewardq = getattr(app.state, "reward_queue", None)
        if rewardq:
            await rewardq.send(reward.message)
        return {"success": True, "message": "Reward sent"}
    except Exception as e:
        return {"success": False, "error": str(e)}

@app.get("/read-command")
async def read_command():
    cmdq = getattr(app.state, "cmd_queue", None)
    payload = cmdq.current_command if cmdq else {"message": "Receiver not started"}
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
