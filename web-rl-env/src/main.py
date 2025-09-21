import os, json, asyncio, logging
from typing import Optional, Any, Dict, List
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict
from azure.servicebus import ServiceBusClient, ServiceBusMessage, TransportType
from azure.servicebus.aio import ServiceBusClient as AioServiceBusClient
from .command_queue import CommandQueue
from .reward_queue import RewardQueue
from dotenv import load_dotenv
from uuid import uuid4

from web_env.web_tool import WebTool

# Load environment variables from .env file
load_dotenv()

app = FastAPI(title="Simple Web RL Environment", version="0.2.3")

class CommandRequest(BaseModel):
    command_type: Optional[str] = None
    action: Optional[Any] = None
    parameters: Optional[dict] = None
    q: Optional[str] = None
    k: Optional[int] = None

    model_config = ConfigDict(extra="allow")


class RewardRequest(BaseModel):
    message: Any


# ✅ DO NOT hardcode secrets; use env var
# Use .get so a missing env var does not throw a TypeError
SERVICE_BUS_CONNECTION_STRING = os.environ.get("AZURE_SERVICE_BUS_CONNECTION_STRING")
if not SERVICE_BUS_CONNECTION_STRING:
    raise ValueError("AZURE_SERVICE_BUS_CONNECTION_STRING environment variable is required")
# Topic + subscription names
COMMAND_TOPIC_NAME = os.environ.get("COMMAND_TOPIC_NAME", "commandtopic")
COMMAND_SUBSCRIPTION_NAME = os.environ.get("COMMAND_SUBSCRIPTION_NAME", "rlcommandbustopic")
REWARD_TOPIC_NAME  = os.environ.get("REWARD_TOPIC_NAME",  "rewardtopic")
REWARD_SUBSCRIPTION_NAME = os.environ.get("REWARD_SUBSCRIPTION_NAME", "rlcommandbustopic")

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
    # Simple concise console logger
    def _setup_logger() -> logging.Logger:
        logger = logging.getLogger("web_tool")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(fmt)
        logger.addHandler(stream_handler)
        return logger

    app.state.logger = _setup_logger()

    # Start background async subscription receiver (push-style)
    app.state.cmd_queue = CommandQueue(aio_servicebus_client, COMMAND_TOPIC_NAME, COMMAND_SUBSCRIPTION_NAME)
    await app.state.cmd_queue.start()
    # Prepare reward sender (topic based)
    app.state.reward_queue = RewardQueue(aio_servicebus_client, REWARD_TOPIC_NAME)

    # Background worker: process new subscription messages and act via WebTool
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
                        if tool_type != "web":
                            app.state._processed_ids.add(msg_id)
                            continue

                        q = data.get("q")
                        k = data.get("k")
                        if q is not None:
                            tool: WebTool = getattr(app.state, "web_tool", None) or WebTool()
                            app.state.web_tool = tool
                            logger: logging.Logger = getattr(app.state, "logger", logging.getLogger("web_tool"))
                            kk = int(k) if isinstance(k, (int, float, str)) and str(k).isdigit() else 3
                            start_msg = f"[web_tool][topic] starting query='{str(q)}' k={kk}"
                            print(start_msg)
                            logger.info(start_msg)

                            normalized_results: List[Dict[str, Any]] = []
                            reward_status = "success"
                            reward_error: Optional[str] = None
                            has_result_error = False

                            try:
                                results = await tool.run(str(q), kk)
                                if not results:
                                    reward_status = "no_results"
                                    nores = f"[web_tool] no results for query='{str(q)}'"
                                    print(nores)
                                    logger.info(nores)
                                for idx, r in enumerate(results or []):
                                    url = r.get("url", "")
                                    title = r.get("title", "")
                                    content = r.get("content", "")
                                    if r.get("error"):
                                        has_result_error = True
                                        msg = f"[web_tool] ERROR url={url} err={r['error']}"
                                        print(msg)
                                        logger.info(msg)
                                        entry: Dict[str, Any] = {
                                            "rank": idx + 1,
                                            "url": url,
                                            "title": title,
                                            "error": r.get("error"),
                                        }
                                    else:
                                        preview = (content or "")[:300].replace("\n", " ")
                                        msg = f"[web_tool] url={url}\n  title={title}\n  content={preview}..."
                                        print(msg)
                                        logger.info(msg)
                                        entry = {
                                            "rank": idx + 1,
                                            "url": url,
                                            "title": title,
                                            "snippet": (content or "").replace("\n", " ")[:20000],
                                        }

                                    if "score" in r and r["score"] is not None:
                                        entry["score"] = r["score"]
                                    if "source" in r and r["source"]:
                                        entry["source"] = r["source"]
                                    if "metadata" in r and r["metadata"]:
                                        entry["metadata"] = r["metadata"]

                                    normalized_results.append(entry)

                                if has_result_error and reward_status == "success":
                                    reward_status = "partial"

                            except Exception as e:
                                reward_status = "error"
                                reward_error = str(e)
                                msg = f"[web_tool] failed to run: {e}"
                                print(msg)
                                logger.exception(msg)

                            reward_payload: Dict[str, Any] = {
                                "type": "web_tool_results",
                                "query": str(q),
                                "k": kk,
                                "status": reward_status,
                                "result_count": len(normalized_results),
                                "results": normalized_results,
                            }
                            if reward_error:
                                reward_payload["error"] = reward_error

                            try:
                                await app.state.reward_queue.send(reward_payload)
                            except Exception as exc:
                                logger.exception(f"[reward_topic] failed to publish results: {exc}")

                            app.state._processed_ids.add(msg_id)

                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                # keep the loop alive
                logging.getLogger("web_tool").exception(f"[worker] error: {e}")
                await asyncio.sleep(1.0)

    app.state.worker = asyncio.create_task(_process_loop(), name="web-tool-worker")


@app.on_event("shutdown")
async def _shutdown():
    # Stop background async poller
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
        # Only enqueue and ack; background worker will process topic messages
        msg_id = str(uuid4())
        payload_dict = command.model_dump(exclude_none=True)
        payload = json.dumps(payload_dict)
        message = ServiceBusMessage(payload, message_id=msg_id)
        with servicebus_client:
            with servicebus_client.get_topic_sender(COMMAND_TOPIC_NAME) as sender:
                sender.send_messages(message)
        # Return a simple success response; background worker will pick it up
        return {"success": True, "message": "Enqueued; background worker will process", "message_id": msg_id}
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
                with servicebus_client.get_topic_sender(REWARD_TOPIC_NAME) as sender:
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
