import os, json, asyncio, logging
import httpx
from typing import Optional, Any
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel, ConfigDict
from azure.servicebus import ServiceBusClient, ServiceBusMessage, TransportType
from azure.servicebus.aio import ServiceBusClient as AioServiceBusClient
from uuid import uuid4
from dotenv import load_dotenv

from .command_queue import CommandQueue
from .reward_queue import RewardQueue

load_dotenv()

app = FastAPI(title="Grader Environment", version="0.1.0")

class CommandRequest(BaseModel):
    command_type: Optional[str] = None
    query: Optional[str] = None
    completion: Optional[str] = None
    model_config = ConfigDict(extra="allow")

class RewardRequest(BaseModel):
    message: Any

SERVICE_BUS_CONNECTION_STRING = os.environ.get("SERVICE_BUS_CONNECTION_STRING")
if not SERVICE_BUS_CONNECTION_STRING:
    raise ValueError("SERVICE_BUS_CONNECTION_STRING env var required")
COMMAND_TOPIC_NAME = os.environ.get("COMMAND_TOPIC_NAME", "commandtopic")
COMMAND_SUBSCRIPTION_NAME = os.environ.get("COMMAND_SUBSCRIPTION_NAME", "gradersubscription")
REWARD_TOPIC_NAME  = os.environ.get("REWARD_TOPIC_NAME",  "rewardtopic")
REWARD_SUBSCRIPTION_NAME = os.environ.get("REWARD_SUBSCRIPTION_NAME", "webrewardsubscription")

# LMCache proxy configuration
PROXY_HOST = os.getenv('PROXY_HOST', 'localhost')
PROXY_PORT = os.getenv('PROXY_PORT', '9100')
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
MODEL_NAME = os.getenv('MODEL_NAME', 'gpt-120b-oss')

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
        logger = logging.getLogger("grader_tool")
        if logger.handlers:
            return logger
        logger.setLevel(logging.INFO)
        fmt = logging.Formatter("%(asctime)s %(levelname)s %(message)s")
        h = logging.StreamHandler()
        h.setFormatter(fmt)
        logger.addHandler(h)
        return logger

    app.state.logger = _setup_logger()
    logger = app.state.logger

    # Create HTTP client for communicating with LMCache proxy
    app.state.http_client = httpx.AsyncClient(timeout=300.0)

    logger.info("[startup] Initializing command and reward queues...")
    app.state.cmd_queue = CommandQueue(aio_servicebus_client, COMMAND_TOPIC_NAME, COMMAND_SUBSCRIPTION_NAME)
    await app.state.cmd_queue.start()
    app.state.reward_queue = RewardQueue(aio_servicebus_client, REWARD_TOPIC_NAME)
    app.state._processed_ids = set()

    logger.info(f"[startup] Connected to LMCache proxy at {PROXY_URL}")
    logger.info("[startup] System ready for grading requests.")

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
                        if tool_type != "grader_command":
                            app.state._processed_ids.add(msg_id)
                            continue

                        query = data.get("query")
                        completion = data.get("completion")

                        if query and completion:
                            logger: logging.Logger = getattr(app.state, "logger", logging.getLogger("grader_tool"))
                            logger.info(f"[grader_tool][topic] starting grading for query='{query[:50]}...'")

                            try:
                                # Run grading through LMCache proxy
                                score = await run_grading(query, completion, logger)

                                # Send only the numeric score to the reward queue
                                reward_payload: int = score
                                try:
                                    await app.state.reward_queue.send(reward_payload)
                                except Exception as exc:
                                    logger.exception(f"[reward_topic] failed to publish: {exc}")
                            except Exception as e:
                                logger.exception(f"[grader_tool] error during grading: {e}")

                            app.state._processed_ids.add(msg_id)
                await asyncio.sleep(0.5)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logging.getLogger("grader_tool").exception(f"[worker] error: {e}")
                await asyncio.sleep(1.0)

    app.state.worker = asyncio.create_task(_process_loop(), name="grader-tool-worker")

@app.on_event("shutdown")
async def _shutdown():
    logger = getattr(app.state, "logger", logging.getLogger("grader_tool"))

    # Stop command queue
    cmdq = getattr(app.state, "cmd_queue", None)
    if cmdq:
        await cmdq.stop()

    # Stop background worker
    worker = getattr(app.state, "worker", None)
    if worker:
        worker.cancel()
        try:
            await worker
        except asyncio.CancelledError:
            pass

    # Close HTTP client
    http_client = getattr(app.state, "http_client", None)
    if http_client:
        await http_client.aclose()

    logger.info("[shutdown] All services stopped.")

async def run_grading(query: str, completion: str, logger: logging.Logger) -> int:
    """
    Run the grading process by sending request to LMCache proxy.
    The proxy coordinates prefill and decode servers using NIXL for KV cache transfer.
    """
    # Build grading prompt
    grader_prompt = f"User Query: {query}\n\nCompletion: {completion}\n\nPlease rate this completion's quality from 1 to 5."

    logger.info(f"[grader_tool] Sending request to LMCache proxy at {PROXY_URL}")

    try:
        # Send request to proxy server
        response = await app.state.http_client.post(
            f"{PROXY_URL}/v1/completions",
            json={
                "model": MODEL_NAME,
                "prompt": grader_prompt,
                "max_tokens": 100,
            }
        )
        response.raise_for_status()
        result = response.json()

        # Extract the generated text
        grading_result = result.get('choices', [{}])[0].get('text', '')
        logger.info(f"[grader_tool] Grading result: {grading_result}")

        # Parse and validate the numeric score (1-5)
        import re
        match = re.search(r'\d+', grading_result)
        if not match:
            logger.error(f"[grader_tool] Could not extract number from grading result: {grading_result}")
            raise ValueError(f"Failed to extract numeric score from grading result: {grading_result}")

        score = int(match.group())

        # Validate score is in valid range
        if score < 1 or score > 5:
            logger.error(f"[grader_tool] Score {score} is out of valid range (1-5)")
            raise ValueError(f"Score {score} is out of valid range (1-5). Grading result: {grading_result}")

        logger.info(f"[grader_tool] Final score: {score}")
        return score

    except Exception as e:
        logger.exception(f"[grader_tool] Error communicating with LMCache proxy: {e}")
        raise

@app.get("/")
async def root():
    return {"message": "Grader Environment API", "status": "running"}

@app.get("/health")
async def health():
    try:
        with servicebus_client:
            # Also check if proxy is reachable
            try:
                response = await app.state.http_client.get(f"{PROXY_URL}/health", timeout=5.0)
                proxy_status = "healthy" if response.status_code == 200 else "degraded"
            except:
                proxy_status = "unreachable"

            return {"status": "healthy", "proxy_status": proxy_status}
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
