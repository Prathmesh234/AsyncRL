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
                        if tool_type != "grader_command":
                            app.state._processed_ids.add(msg_id)
                            continue

                        query = data.get("query")
                        completion = data.get("completion")

                        if query and completion:
                            logger: logging.Logger = getattr(app.state, "logger", logging.getLogger("grader_tool"))
                            logger.info(f"[grader_tool][topic] starting grading for query='{query[:50]}...'")

                            try:
                                # Run prefill and decode - returns just the score (1-5)
                                score = await run_grading(query, completion, logger)
                            except Exception as e:
                                score = 3  # Default to middle score on error
                                logger.exception(f"[grader_tool] error during grading: {e}")

                            # Send only the numeric score to the reward queue
                            reward_payload: int = score
                            try:
                                await app.state.reward_queue.send(reward_payload)
                            except Exception as exc:
                                logger.exception(f"[reward_topic] failed to publish: {exc}")
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

async def run_grading(query: str, completion: str, logger: logging.Logger) -> int:
    """
    Run the grading process using prefill and decode scripts.
    """
    # Set environment variable with the full prompt (query + completion)
    grader_prompt = f"User Query: {query}\n\nCompletion: {completion}\n\nPlease rate this completion's quality from 1 to 5."

    # Run prefill script
    logger.info(f"[grader_tool] Running prefill")
    prefill_process = await asyncio.create_subprocess_exec(
        "python", "/app/prefill_disaggregated.py",
        env={**os.environ, "GRADER_PROMPT": grader_prompt},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    prefill_stdout, prefill_stderr = await prefill_process.communicate()

    if prefill_process.returncode != 0:
        logger.error(f"[grader_tool] Prefill failed: {prefill_stderr.decode()}")
        raise RuntimeError(f"Prefill failed: {prefill_stderr.decode()}")

    # Extract cache_uri from prefill output
    prefill_output = prefill_stdout.decode()
    logger.info(f"[grader_tool] Prefill output: {prefill_output}")

    # Parse cache_uri from output (looking for "Final cache URI: ...")
    cache_uri = None
    for line in prefill_output.split('\n'):
        if "Final cache URI:" in line:
            cache_uri = line.split("Final cache URI:")[-1].strip()
            break

    if not cache_uri:
        raise RuntimeError("Could not extract cache_uri from prefill output")

    logger.info(f"[grader_tool] Cache URI: {cache_uri}")

    # Run decode script
    logger.info(f"[grader_tool] Running decode")
    decode_process = await asyncio.create_subprocess_exec(
        "python", "/app/decode_disaggregated.py",
        "--cache-uri", cache_uri,
        env={**os.environ, "GRADER_PROMPT": grader_prompt},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    decode_stdout, decode_stderr = await decode_process.communicate()

    if decode_process.returncode != 0:
        logger.error(f"[grader_tool] Decode failed: {decode_stderr.decode()}")
        raise RuntimeError(f"Decode failed: {decode_stderr.decode()}")

    decode_output = decode_stdout.decode()
    logger.info(f"[grader_tool] Decode output: {decode_output}")

    # Extract generated text from decode output
    grading_result = None
    for line in decode_output.split('\n'):
        if "->" in line:
            grading_result = line.split("->")[-1].strip()
            break

    if not grading_result:
        grading_result = decode_output.strip()

    # Parse and validate the numeric score (1-5)
    try:
        # Extract only digits from the result
        import re
        match = re.search(r'\d+', grading_result)
        if match:
            score = int(match.group())
            # Clamp score to 1-5 range
            score = max(1, min(5, score))
        else:
            logger.warning(f"[grader_tool] Could not extract number from: {grading_result}, defaulting to 3")
            score = 3
    except (ValueError, AttributeError) as e:
        logger.warning(f"[grader_tool] Error parsing score: {e}, defaulting to 3")
        score = 3

    logger.info(f"[grader_tool] Final score: {score}")

    return score

@app.get("/")
async def root():
    return {"message": "Grader Environment API", "status": "running"}

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
