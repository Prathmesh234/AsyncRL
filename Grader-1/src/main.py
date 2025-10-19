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

# Grader-1 proxy configuration
PROXY_HOST = os.getenv('PROXY_HOST', 'localhost')
PROXY_PORT = os.getenv('PROXY_PORT', '8000')
PROXY_URL = f"http://{PROXY_HOST}:{PROXY_PORT}"
MODEL_NAME = os.getenv('MODEL_NAME', './gpt-oss-20b')
SYSTEM_PROMPT = os.getenv('SYSTEM_PROMPT')
if not SYSTEM_PROMPT:
    raise ValueError("SYSTEM_PROMPT env var required")

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

    # Create HTTP client for communicating with disaggregated proxy
    app.state.http_client = httpx.AsyncClient(timeout=300.0)

    logger.info("[startup] Initializing command and reward queues...")
    app.state.cmd_queue = CommandQueue(aio_servicebus_client, COMMAND_TOPIC_NAME, COMMAND_SUBSCRIPTION_NAME)
    await app.state.cmd_queue.start()
    app.state.reward_queue = RewardQueue(aio_servicebus_client, REWARD_TOPIC_NAME)
    app.state._processed_ids = set()

    logger.info(f"[startup] Connected to disaggregated proxy at {PROXY_URL}")
    logger.info("[startup] System ready for grading requests.")

    async def _process_loop():
        logger = app.state.logger  # Get logger at the start of the function
        idle_logged = False
        while True:
            try:
                cmdq = getattr(app.state, "cmd_queue", None)
                payload = cmdq.current_command if cmdq else None
                if isinstance(payload, dict):
                    msg_id = payload.get("message_id")
                    data = payload.get("data")
                    
                    # Reset idle flag when we have a payload to process
                    idle_logged = False
                    
                    if msg_id and msg_id not in app.state._processed_ids and isinstance(data, dict):
                        cmd_type = data.get("type")  # Look for "type" field
                        
                        if isinstance(cmd_type, str):
                            cmd_type = cmd_type.strip().lower()
                        if cmd_type != "grader_command":
                            app.state._processed_ids.add(msg_id)
                            continue

                        query = data.get("query")
                        completion = data.get("completion")

                        if query and completion:
                            logger.info(f"[grader_tool] Starting grading for query: '{query[:50]}...'")
                            
                            # Mark as processed immediately to prevent reprocessing
                            app.state._processed_ids.add(msg_id)

                            try:
                                # Run grading through disaggregated proxy
                                score = await run_grading(query, completion, logger)
                                
                                if score is not None:
                                    # Send the numeric score to the reward queue
                                    try:
                                        await app.state.reward_queue.send(score)
                                        logger.info(f"[grader_tool] Score {score} sent to reward queue")
                                    except Exception as exc:
                                        logger.exception(f"[reward_topic] failed to publish score: {exc}")
                                else:
                                    print("ERRRRRORRRR: Grading failed, no score will be sent to reward queue")
                            except Exception as e:
                                print("ERRRRRORRRR: Exception during grading process")
                                logger.exception(f"[grader_tool] Error during grading: {e}")
                                # DO NOT send any default score - this is a security vulnerability
                        else:
                            logger.debug(f"[grader_tool] Missing query or completion in message")
                            app.state._processed_ids.add(msg_id)
                    elif msg_id and msg_id in app.state._processed_ids:
                        # Don't log every processed message to reduce spam
                        pass
                else:
                    # Log idle state only once to avoid spam
                    if not idle_logged:
                        logger.debug("[grader_tool] No new messages, worker idle")
                        idle_logged = True
                        
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

async def run_grading(query: str, completion: str, logger: logging.Logger) -> Optional[int]:
    """
    Run the grading process by sending request to disaggregated proxy.
    The proxy coordinates prefill and decode servers for disaggregated inference.
    """
    # Build grading prompt for agent trajectory evaluation
    user_prompt = f"User Query: {query}\n\nAgent Trajectory/Completion: {completion}\n\nPlease evaluate this agent trajectory and provide your rating."

    logger.info(f"[grader_tool] Sending request to disaggregated proxy at {PROXY_URL}")

    try:
        # Send request to proxy server using chat completions format with system prompt
        response = await app.state.http_client.post(
            f"{PROXY_URL}/v1/chat/completions",
            json={
                "model": MODEL_NAME,
                "messages": [
                    {
                        "role": "system",
                        "content": SYSTEM_PROMPT
                    },
                    {
                        "role": "user", 
                        "content": user_prompt
                    }
                ],
                "max_tokens": 8000,  # Only need a single digit for grading
                "temperature": 0.3  # Low temperature for consistent grading
            }
        )
        response.raise_for_status()
        result = response.json()

        # Extract the generated text from chat completion response
        grading_result = result.get('choices', [{}])[0].get('message', {}).get('content', '')
        
        # Always log what we got from the model - this is the key output for debugging
        print(f"Model output: '{result}'")

        # Check if grading_result is None or not a string
        if grading_result is None:
            print("ERRRRRORRRR: Model returned None as content")
            logger.error(f"[grader_tool] Model returned None as content")
            return None
            
        if not isinstance(grading_result, str):
            print(f"ERRRRRORRRR: Model returned non-string content: {type(grading_result)}")
            logger.error(f"[grader_tool] Model returned non-string content: {type(grading_result)}")
            return None

        # Parse and validate the numeric score (1-5)
        import re
        match = re.search(r'\d+', grading_result)
        if not match:
            print("ERRRRRORRRR: Could not extract number from grading result")
            print(f"Model output: '{grading_result}'")
            logger.error(f"[grader_tool] Could not extract number from grading result: '{grading_result}'")
            return None

        score = int(match.group())

        # Validate score is in valid range
        if score < 1 or score > 5:
            print("ERRRRRORRRR: Score out of valid range (1-5)")
            print(f"Model output: '{grading_result}'")
            print(f"Extracted score: {score}")
            logger.error(f"[grader_tool] Score {score} is out of valid range (1-5)")
            return None

        print(f"Model output: '{grading_result}'")
        logger.info(f"[grader_tool] Final score: {score}")
        return score

    except httpx.HTTPError as e:
        print("ERRRRRORRRR: HTTP ERROR FROM DECODE GPU!")
        print(f"HTTP Status: {e.response.status_code if e.response else 'UNKNOWN'}")
        if e.response:
            try:
                response_text = e.response.text
                print(f"Response: {response_text}")
            except:
                print("Could not read response body")
        logger.error(f"[grader_tool] HTTP error from vLLM server: {e}")
        return None
    except Exception as e:
        print("ERRRRRORRRR: Unexpected error during grading")
        print(f"Error: {str(e)}")
        logger.exception(f"[grader_tool] Unexpected error during grading: {e}")
        return None

@app.get("/")
async def root():
    return {"message": "Grader Environment API", "status": "running"}

@app.get("/health")
async def health():
    try:
        with servicebus_client:
            # Also check if proxy is reachable
            try:
                response = await app.state.http_client.get(f"{PROXY_URL}/status", timeout=5.0)
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
        
        # Convert command_type field to "type" in the message payload
        message_payload = {
            "message_id": msg_id,
            "data": {
                "type": payload_dict.get("command_type", "grader_command"),
                "query": payload_dict.get("query"),
                "completion": payload_dict.get("completion")
            }
        }
        
        payload = json.dumps(message_payload)
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
