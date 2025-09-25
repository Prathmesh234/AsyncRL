import os
import json
import asyncio
import logging
from typing import Any, Dict
from uuid import uuid4
from dotenv import load_dotenv
from servicebus_web import ServiceBusTopic

load_dotenv()
logger = logging.getLogger(__name__)

SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")
WEB_TOPIC_NAME = os.getenv("COMMAND_TOPIC_NAME", "commandtopic")  # topic to send commands
REWARD_TOPIC_NAME = os.getenv("REWARD_TOPIC_NAME", "rewardtopic")  # topic to receive results
SUBSCRIPTION_NAME = os.getenv("WEB_SUBSCRIPTION_NAME", "rlcommandbustopic")  # command subscription (if distinct)
REWARD_SUBSCRIPTION_NAME = os.getenv("REWARD_SUBSCRIPTION_NAME")


def send_web_command(payload: Dict[str, Any], timeout_s: int = 10) -> str:
    """Send a web tool command to Service Bus (topic) and wait briefly for a response from reward topic.

    Flow:
      1. Publish a strictly formatted command message to the command topic (WEB_TOPIC_NAME). The AMQP `message_id` is populated for telemetry only.
      2. Poll the reward topic (REWARD_TOPIC_NAME) via its subscription for the first non-placeholder response.
      3. Return the JSON response (stringified) or a fallback message.
    """
    if not SERVICE_BUS_CONNECTION_STRING:
        return "[web-error] Missing SERVICE_BUS_CONNECTION_STRING env var"

    q = str(payload.get("q", "")).strip()
    if not q:
        return "[web-error] Empty web query"
    k_val = payload.get("k", 3)
    try:
        k = int(k_val)
    except (TypeError, ValueError):
        k = 3

    payload_type_raw = str(payload.get("type", "web")).strip().lower()
    payload_type = payload_type_raw or "web"
    message_id = str(uuid4())
    message = {"type": payload_type, "q": q, "k": k}

    try:
        with ServiceBusTopic(SERVICE_BUS_CONNECTION_STRING, topic_name=WEB_TOPIC_NAME, subscription_name=SUBSCRIPTION_NAME) as web_topic:
            ok = web_topic.send_web_result(
                message,
                message_id=message_id,
                wrap=False
            )
            if not ok:
                return "[web-error] Failed to publish web command"
    except Exception as e:
        logger.error(f"Failed sending web command: {e}")
        return f"[web-error] {e}"

    async def _wait_for_response():
        reward_topic = ServiceBusTopic(
            SERVICE_BUS_CONNECTION_STRING,
            topic_name=REWARD_TOPIC_NAME,
            subscription_name=REWARD_SUBSCRIPTION_NAME,
        )
        # Poll up to timeout_s seconds (1s interval)
        for _ in range(timeout_s):
            try:
                resp = await reward_topic.receive_web_reward_async()
                if resp and resp.get("message") not in {"No rewards received", "No messages received"}:
                    return resp
            except Exception as e:  # noqa
                logger.error(f"Error polling reward topic '{REWARD_TOPIC_NAME}/{REWARD_SUBSCRIPTION_NAME}': {e}")
            await asyncio.sleep(1)
        return {"message": "No response within timeout"}

    try:
        response_obj = asyncio.run(_wait_for_response())
        return f"[web-result] {json.dumps(response_obj, ensure_ascii=False)}"
    except RuntimeError:
        # If already inside an event loop, create a new one
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_obj = loop.run_until_complete(_wait_for_response())
            return f"[web-result] {json.dumps(response_obj, ensure_ascii=False)}"
        finally:
            asyncio.set_event_loop(None)
    except Exception as e:  # noqa
        logger.error(f"Failure waiting for web response: {e}")
        return f"[web-error] {e}"
