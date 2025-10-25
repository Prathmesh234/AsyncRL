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
SUBSCRIPTION_NAME = os.getenv("WEB_SUBSCRIPTION_NAME", "azuresubscription")


def send_azure_command(payload: Dict[str, Any], timeout_s: int = 10) -> str:
    """Send an azure tool command to Service Bus (topic) and await reward topic response."""
    if not SERVICE_BUS_CONNECTION_STRING:
        return "[azure-error] Missing SERVICE_BUS_CONNECTION_STRING env var"

    command = str(payload.get("azure_command", "")).strip()
    if not command:
        return "[azure-error] Empty azure command"

    payload_type_raw = str(payload.get("type", "azure")).strip().lower()
    payload_type = payload_type_raw or "azure"
    message_id = str(uuid4())
    message = {"type": payload_type, "azure_command": command}

    try:
        with ServiceBusTopic(SERVICE_BUS_CONNECTION_STRING, topic_name=WEB_TOPIC_NAME, subscription_name=SUBSCRIPTION_NAME) as web_topic:
            ok = web_topic.send_web_result(
                message,
                message_id=message_id,
                wrap=False
            )
            if not ok:
                return "[azure-error] Failed to publish azure command"
    except Exception as e:
        logger.error(f"Failed sending azure command: {e}")
        return f"[azure-error] {e}"

    async def _wait_for_response():
        for _ in range(timeout_s):
            try:
                reward_topic = ServiceBusTopic(SERVICE_BUS_CONNECTION_STRING, topic_name=REWARD_TOPIC_NAME, subscription_name=SUBSCRIPTION_NAME)
                resp = await reward_topic.receive_web_reward_async()
                if resp and resp.get("message") not in {"No rewards received", "No messages received"}:
                    return resp
            except Exception as e:  # noqa
                logger.error(f"Error polling reward topic: {e}")
            await asyncio.sleep(1)
        return {"message": "No response within timeout"}

    try:
        response_obj = asyncio.run(_wait_for_response())
        return f"[azure-result] {json.dumps(response_obj, ensure_ascii=False)}"
    except RuntimeError:
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_obj = loop.run_until_complete(_wait_for_response())
            return f"[azure-result] {json.dumps(response_obj, ensure_ascii=False)}"
        finally:
            asyncio.set_event_loop(None)
    except Exception as e:  # noqa
        logger.error(f"Failure waiting for azure response: {e}")
        return f"[azure-error] {e}"
