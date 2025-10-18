import os
import json
import asyncio
import logging
from uuid import uuid4
from dotenv import load_dotenv
from servicebus_web import ServiceBusTopic

load_dotenv()
logger = logging.getLogger(__name__)

SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")
COMMAND_TOPIC_NAME = os.getenv("COMMAND_TOPIC_NAME", "commandtopic")  # topic to send commands
REWARD_TOPIC_NAME = os.getenv("REWARD_TOPIC_NAME", "rewardtopic")  # topic to receive results
GRADER_SUBSCRIPTION_NAME = os.getenv("GRADER_SUBSCRIPTION_NAME", "gradersubscription")
GRADER_REWARD_SUBSCRIPTION_NAME = os.getenv("GRADER_REWARD_SUBSCRIPTION_NAME", "webrewardsubscription")


def send_grader_command(query: str, completion: str, timeout_s: int = 30) -> str:
    """Send a grader command to Service Bus (topic) and wait for numeric score from reward topic.

    Flow:
      1. Publish a grader_command message with query and completion to the command topic.
      2. Poll the reward topic via its subscription for the numeric score response.
      3. Return the score or a fallback message.

    Args:
        query: The user query/prompt
        completion: The model's completion
        timeout_s: Timeout in seconds to wait for grading response (default 30s)

    Returns:
        String containing the grading score or error message
    """
    if not SERVICE_BUS_CONNECTION_STRING:
        return "[grader-error] Missing SERVICE_BUS_CONNECTION_STRING env var"

    if not query or not completion:
        return "[grader-error] Both query and completion are required"

    message_id = str(uuid4())
    message = {
        "type": "grader_command",
        "query": query,
        "completion": completion
    }

    try:
        with ServiceBusTopic(
            SERVICE_BUS_CONNECTION_STRING,
            topic_name=COMMAND_TOPIC_NAME,
            subscription_name=GRADER_SUBSCRIPTION_NAME
        ) as grader_topic:
            ok = grader_topic.send_web_result(
                message,
                message_id=message_id,
                wrap=False
            )
            if not ok:
                return "[grader-error] Failed to publish grader command"
    except Exception as e:
        logger.error(f"Failed sending grader command: {e}")
        return f"[grader-error] {e}"

    async def _wait_for_response():
        reward_topic = ServiceBusTopic(
            SERVICE_BUS_CONNECTION_STRING,
            topic_name=REWARD_TOPIC_NAME,
            subscription_name=GRADER_REWARD_SUBSCRIPTION_NAME,
        )
        # Poll up to timeout_s seconds (1s interval)
        for _ in range(timeout_s):
            try:
                resp = await reward_topic.receive_web_reward_async()
                if resp and resp.get("message") not in {"No rewards received", "No messages received"}:
                    return resp
            except Exception as e:  # noqa
                logger.error(f"Error polling reward topic '{REWARD_TOPIC_NAME}/{GRADER_REWARD_SUBSCRIPTION_NAME}': {e}")
            await asyncio.sleep(1)
        return {"message": "No response within timeout"}

    try:
        response_obj = asyncio.run(_wait_for_response())
        # The grader returns a numeric score (1-5), extract it
        if isinstance(response_obj, dict):
            score = response_obj.get("message")
            if isinstance(score, (int, float)):
                return f"[grader-score] {score}"
            else:
                return f"[grader-result] {json.dumps(response_obj, ensure_ascii=False)}"
        else:
            # Direct numeric response
            return f"[grader-score] {response_obj}"
    except RuntimeError:
        # If already inside an event loop, create a new one
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_obj = loop.run_until_complete(_wait_for_response())
            if isinstance(response_obj, dict):
                score = response_obj.get("message")
                if isinstance(score, (int, float)):
                    return f"[grader-score] {score}"
                else:
                    return f"[grader-result] {json.dumps(response_obj, ensure_ascii=False)}"
            else:
                return f"[grader-score] {response_obj}"
        finally:
            asyncio.set_event_loop(None)
    except Exception as e:  # noqa
        logger.error(f"Failure waiting for grader response: {e}")
        return f"[grader-error] {e}"
