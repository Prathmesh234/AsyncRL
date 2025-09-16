import os
import json
import asyncio
import logging
from typing import Any, Dict
from uuid import uuid4
from dotenv import load_dotenv
from servicebus_web import ServiceBusQueueWeb

load_dotenv()
logger = logging.getLogger(__name__)

SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")
WEB_QUEUE_NAME = os.getenv("QUEUE_NAME", "commandqueue")  # queue to send commands
REWARD_QUEUE_NAME = os.getenv("REWARD_QUEUE_NAME", "rewardqueue")  # queue to receive results


def send_web_command(payload: Dict[str, Any], timeout_s: int = 10) -> str:
    """Send a web tool command to Service Bus and wait briefly for a response from reward queue.

    Flow:
      1. Enqueue command message to command queue (WEB_QUEUE_NAME)
      2. Poll reward queue (REWARD_QUEUE_NAME) for matching request_id
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

    request_id = str(uuid4())
    message = {"q": q, "k": k, "request_id": request_id, "type": "web_command"}

    try:
        # Send request (re-using send_web_result for simplicity)
        with ServiceBusQueueWeb(SERVICE_BUS_CONNECTION_STRING, queue_name=WEB_QUEUE_NAME) as web_queue:
            ok = web_queue.send_web_result(message, request_id=request_id)
            if not ok:
                return "[web-error] Failed to enqueue web command"
    except Exception as e:
        logger.error(f"Failed sending web command: {e}")
        return f"[web-error] {e}"

    async def _wait_for_response():
        # Poll up to timeout_s seconds (1s interval)
        for _ in range(timeout_s):
            try:
                reward_queue = ServiceBusQueueWeb(SERVICE_BUS_CONNECTION_STRING, queue_name=REWARD_QUEUE_NAME)
                resp = await reward_queue.receive_web_reward_async()
                if resp and resp.get("message") not in {"No rewards received", "No messages received"}:
                    # Try to correlate by request_id if present
                    resp_request_id = resp.get("request_id") or resp.get("data", {}).get("request_id")
                    if resp_request_id is None or resp_request_id == request_id:
                        return resp
            except Exception as e:  # noqa
                logger.error(f"Error polling reward queue: {e}")
            await asyncio.sleep(1)
        return {"message": "No response within timeout", "request_id": request_id}

    try:
        response_obj = asyncio.run(_wait_for_response())
        # Expected reward queue payload example (response_obj):
        # {
        #   "message_id": "4ab73cea-...",
        #   "q": "Sora api documentation azure",   # original query
        #   "k": 3,                                 # requested top-k
        #   "results": [                            # search results list
        #       {
        #         "url": "https://...",
        #         "title": "Quickstart: Generate video with Sora - Azure OpenAI | Microsoft Learn",
        #         "content": "Long snippet text ..."
        #       },
        #       {"url": "https://...", "title": "SORA API on Azure ...", "content": "..."},
        #       {"url": "https://...", "title": "How to use the Sora API ...", "content": "..."}
        #   ]
        # }
        # We wrap the entire object as the tool result string below.
        return f"[web-result] {json.dumps(response_obj, ensure_ascii=False)}"
    except RuntimeError:
        # If already inside an event loop, create a new one
        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            response_obj = loop.run_until_complete(_wait_for_response())
            # Same structure described above for response_obj
            return f"[web-result] {json.dumps(response_obj, ensure_ascii=False)}"
        finally:
            asyncio.set_event_loop(None)
    except Exception as e:  # noqa
        logger.error(f"Failure waiting for web response: {e}")
        return f"[web-error] {e}"
