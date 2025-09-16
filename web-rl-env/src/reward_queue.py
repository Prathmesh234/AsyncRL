import json
import logging
from typing import Any

from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage


class RewardQueue:
    """Minimal async sender for rewards.

    Structure mirrors CommandQueue: constructed with the async client and
    queue name; exposes a single async method to send a reward.
    """

    def __init__(self, client: ServiceBusClient, queue_name: str) -> None:
        self._client = client
        self._queue = queue_name
        self._logger = logging.getLogger("web_tool")

    async def send(self, reward: Any) -> None:
        """Send a reward payload to the queue. Accepts str/dict/bytes."""
        # Normalize to string payload where possible
        if isinstance(reward, (bytes, bytearray)):
            message = ServiceBusMessage(reward)
            preview = f"{len(reward)} bytes"
        else:
            payload = reward if isinstance(reward, str) else json.dumps(reward, ensure_ascii=False)
            message = ServiceBusMessage(payload)
            preview = (payload if isinstance(payload, str) else str(payload))[:500]

        telemetry = f"[reward_queue] sending message to '{self._queue}' preview={preview}"
        print(telemetry)
        if self._logger:
            self._logger.info(telemetry)

        # Use the long-lived client; do not open/close it here
        sender = self._client.get_queue_sender(queue_name=self._queue)
        try:
            async with sender:
                await sender.send_messages(message)
        except Exception:
            if self._logger:
                self._logger.exception("[reward_queue] failed to send message")
            raise
        else:
            if self._logger:
                self._logger.info(f"[reward_queue] message delivered to '{self._queue}'")
