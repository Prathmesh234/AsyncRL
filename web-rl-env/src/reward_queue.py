import json
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

    async def send(self, reward: Any) -> None:
        """Send a reward payload to the queue. Accepts str/dict/bytes."""
        # Normalize to string payload where possible
        if isinstance(reward, (bytes, bytearray)):
            message = ServiceBusMessage(reward)
        else:
            payload = reward if isinstance(reward, str) else json.dumps(reward)
            message = ServiceBusMessage(payload)

        # Use the long-lived client; do not open/close it here
        sender = self._client.get_queue_sender(queue_name=self._queue)
        async with sender:
            await sender.send_messages(message)
