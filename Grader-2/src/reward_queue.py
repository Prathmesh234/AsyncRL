import json, logging
from typing import Any
from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage

class RewardQueue:
    def __init__(self, client: ServiceBusClient, topic_name: str) -> None:
        self._client = client
        self._topic = topic_name
        self._logger = logging.getLogger("grader_tool")

    async def send(self, reward: Any) -> None:
        if isinstance(reward, (bytes, bytearray)):
            message = ServiceBusMessage(reward)
        else:
            payload = reward if isinstance(reward, str) else json.dumps(reward, ensure_ascii=False)
            message = ServiceBusMessage(payload)
        if self._logger:
            self._logger.info(f"[reward_topic] sending to topic '{self._topic}'")
        sender = self._client.get_topic_sender(topic_name=self._topic)
        async with sender:
            await sender.send_messages(message)
