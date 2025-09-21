import json
import logging
from typing import Any

from azure.servicebus.aio import ServiceBusClient
from azure.servicebus import ServiceBusMessage


class RewardQueue:
    """Minimal async sender for rewards (topic based)."""

    def __init__(self, client: ServiceBusClient, topic_name: str) -> None:
        self._client = client
        self._topic = topic_name
        self._logger = logging.getLogger("web_tool")

    async def send(self, reward: Any) -> None:
        """Send a reward payload to the topic. Accepts str/dict/bytes.

        Logs a concise summary (type/query/result_count) rather than payload contents.
        """
        # Normalize to string payload where possible
        if isinstance(reward, (bytes, bytearray)):
            message = ServiceBusMessage(reward)
            summary = f"bytes={len(reward)}"
        else:
            payload = reward if isinstance(reward, str) else json.dumps(reward, ensure_ascii=False)
            message = ServiceBusMessage(payload)
            if isinstance(reward, dict):
                q = reward.get("query")
                count = len(reward.get("results") or []) if isinstance(reward.get("results"), list) else 0
                rtype = reward.get("type") or "reward"
                qprev = (q if isinstance(q, str) else str(q)) if q is not None else ""
                if len(qprev) > 120:
                    qprev = qprev[:117] + "..."
                summary = f"type={rtype} query={qprev!r} result_count={count}"
            else:
                summary = f"text_len={len(payload)}"

        telemetry = f"[reward_topic] sending results to topic '{self._topic}' {summary}"
        print(telemetry)
        if self._logger:
            self._logger.info(telemetry)

        # Topic sender
        sender = self._client.get_topic_sender(topic_name=self._topic)
        try:
            async with sender:
                await sender.send_messages(message)
        except Exception:
            if self._logger:
                self._logger.exception("[reward_topic] failed to send message")
            raise
        else:
            if self._logger:
                self._logger.info(f"[reward_topic] delivered to topic '{self._topic}'")
