import asyncio, json, logging
from typing import Optional, Dict, Any
from azure.servicebus.aio import ServiceBusClient

class CommandQueue:
    """Async receiver identical in shape to web-rl-env but filtering for azure type (topic/subscription)."""
    def __init__(self, client: ServiceBusClient, topic_name: str, subscription_name: str) -> None:
        self._client = client
        self._topic = topic_name
        self._subscription = subscription_name
        self._task: Optional[asyncio.Task] = None
        self.current_command: Dict[str, Any] = {"status": "idle"}
        self._logger = logging.getLogger("azure_tool")

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="azure-command-subscription-receiver")

    async def stop(self) -> None:
        if not self._task:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def _run(self) -> None:
        try:
            receiver = self._client.get_subscription_receiver(topic_name=self._topic, subscription_name=self._subscription)
            async with receiver:
                async for msg in receiver:
                    body_bytes = b"".join(part if isinstance(part, (bytes, bytearray)) else bytes(part) for part in msg.body)
                    try:
                        raw_text = body_bytes.decode("utf-8", errors="strict")
                    except Exception:
                        raw_text = body_bytes.decode("utf-8", errors="replace")
                    try:
                        parsed = json.loads(raw_text)
                    except Exception:
                        parsed = None

                    subset: Optional[Dict[str, Any]] = None
                    normalized_type: Optional[str] = None
                    if isinstance(parsed, dict):
                        keys_of_interest = ("type", "azure_command", "request_id")
                        subset = {k: parsed.get(k) for k in keys_of_interest if parsed.get(k) is not None} or None
                        tool_type_raw = parsed.get("type")
                        if isinstance(tool_type_raw, str):
                            normalized_type = tool_type_raw.strip().lower()
                    if normalized_type != "azure":
                        # Ignore silently (still complete message)
                        if self._logger:
                            self._logger.info("Not azure command, ignored")
                        self.current_command = {"status": "ignored", "tool_type": normalized_type, "message_id": str(msg.message_id)}
                        await receiver.complete_message(msg)
                        continue

                    if self._logger:
                        self._logger.info("request accepted by azure tool")

                    self.current_command = {
                        "received_command": parsed if parsed is not None else raw_text,
                        "data": subset,
                        "message_id": str(msg.message_id),
                        "raw_content": raw_text,
                        "content_type": "bytes",
                        "tool_type": normalized_type,
                    }
                    await receiver.complete_message(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.current_command = {"status": "error", "error": str(e)}
