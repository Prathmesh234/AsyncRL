import asyncio, json, logging
from typing import Optional, Dict, Any
from azure.servicebus.aio import ServiceBusClient

class CommandQueue:
    """Async receiver for grader commands (type: grader_command) with query and completion."""
    def __init__(self, client: ServiceBusClient, topic_name: str, subscription_name: str) -> None:
        self._client = client
        self._topic = topic_name
        self._subscription = subscription_name
        self._task: Optional[asyncio.Task] = None
        self.current_command: Dict[str, Any] = {"status": "idle"}
        self._logger = logging.getLogger("grader_tool")

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="grader-command-subscription-receiver")

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
                    normalized_cmd_type: Optional[str] = None
                    if isinstance(parsed, dict):
                        # Handle both the direct message format and wrapped format
                        if "data" in parsed and isinstance(parsed["data"], dict):
                            # Wrapped format from /receive-command endpoint
                            data = parsed["data"]
                            keys_of_interest = ("type", "query", "completion")
                            subset = {k: data.get(k) for k in keys_of_interest if data.get(k) is not None} or None
                            cmd_type_raw = data.get("type")
                        else:
                            # Direct format (legacy)
                            keys_of_interest = ("type", "query", "completion")
                            subset = {k: parsed.get(k) for k in keys_of_interest if parsed.get(k) is not None} or None
                            cmd_type_raw = parsed.get("type")
                        
                        if isinstance(cmd_type_raw, str):
                            normalized_cmd_type = cmd_type_raw.strip().lower()
                    if normalized_cmd_type != "grader_command":
                        # Ignore silently (still complete message)
                        if self._logger:
                            self._logger.info("Not grader_command, ignored")
                        self.current_command = {"status": "ignored", "tool_type": normalized_cmd_type, "message_id": str(msg.message_id)}
                        await receiver.complete_message(msg)
                        continue

                    if self._logger:
                        self._logger.info("request accepted by grader tool")

                    self.current_command = {
                        "received_command": parsed if parsed is not None else raw_text,
                        "data": subset,
                        "message_id": str(msg.message_id),
                        "raw_content": raw_text,
                        "content_type": "bytes",
                        "tool_type": normalized_cmd_type,
                    }
                    await receiver.complete_message(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self.current_command = {"status": "error", "error": str(e)}
