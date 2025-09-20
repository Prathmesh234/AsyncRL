import asyncio
import json
import logging
from typing import Optional, Dict, Any

from azure.servicebus.aio import ServiceBusClient


class CommandQueue:
    """Very small async receiver that pushes messages as they arrive.

    Methods: start(), stop(); background loop reads messages and updates
    current_command.
    """

    def __init__(self, client: ServiceBusClient, queue_name: str) -> None:
        self._client = client
        self._queue = queue_name
        self._task: Optional[asyncio.Task] = None
        self.current_command: Dict[str, Any] = {"status": "idle"}
        self._logger = logging.getLogger("web_tool")

    async def start(self) -> None:
        if self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._run(), name="command-queue-receiver")

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
            # Use long-lived client; create a receiver context that stays open
            receiver = self._client.get_queue_receiver(queue_name=self._queue)
            async with receiver:
                async for msg in receiver:
                        body_bytes = b"".join(
                            part if isinstance(part, (bytes, bytearray)) else bytes(part)
                            for part in msg.body
                        )
                        try:
                            raw_text = body_bytes.decode("utf-8", errors="strict")
                        except Exception:
                            raw_text = body_bytes.decode("utf-8", errors="replace")

                        try:
                            parsed = json.loads(raw_text)
                        except Exception:
                            parsed = None

                        # Preserve a subset of relevant fields for the monitoring endpoints.
                        subset: Optional[Dict[str, Any]] = None
                        normalized_type: Optional[str] = None
                        if isinstance(parsed, dict):
                            keys_of_interest = (
                                "type",
                                "q",
                                "k",
                                "code_command",
                                "azure_command",
                                "request_id",
                            )
                            subset = {
                                key: parsed.get(key)
                                for key in keys_of_interest
                                if parsed.get(key) is not None
                            } or None
                            tool_type_raw = parsed.get("type")
                            if isinstance(tool_type_raw, str):
                                normalized_type = tool_type_raw.strip().lower()
                            log_msg = (
                                "request accepted by web"
                                if normalized_type == "web"
                                else "request rejected by web"
                            )
                            if self._logger:
                                self._logger.info(log_msg)
                            else:
                                print(log_msg)
                        else:
                            log_msg = "request rejected by web"
                            if self._logger:
                                self._logger.info(log_msg)
                            else:
                                print(log_msg)

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
            # Graceful shutdown
            raise
        except Exception as e:
            # Surface receiver issues via the endpoint for quick debugging
            self.current_command = {"status": "error", "error": str(e)}
