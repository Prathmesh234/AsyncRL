import asyncio
import json
from pathlib import Path
from typing import Any, Dict


class AzureTool:
    """Runs Azure CLI commands and returns their output."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self._log_path = base_dir / "logger" / "azurelogging.txt"

    async def run(self, command: str) -> Dict[str, Any]:
        azure_command = (command or "").strip()
        if not azure_command:
            result = {
                "status": "error",
                "azure_command": azure_command,
                "response": "No azure command provided",
            }
            self._log({"event": "response", "data": result})
            return result

        self._log({"event": "received", "azure_command": azure_command})

        process = await asyncio.create_subprocess_shell(
            azure_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        stdout_bytes, stderr_bytes = await process.communicate()

        if not stdout_bytes and not stderr_bytes:
            await asyncio.sleep(2)

        stdout_text = stdout_bytes.decode("utf-8", errors="replace").strip()
        stderr_text = stderr_bytes.decode("utf-8", errors="replace").strip()

        if process.returncode == 0:
            status = "ok"
            response_text = stdout_text or "Command completed with no output"
        else:
            status = "error"
            response_text = stderr_text or stdout_text or f"Command failed with exit code {process.returncode}"

        result = {
            "status": status,
            "azure_command": azure_command,
            "response": response_text,
        }

        self._log({"event": "response", "data": result})
        return result

    def _log(self, payload: Dict[str, Any]) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
