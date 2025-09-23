import asyncio
import json
import os
from pathlib import Path
from typing import Any, Dict


class CodeTool:
    """Runs shell commands inside the code workspace and returns their output."""

    def __init__(self) -> None:
        base_dir = Path(__file__).resolve().parent.parent
        self._log_path = base_dir / "logger" / "codelogging.txt"
        workspace_env = os.environ.get("CODE_WORKSPACE_DIR", "/app/projects")
        workspace_path = Path(workspace_env).expanduser()
        workspace_path.mkdir(parents=True, exist_ok=True)
        try:
            self._workspace_path = workspace_path.resolve()
        except Exception:
            self._workspace_path = workspace_path

    async def run(self, command: str) -> Dict[str, Any]:
        code_command = (command or "").strip()
        if not code_command:
            result = {
                "status": "error",
                "code_command": code_command,
                "response": "No code command provided",
            }
            self._log({"event": "response", "data": result})
            return result

        self._log(
            {
                "event": "received",
                "code_command": code_command,
                "workspace": str(self._workspace_path),
            }
        )

        process = await asyncio.create_subprocess_shell(
            code_command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(self._workspace_path),
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
            "code_command": code_command,
            "response": response_text,
            "working_directory": str(self._workspace_path),
        }

        self._log({"event": "response", "data": result})
        return result

    def _log(self, payload: Dict[str, Any]) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        with self._log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(json.dumps(payload, ensure_ascii=False) + "\n")
