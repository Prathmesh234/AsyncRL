from __future__ import annotations

import json
import shlex
from typing import Any, Dict, List, Union, Optional
from pydantic import BaseModel, Field, ValidationError, field_validator


class WebCommand(BaseModel):
    q: str = Field(..., min_length=1, description="Search query text")
    k: int = Field(3, ge=1, le=10, description="Top-K results to return")

    @classmethod
    def coerce(cls, payload: Union[str, Dict[str, Any]], default_k: int = 3) -> "WebCommand":
        # If string, try JSON first; otherwise treat as raw query
        if isinstance(payload, str):
            text = payload.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    data = json.loads(text)
                    return cls(q=str(data.get("q", "")).strip(), k=int(data.get("k", default_k)))
                except Exception:
                    pass
            return cls(q=text, k=default_k)
        elif isinstance(payload, dict):
            q = payload.get("q") or payload.get("query") or payload.get("raw_content")
            k = payload.get("k", default_k)
            return cls(q=str(q or "").strip(), k=int(k))
        else:
            raise ValidationError(["Unsupported payload type for WebCommand"])


class CodeCommand(BaseModel):
    cmd: str = Field(..., min_length=1, description="Shell command to run")
    cwd: Optional[str] = Field(None, description="Working directory")
    timeout_s: int = Field(15, ge=1, le=900, description="Timeout in seconds")

    @classmethod
    def coerce(cls, payload: Union[str, Dict[str, Any]], default_timeout: int = 15) -> "CodeCommand":
        if isinstance(payload, str):
            text = payload.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    data = json.loads(text)
                    return cls(
                        cmd=str(data.get("cmd", "")).strip(),
                        cwd=data.get("cwd"),
                        timeout_s=int(data.get("timeout_s", default_timeout)),
                    )
                except Exception:
                    pass
            return cls(cmd=text, timeout_s=default_timeout)
        elif isinstance(payload, dict):
            return cls(
                cmd=str(payload.get("cmd", "")).strip(),
                cwd=payload.get("cwd"),
                timeout_s=int(payload.get("timeout_s", default_timeout)),
            )
        else:
            raise ValidationError(["Unsupported payload type for CodeCommand"])


class AzureCommand(BaseModel):
    args: List[str] = Field(..., min_length=1, description="Azure CLI arguments list")

    @field_validator("args")
    @classmethod
    def _strip_args(cls, v: List[str]) -> List[str]:
        return [str(a).strip() for a in v if str(a).strip()]

    @classmethod
    def coerce(cls, payload: Union[str, Dict[str, Any]]) -> "AzureCommand":
        if isinstance(payload, str):
            text = payload.strip()
            if text.startswith("{") and text.endswith("}"):
                try:
                    data = json.loads(text)
                    args = data.get("args")
                    if isinstance(args, list):
                        return cls(args=[str(a) for a in args])
                except Exception:
                    pass
            # best-effort split into tokens
            return cls(args=shlex.split(text))
        elif isinstance(payload, dict):
            args = payload.get("args")
            if isinstance(args, list):
                return cls(args=[str(a) for a in args])
            # fallback if someone used raw_content
            rc = payload.get("raw_content")
            if isinstance(rc, str):
                return cls(args=shlex.split(rc))
            raise ValidationError(["Missing 'args' for AzureCommand"])
        else:
            raise ValidationError(["Unsupported payload type for AzureCommand"])


def ensure_web_payload(payload: Union[str, Dict[str, Any]], default_k: int = 3) -> Dict[str, Any]:
    model = WebCommand.coerce(payload, default_k=default_k)
    return model.model_dump()


def ensure_code_payload(payload: Union[str, Dict[str, Any]], default_timeout: int = 15) -> Dict[str, Any]:
    model = CodeCommand.coerce(payload, default_timeout=default_timeout)
    return model.model_dump()


def ensure_azure_payload(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    model = AzureCommand.coerce(payload)
    return model.model_dump()

