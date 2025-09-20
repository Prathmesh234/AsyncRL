from __future__ import annotations

import json
from typing import Any, Dict, Literal, Union

from pydantic import BaseModel, Field, ValidationError


def _as_dict(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Coerce a tool payload into a dictionary."""
    if isinstance(payload, dict):
        return payload
    if isinstance(payload, str):
        text = payload.strip()
        if not text:
            raise ValueError("Tool payload cannot be empty")
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            decoder = json.JSONDecoder()
            search_pos = 0
            while True:
                brace_idx = text.find('{', search_pos)
                if brace_idx == -1:
                    raise ValueError("Tool payload must be JSON: could not locate an object literal")
                try:
                    data, offset = decoder.raw_decode(text, brace_idx)
                except json.JSONDecodeError:
                    search_pos = brace_idx + 1
                    continue
                if isinstance(data, dict):
                    return data
                search_pos = brace_idx + max(1, offset - brace_idx)
            raise ValueError("Tool payload must decode to an object")
        if not isinstance(data, dict):
            raise ValueError("Tool payload must decode to an object")
        return data
    raise ValueError("Tool payload must be a dict or JSON object string")


class WebCommand(BaseModel):
    type: Literal["web"] = Field(..., description="Tool discriminator for web searches")
    q: str = Field(..., min_length=1, description="Search query text")
    k: int = Field(3, ge=1, le=10, description="Top-K results to return")

    @classmethod
    def coerce(cls, payload: Union[str, Dict[str, Any]], default_k: int = 3) -> "WebCommand":
        data = _as_dict(payload)
        type_raw = data.get("type")
        if not isinstance(type_raw, str) or type_raw.strip().lower() != "web":
            raise ValueError('Tool payload must include "type": "web"')
        allowed_keys = {"type", "q", "k"}
        extras = set(data.keys()) - allowed_keys
        if extras:
            raise ValueError(f"Unexpected fields for web payload: {sorted(extras)}")
        q = str(data.get("q", "")).strip()
        k_raw = data.get("k", default_k)
        try:
            k = int(k_raw)
        except (TypeError, ValueError):
            k = default_k
        try:
            return cls(type="web", q=q, k=k)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc


class CodeCommand(BaseModel):
    type: Literal["code"] = Field(..., description="Tool discriminator for code execution")
    code_command: str = Field(..., min_length=1, description="Code command string")

    @classmethod
    def coerce(cls, payload: Union[str, Dict[str, Any]]) -> "CodeCommand":
        data = _as_dict(payload)
        type_raw = data.get("type")
        if not isinstance(type_raw, str) or type_raw.strip().lower() != "code":
            raise ValueError('Tool payload must include "type": "code"')
        allowed_keys = {"type", "code_command"}
        extras = set(data.keys()) - allowed_keys
        if extras:
            raise ValueError(f"Unexpected fields for code payload: {sorted(extras)}")
        command = str(data.get("code_command", "")).strip()
        try:
            return cls(type="code", code_command=command)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc


class AzureCommand(BaseModel):
    type: Literal["azure"] = Field(..., description="Tool discriminator for Azure CLI")
    azure_command: str = Field(..., min_length=1, description="Azure command string")

    @classmethod
    def coerce(cls, payload: Union[str, Dict[str, Any]]) -> "AzureCommand":
        data = _as_dict(payload)
        type_raw = data.get("type")
        if not isinstance(type_raw, str) or type_raw.strip().lower() != "azure":
            raise ValueError('Tool payload must include "type": "azure"')
        allowed_keys = {"type", "azure_command"}
        extras = set(data.keys()) - allowed_keys
        if extras:
            raise ValueError(f"Unexpected fields for azure payload: {sorted(extras)}")
        command = str(data.get("azure_command", "")).strip()
        try:
            return cls(type="azure", azure_command=command)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc


def ensure_web_payload(payload: Union[str, Dict[str, Any]], default_k: int = 3) -> Dict[str, Any]:
    model = WebCommand.coerce(payload, default_k=default_k)
    return {"type": model.type, "q": model.q, "k": model.k}


def ensure_code_payload(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    model = CodeCommand.coerce(payload)
    return {"type": model.type, "code_command": model.code_command}


def ensure_azure_payload(payload: Union[str, Dict[str, Any]]) -> Dict[str, Any]:
    model = AzureCommand.coerce(payload)
    return {"type": model.type, "azure_command": model.azure_command}

