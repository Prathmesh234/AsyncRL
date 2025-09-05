import json
from typing import Any, Dict, Optional, Union


def parse_received_command(obj: Union[str, Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Accepts either a JSON string or a dict for the received command body
    and returns it as a dict, or None if it cannot be parsed.
    """
    if obj is None:
        return None

    if isinstance(obj, dict):
        return obj

    if isinstance(obj, str):
        obj = obj.strip()
        if not obj:
            return None
        try:
            return json.loads(obj)
        except Exception:
            return None

    # Unsupported type
    return None


def extract_qk_from_data(received_command: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Given a parsed received_command dict, extract the `q` and `k` fields
    from its `data` object, if present. Returns None if unavailable.
    """
    if not isinstance(received_command, dict):
        return None

    data = received_command.get("data")
    if not isinstance(data, dict):
        return None

    q = data.get("q")
    k = data.get("k")
    if q is None and k is None:
        return None

    return {"q": q, "k": k}


def parse_read_command_payload(payload: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convenience parser for the entire `/read-command` response payload
    to obtain just the `q` and `k` values, if present.
    """
    if not isinstance(payload, dict):
        return None

    # Prefer already-parsed structure when available
    received = payload.get("received_command")
    parsed = parse_received_command(received)
    if parsed is None:
        # Fall back to raw_content (stringified JSON)
        raw = payload.get("raw_content")
        parsed = parse_received_command(raw)

    return extract_qk_from_data(parsed)

