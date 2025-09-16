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


def _extract_from_mapping(mapping: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    if not isinstance(mapping, dict):
        return None
    q = mapping.get("q")
    k = mapping.get("k")
    if q is None and k is None:
        return None
    return {"q": q, "k": k}


def extract_qk_from_data(received_command: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """
    Given a parsed received_command dict, extract the `q` and `k` fields
    regardless of whether they are top-level or nested inside "data".
    Returns None if unavailable.
    """
    if not isinstance(received_command, dict):
        return None

    direct = _extract_from_mapping(received_command)
    if direct:
        return direct

    for key in ("data", "action", "parameters"):
        nested = _extract_from_mapping(received_command.get(key))
        if nested:
            return nested

    return None


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

    extracted = extract_qk_from_data(parsed)
    if extracted:
        return extracted

    return extract_qk_from_data(payload)

