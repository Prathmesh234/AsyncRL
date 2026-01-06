"""
Parser module for handling different types of content extraction from model outputs.
Contains parsers for thinking tags, solution tags, and tool calls.
"""

import json
import re
from typing import List, Dict, Any, Tuple, Optional, Iterable

TOOL_TAGS = ("web", "code", "azure")
TOOL_SCHEMAS = {
    "web": ["q", "k"],
    "code": ["code_command"],
    "azure": ["azure_command"],
}


def _extract_tag(content: str, tag: str) -> List[Tuple[int, str]]:
    """Return list of (start_index, inner_text) for each completed tag occurrence."""
    open_tag = f"<{tag}>"
    close_tag = f"</{tag}>"
    matches: List[Tuple[int, str]] = []
    search_pos = 0
    while True:
        start = content.find(open_tag, search_pos)
        if start == -1:
            break
        end = content.find(close_tag, start + len(open_tag))
        if end == -1:
            break
        inner = content[start + len(open_tag):end]
        matches.append((start, inner))
        search_pos = end + len(close_tag)
    return matches


def stream_parser(buffer: str, allowed_tools: Optional[Iterable[str]] = None):
    """
    Detect complete tool tags in a streaming buffer.
    Returns {"type": tool_type, "content": payload} if found, else None.
    """
    earliest = None
    tools = tuple(allowed_tools) if allowed_tools is not None else TOOL_TAGS
    for tool in tools:
        matches = _extract_tag(buffer, tool)
        if matches:
            start, inner = matches[0]
            if earliest is None or start < earliest[0]:
                earliest = (start, tool, inner.strip())
    if earliest:
        return {"type": earliest[1], "content": earliest[2]}
    return None


def parse_thinking_tags(content: str) -> Tuple[str, str, str]:
    """Extract reasoning (<think>) and solution (<solution>) tags, returning (reasoning, solution, clean_content)."""
    reasoning = ""
    solution = ""
    clean_content = content
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, content, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        clean_content = re.sub(think_pattern, '', clean_content, flags=re.DOTALL).strip()
    solution_pattern = r'<solution>(.*?)</solution>'
    solution_match = re.search(solution_pattern, content, re.DOTALL)
    if solution_match:
        solution = solution_match.group(1).strip()
        clean_content = re.sub(solution_pattern, '', clean_content, flags=re.DOTALL).strip()
    return reasoning, solution, clean_content


def parse_tool_tags(content: str) -> List[Dict[str, Any]]:
    """Return list of tool calls with type and content."""
    tool_calls = []
    for tool_type in TOOL_TAGS:
        for _, inner in _extract_tag(content, tool_type):
            tool_calls.append({"type": tool_type, "content": inner.strip()})
    return tool_calls


def parse_json_from_tool_content(tool_content: str) -> Dict[str, Any]:
    """Parse a tool payload and require a strict JSON object."""
    text = tool_content.strip()
    if not text:
        raise ValueError("Tool payload cannot be empty")
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("Tool payload must decode to an object")
    return data


def validate_tool_schema(tool_type: str, tool_data: Dict[str, Any]) -> bool:
    """Basic schema validation per tool type."""
    if "raw_content" in tool_data and len(tool_data) == 1:
        return False
    required = TOOL_SCHEMAS.get(tool_type)
    if required is None:
        return False
    if tool_data.get("type") != tool_type:
        return False
    allowed_keys = {"type", *required}
    if set(tool_data.keys()) != allowed_keys:
        return False
    return all(key in tool_data for key in required)


def parse_and_validate_tools(content: str) -> List[Dict[str, Any]]:
    """Parse tool tags and attach parsed JSON + validity flag."""
    tool_calls = parse_tool_tags(content)
    validated = []
    for call in tool_calls:
        try:
            parsed = parse_json_from_tool_content(call["content"])
        except ValueError as exc:
            validated.append({
                "type": call["type"],
                "content": call["content"],
                "parsed_data": None,
                "is_valid": False,
                "error": str(exc),
            })
            continue
        is_valid = validate_tool_schema(call["type"], parsed)
        entry: Dict[str, Any] = {
            "type": call["type"],
            "content": call["content"],
            "parsed_data": parsed,
            "is_valid": is_valid,
        }
        if not is_valid:
            entry["error"] = "Invalid tool schema"
        validated.append(entry)
    return validated


def extract_all_content(content: str) -> Dict[str, Any]:
    """Aggregate extraction of reasoning, solution, and tool metadata."""
    reasoning, solution, clean_content = parse_thinking_tags(content)
    validated_tools = parse_and_validate_tools(content)
    return {
        "reasoning": reasoning or None,
        "solution": solution or None,
        "clean_content": clean_content,
        "tool_calls": validated_tools or None,
        "has_tools": len(validated_tools) > 0,
        "valid_tools": [t for t in validated_tools if t["is_valid"]],
        "invalid_tools": [t for t in validated_tools if not t["is_valid"]],
    }
