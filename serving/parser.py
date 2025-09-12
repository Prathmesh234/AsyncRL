"""
Parser module for handling different types of content extraction from model outputs.
Contains parsers for thinking tags, solution tags, and tool calls.
"""

import re
from typing import List, Dict, Any, Tuple


def stream_parser(buffer: str):
    """
    Detect complete tool tags in a streaming buffer.
    Returns {"type": tool_type, "content": payload} if found, else None.
    """
    patterns = {
        "web": r"<web>(.*?)</web>",
        "code": r"<code>(.*?)</code>",
        "azure": r"<azure>(.*?)</azure>",
    }
    for ttype, pat in patterns.items():
        match = re.search(pat, buffer, re.DOTALL)
        if match:
            content = match.group(1).strip()
            return {"type": ttype, "content": content}
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
    for tool_type, pattern in {
        "web": r'<web>(.*?)</web>',
        "code": r'<code>(.*?)</code>',
        "azure": r'<azure>(.*?)</azure>',
    }.items():
        matches = re.findall(pattern, content, re.DOTALL)
        for match in matches:
            tool_calls.append({"type": tool_type, "content": match.strip()})
    return tool_calls


def parse_json_from_tool_content(tool_content: str) -> Dict[str, Any]:
    """Attempt JSON parse; fall back to raw_content."""
    import json
    try:
        return json.loads(tool_content.strip())
    except json.JSONDecodeError:
        json_pattern = r'\{.*\}'
        json_matches = re.findall(json_pattern, tool_content, re.DOTALL)
        for match in json_matches:
            try:
                return json.loads(match.strip())
            except json.JSONDecodeError:
                continue
        return {"raw_content": tool_content.strip()}


def validate_tool_schema(tool_type: str, tool_data: Dict[str, Any]) -> bool:
    """Basic schema validation per tool type."""
    if "raw_content" in tool_data and len(tool_data) == 1:
        return False
    schemas = {
        "web": ["q", "k"],
        "code": ["cmd", "cwd", "timeout_s"],
        "azure": ["args"],
    }
    required = schemas.get(tool_type)
    if required is None:
        return False
    return all(key in tool_data for key in required)


def parse_and_validate_tools(content: str) -> List[Dict[str, Any]]:
    """Parse tool tags and attach parsed JSON + validity flag."""
    tool_calls = parse_tool_tags(content)
    validated = []
    for call in tool_calls:
        parsed = parse_json_from_tool_content(call["content"])
        is_valid = validate_tool_schema(call["type"], parsed)
        validated.append({
            "type": call["type"],
            "content": call["content"],
            "parsed_data": parsed,
            "is_valid": is_valid,
        })
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
