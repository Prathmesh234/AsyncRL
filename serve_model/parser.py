"""
Parser module for handling different types of content extraction from model outputs.
Contains parsers for thinking tags, solution tags, and tool calls.
"""

import re
from typing import List, Dict, Any, Tuple


def parse_thinking_tags(content: str) -> Tuple[str, str, str]:
    """
    Parse content to extract reasoning and solution sections.
    
    Args:
        content (str): The raw content from model output
        
    Returns:
        Tuple[str, str, str]: (reasoning, solution, clean_content)
            - reasoning: Content within <think>...</think> tags
            - solution: Content within <solution>...</solution> tags  
            - clean_content: Original content with tags removed
    """
    reasoning = ""
    solution = ""
    clean_content = content
    
    # Extract thinking/reasoning content
    think_pattern = r'<think>(.*?)</think>'
    think_match = re.search(think_pattern, content, re.DOTALL)
    if think_match:
        reasoning = think_match.group(1).strip()
        clean_content = re.sub(think_pattern, '', clean_content, flags=re.DOTALL).strip()
    
    # Extract solution content
    solution_pattern = r'<solution>(.*?)</solution>'
    solution_match = re.search(solution_pattern, content, re.DOTALL)
    if solution_match:
        solution = solution_match.group(1).strip()
        clean_content = re.sub(solution_pattern, '', clean_content, flags=re.DOTALL).strip()
    
    return reasoning, solution, clean_content


def parse_tool_tags(content: str) -> List[Dict[str, Any]]:
    """
    Parse content to detect and handle tool XML tags.
    
    Args:
        content (str): The raw content from model output
        
    Returns:
        List[Dict[str, Any]]: List of tool calls with type and content
    """
    tool_calls = []
    
    # Check for web tool usage
    web_pattern = r'<web>(.*?)</web>'
    web_matches = re.findall(web_pattern, content, re.DOTALL)
    for match in web_matches:
        tool_calls.append({"type": "web", "content": match.strip()})
        print("Initiating the web browser container ...")
    
    # Check for code execution tool usage  
    code_pattern = r'<code>(.*?)</code>'
    code_matches = re.findall(code_pattern, content, re.DOTALL)
    for match in code_matches:
        tool_calls.append({"type": "code", "content": match.strip()})
        print("Initiating the code execution container ...")
    
    # Check for azure tool usage
    azure_pattern = r'<azure>(.*?)</azure>'
    azure_matches = re.findall(azure_pattern, content, re.DOTALL)
    for match in azure_matches:
        tool_calls.append({"type": "azure", "content": match.strip()})
        print("Initiating the azure container ...")
    
    return tool_calls


def parse_json_from_tool_content(tool_content: str) -> Dict[str, Any]:
    """
    Parse JSON content from tool tags.
    
    Args:
        tool_content (str): The content within tool tags
        
    Returns:
        Dict[str, Any]: Parsed JSON object or empty dict if parsing fails
    """
    import json
    
    try:
        # Try to parse the content as JSON
        parsed_json = json.loads(tool_content.strip())
        return parsed_json
    except json.JSONDecodeError as e:
        print(f"Warning: Failed to parse JSON from tool content: {e}")
        return {"raw_content": tool_content.strip()}


def validate_tool_schema(tool_type: str, tool_data: Dict[str, Any]) -> bool:
    """
    Validate tool data against expected schemas.
    
    Args:
        tool_type (str): Type of tool (web, code, azure)
        tool_data (Dict[str, Any]): Parsed tool data
        
    Returns:
        bool: True if valid, False otherwise
    """
    if tool_type == "web":
        required_keys = ["q", "k"]
        return all(key in tool_data for key in required_keys)
    
    elif tool_type == "code":
        required_keys = ["cmd", "cwd", "timeout_s"]
        return all(key in tool_data for key in required_keys)
    
    elif tool_type == "azure":
        required_keys = ["args"]
        return "args" in tool_data and isinstance(tool_data["args"], list)
    
    return False


def parse_and_validate_tools(content: str) -> List[Dict[str, Any]]:
    """
    Parse tool tags and validate their JSON schemas.
    
    Args:
        content (str): The raw content from model output
        
    Returns:
        List[Dict[str, Any]]: List of validated tool calls with parsed JSON
    """
    tool_calls = parse_tool_tags(content)
    validated_tools = []
    
    for tool_call in tool_calls:
        tool_type = tool_call["type"]
        tool_content = tool_call["content"]
        
        # Parse JSON from tool content
        parsed_data = parse_json_from_tool_content(tool_content)
        
        # Validate schema
        is_valid = validate_tool_schema(tool_type, parsed_data)
        
        validated_tool = {
            "type": tool_type,
            "content": tool_content,
            "parsed_data": parsed_data,
            "is_valid": is_valid
        }
        
        if not is_valid:
            print(f"Warning: Invalid schema for {tool_type} tool: {parsed_data}")
        
        validated_tools.append(validated_tool)
    
    return validated_tools


def extract_all_content(content: str) -> Dict[str, Any]:
    """
    Extract all types of content from model output in one pass.
    
    Args:
        content (str): The raw content from model output
        
    Returns:
        Dict[str, Any]: Dictionary containing all extracted content
    """
    # Parse thinking and solution tags
    reasoning, solution, clean_content = parse_thinking_tags(content)
    
    # Parse and validate tool calls
    validated_tools = parse_and_validate_tools(content)
    
    return {
        "reasoning": reasoning if reasoning else None,
        "solution": solution if solution else None,
        "clean_content": clean_content,
        "tool_calls": validated_tools if validated_tools else None,
        "has_tools": len(validated_tools) > 0,
        "valid_tools": [tool for tool in validated_tools if tool["is_valid"]],
        "invalid_tools": [tool for tool in validated_tools if not tool["is_valid"]]
    }
