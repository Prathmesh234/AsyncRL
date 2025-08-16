# Container Feature Implementation

## Overview
This document outlines the implementation of container-based tool execution for our orchestrator agent model. The system is designed to parse XML tags from model outputs and initiate appropriate containers for different tool types.

## Tool Categories
The orchestrator agent supports three main tool categories:

1. **Web Tool** (`<web></web>`)
   - Purpose: API-based web searches and document retrieval
   - Container: Web browser container
   - Usage: `<web>{query, top_k}</web>`

2. **Code Execution Tool** (`<code></code>`)
   - Purpose: Running commands and executing code in workspace
   - Container: Code execution container
   - Usage: `<code>{code to be executed}</code>`

3. **Azure Tool** (`<azure></azure>`)
   - Purpose: Azure infrastructure commands and operations
   - Container: Azure container
   - Usage: `<azure>{terminal_commands}</azure>`

## Implementation Details

### Parser Functionality
Created `parse_tool_tags()` function in `run_model.py` that:
- Uses regex patterns to detect XML tags in model output
- Extracts content within each tag type
- Logs container initiation messages
- Returns structured tool call data

### Integration with Existing System
- Extended the existing `parse_thinking_tags()` functionality
- Added tool_calls to the response JSON structure
- Maintains backward compatibility with existing parsing

### Container Initiation Messages
When tool tags are detected, the system prints:
- `"Initiating the web browser container ..."` for `<web>` tags
- `"Initiating the code execution container ..."` for `<code>` tags  
- `"Initiating the azure container ..."` for `<azure>` tags

## Code Changes Made

### Modified Files:
1. **run_model.py**
   - Added `parse_tool_tags()` function
   - Integrated tool parsing into main response flow
   - Added `tool_calls` field to response JSON

### New Regex Patterns:
```python
web_pattern = r'<web>(.*?)</web>'
code_pattern = r'<code>(.*?)</code>'
azure_pattern = r'<azure>(.*?)</azure>'
```

## Future Enhancements
- Implement actual container spawning logic
- Add error handling for malformed XML tags
- Create container management system
- Add logging and monitoring for container operations
- Implement container lifecycle management

## Testing Considerations
- Test with various XML tag combinations
- Verify proper parsing of nested content
- Ensure graceful handling of incomplete tags
- Test with concurrent tool calls
