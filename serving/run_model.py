from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import logging
from parser import stream_parser
from communication.command_sender import send_web_command
from communication.azure_command_sender import send_azure_command
from communication.code_command_sender import send_code_command
from validation import ensure_web_payload, ensure_code_payload, ensure_azure_payload

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _int_from_env(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None:
        return default
    try:
        parsed = int(value)
        if parsed <= 0:
            raise ValueError
        return parsed
    except ValueError:
        logger.warning("Invalid %s=%r; using default %d", name, value, default)
        return default

# Load environment variables from .env file
load_dotenv()

MAX_TOOL_TURNS = _int_from_env("MAX_TOOL_TURNS", 8)
TURN_MAX_NEW_TOKENS = _int_from_env("TURN_MAX_NEW_TOKENS", 16384)

# Tool execution functions
def run_web_tool(payload: str) -> str:
    print(f"[TOOL][web] payload={payload!r}")
    try:
        web_payload = ensure_web_payload(payload)
    except Exception as exc:
        return f"[web-error] {exc}"
    return send_web_command(web_payload, timeout_s=15)

def run_code_tool(payload: str) -> str:
    print(f"[TOOL][code] payload={payload!r}")
    try:
        code_payload = ensure_code_payload(payload)
    except Exception as exc:
        return f"[code-error] {exc}"
    return send_code_command(code_payload, timeout_s=15)

def run_azure_tool(payload: str) -> str:
    print(f"[TOOL][azure] payload={payload!r}")
    try:
        azure_payload = ensure_azure_payload(payload)
    except Exception as exc:
        return f"[azure-error] {exc}"
    return send_azure_command(azure_payload, timeout_s=15)

# Load environment variables from .env file
load_dotenv()

# NOTE: To use the GRPO + thinking adapters:
# 1. Run: uv run main.py (starts vLLM with both GRPO and thinking LoRA adapters)
# 2. This loads Qwen3-4B-Thinking-2507 base model with:
#    - grpo-adapter: GRPO-trained weights for tool usage/formatting
#    - thinking-lora: Enhanced reasoning capabilities

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "token-abc123"),
)

# Service Bus configuration from environment variables
SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "token-abc123"),
)

SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")
task = "Research how to create Azure Blob Storage accounts and containers using Azure CLI. Then use Azure CLI to create a new storage account and blob container in East US. Create a simple Python script that uploads a text file to the newly created blob container and lists all blobs in the container. Test the script to verify the upload and listing works correctly."
# Get system prompt from environment variable
system_prompt = os.getenv("SYSTEM_PROMPT")

def stream_generate_with_tools(messages, max_turns=MAX_TOOL_TURNS, turn_max_new_tokens=TURN_MAX_NEW_TOKENS):
    """Generate tokens with streaming and real-time tool execution."""
    print("[GEN] start")
    conversation = ""
    full_trace = ""
    turns = 0
    
    while turns < max_turns:
        buffer = ""
        
        # Create streaming completion
        stream = client.chat.completions.create(
            model="grpo-adapter",  # Use the GRPO adapter which includes tool/format training
            messages=messages + [{"role": "assistant", "content": conversation}] if conversation else messages,
            stream=True,
            max_tokens=turn_max_new_tokens,
            temperature=0.5
        )
        
        tool_triggered = False
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                new_text = chunk.choices[0].delta.content
                buffer += new_text
                full_trace += new_text
                
                # Check for tool calls using stream_parser
                tool_call = stream_parser(buffer)
                if tool_call:
                    tool_type = tool_call.get("type")
                    content = tool_call.get("content")
                    
                    if tool_type == "web":
                        result = run_web_tool(content)
                    elif tool_type == "code":
                        result = run_code_tool(content)
                    elif tool_type == "azure":
                        result = run_azure_tool(content)
                    else:
                        result = "[error] unknown tool"
                    
                    tool_result = f"<tool_result>{result}</tool_result>\n"
                    conversation += buffer + tool_result
                    full_trace += tool_result
                    buffer = ""
                    tool_triggered = True
                    break
        
        if not tool_triggered:
            conversation += buffer
            
        if "<solution>" in full_trace:
            break
            
        turns += 1
    
    print(full_trace, flush=True)
    return full_trace

# Execute streaming generation with tool support
result = stream_generate_with_tools([
    {"role": "system", "content": system_prompt},
    {"role": "user", "content": task}
])

print(f"\n[FINAL RESULT]\n{result}")
