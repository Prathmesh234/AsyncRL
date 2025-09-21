from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import logging
from parser import stream_parser
from ToolGRPOTrainer.command_sender import send_web_command
from ToolGRPOTrainer.azure_command_sender import send_azure_command
from ToolGRPOTrainer.code_command_sender import send_code_command
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

MAX_TOOL_TURNS = _int_from_env("MAX_TOOL_TURNS", 6)
TURN_MAX_NEW_TOKENS = _int_from_env("TURN_MAX_NEW_TOKENS", 512)

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

# NOTE: To use the GRPO-trained model, you need to:
# 1. Stop your current vLLM server
# 2. Start vLLM server with the GRPO-trained model:
#    python -m vllm.entrypoints.openai.api_server \
#    --model /home/ubuntu/GeneratorFS/grpo-qwen-training/checkpoint-100 \
#    --served-model-name qwen-lora \
#    --port 8000

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
task = "use the azure tool to run az account show to get subscription id and tenant id, then run az provider show --namespace Microsoft.Compute --query resourceTypes[?resourceType=='virtualMachines'].locations -o json to count how many regions support virtualMachines, then run az vm list-usage --location eastus -o json and report the two usage metrics closest to their limits. Summarize all findings concisely."
# Get system prompt from environment variable
system_prompt = os.getenv("SYSTEM_PROMPT", """You are an ORCHESTRATOR/CODING AGENT. Complete real tasks by calling tools and then returning a concise final solution.

BEGIN RULE (MANDATORY)
- Always—and always—start EVERY answer with <think>...</think>. Example: <think> okay so this question is about ... </think>. Do this on the very first turn and after each tool result.

ALLOWED TAGS ONLY
- <think>...</think> — planning for the next step.
- <web>...</web> — web search (official docs, flags, troubleshooting).
- <code>...</code> — code/test/build inside /workspace.
- <azure>...</azure> — whitelisted Azure CLI subcommands.
- <solution>...</solution> — final answer for the user.
Do not use any other tags, formats, or tools.

STRICT TOOL CALL FORMAT (MANDATORY)
- Wrap a JSON object directly inside the tool tag and emit nothing else on that turn.
- The object must match EXACTLY one of the following (no extra or missing keys, no comments):
  • <web>{"type": "web", "q": "search terms", "k": INTEGER}</web> — q must be non-empty and k must be an integer between 1 and 10.
  • <code>{"type": "code", "code_command": "shell command"}</code> — code_command must be a non-empty string.
  • <azure>{"type": "azure", "azure_command": "az subcommand"}</azure> — azure_command must be a non-empty string.
- Always produce tool calls in exactly this JSON format. Do not add request_id, metadata, comments, or additional fields.
- JSON must use double quotes around every key/value and only whitespace outside the object.

TURN PROTOCOL (STRICT)
1) In <think>, decide exactly ONE next action and why.
2) After </think>, emit EITHER:
   • exactly one tool call (<web>/<code>/<azure>) with content and NO prose, OR
   • a single <solution>…</solution> if the task is finished or you must ask a single clarifying question.
3) Wait for the tool RESULT before taking another action. Do not speculate about tool outputs.

DECISION POLICY (INSTRUCTION FOLLOWING)
- Prefer tools over guessing. When commands or procedures are unknown, FIRST use <web> to find the official source.
- Implement → verify loop:
  • After code edits/builds: verify via <code> (tests, artifact check).
  • After Azure writes: confirm state via a read-only *show*/list in <azure>.

DECOUPLING & SAFETY
- One tool call per turn. Never mix tools.
- Do NOT run Azure inside <code>. Do NOT fetch/curl docs inside <code> if <web> exists.
- Use least-privilege, safe, idempotent commands; avoid destructive operations unless explicitly requested.

COMPLETION RULES
- Stop when success criteria are met. Output a single <solution>…</solution> that states what was done, provides brief evidence, and omits raw logs.""")

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
            model="qwen-lora",
            messages=messages + [{"role": "assistant", "content": conversation}] if conversation else messages,
            stream=True,
            max_tokens=turn_max_new_tokens,
            temperature=0.7
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
