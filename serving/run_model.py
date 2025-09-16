from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import logging
from parser import stream_parser
from ToolGRPOTrainer.command_sender import send_web_command
from ToolGRPOTrainer.azure_command_sender import send_azure_command
from ToolGRPOTrainer.code_command_sender import send_code_command

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

# Tool execution functions
def run_web_tool(payload: str) -> str:
    print(f"[TOOL][web] payload={payload!r}")
    return send_web_command(payload, k=3, timeout_s=15)

def run_code_tool(payload: str) -> str:
    print(f"[TOOL][code] payload={payload!r}")
    return send_code_command(payload, timeout_s=15)

def run_azure_tool(payload: str) -> str:
    print(f"[TOOL][azure] payload={payload!r}")
    return send_azure_command(payload, timeout_s=15)

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
QUEUE_NAME = os.getenv("QUEUE_NAME", "commandqueue")
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
QUEUE_NAME = os.getenv("QUEUE_NAME", "commandqueue")
task = "Search the web for Azure Cognitive Services sentiment analysis API documentation, provision a Cognitive Services resource in Azure, and write a Python script using SQLAlchemy models to call the API for sentiment scoring on customer feedback data. Include error handling for API failures, logging, and a command-line interface to run inference on new inputs at runtime."
placeholder = 'You are a helpful AI assistant. Make sure to use <think> and <solution> xml tags as it is very very crucial for user experience'
# Get system prompt from environment variable
system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")

def stream_generate_with_tools(messages, max_turns=6, turn_max_new_tokens=256):
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
