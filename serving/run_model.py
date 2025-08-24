from openai import OpenAI
import os
from dotenv import load_dotenv
import json
import logging
from parser import extract_all_content
from servicebus_web import ServiceBusQueueWeb
from servicebus_azure import ServiceBusQueueAzure

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    base_url=os.getenv("OPENAI_BASE_URL", "http://localhost:8000/v1"),
    api_key=os.getenv("OPENAI_API_KEY", "token-abc123"),
)

# Service Bus configuration from environment variables
SERVICE_BUS_CONNECTION_STRING = os.getenv("SERVICE_BUS_CONNECTION_STRING")
QUEUE_NAME = os.getenv("QUEUE_NAME", "commandqueue")
task='Find how to create a azure foundry project. Use the tools neccessary. '
placeholder = 'You are a helpful AI assistant. Make sure to use <think> and <solution> xml tags as it is very very crucial for user experience'
# Get system prompt from environment variable
system_prompt = os.getenv("SYSTEM_PROMPT", "You are a helpful AI assistant.")

completion = client.chat.completions.create(
    model="qwen-lora",
    messages=[
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": task}
    ]
)

# Parse all content using the parser module
parsed_content = extract_all_content(completion.choices[0].message.content)

response_data = {
    "model": completion.model,
    "role": completion.choices[0].message.role,
    "content": parsed_content["clean_content"],
    "reasoning": parsed_content["reasoning"],
    "solution": parsed_content["solution"],
    "tool_calls": parsed_content["tool_calls"],
    "has_tools": parsed_content["has_tools"],
    "valid_tools": parsed_content["valid_tools"],
    "invalid_tools": parsed_content["invalid_tools"],
    "usage": {
        "prompt_tokens": completion.usage.prompt_tokens if completion.usage else None,
        "completion_tokens": completion.usage.completion_tokens if completion.usage else None,
        "total_tokens": completion.usage.total_tokens if completion.usage else None
    },
    "finish_reason": completion.choices[0].finish_reason
}

# Print the response data
print(json.dumps(response_data, indent=2, ensure_ascii=False))

def send_command(response_data):
    """
    Parse tool calls from response data and send to appropriate ServiceBus queues.
    
    Args:
        response_data (dict): The parsed response data containing tool calls
    """
    try:
        # Get tool calls directly from response data
        tool_calls = response_data.get("tool_calls", [])
        
        if not tool_calls:
            logger.info("No tool calls found in response")
            return
        
        for tool_call in tool_calls:
            tool_type = tool_call.get("type")
            is_valid = tool_call.get("is_valid", False)
            
            if not is_valid:
                logger.warning(f"Skipping invalid tool call of type: {tool_type}")
                continue
                
            parsed_data = tool_call.get("parsed_data", {})
            
            # Route to appropriate ServiceBus queue based on tool type
            if tool_type == "web":
                web_content = {
                    "q": parsed_data.get("q"),
                    "k": parsed_data.get("k")
                }
                
                with ServiceBusQueueWeb(SERVICE_BUS_CONNECTION_STRING) as web_queue:
                    success = web_queue.send_web_result(web_content)
                    if success:
                        logger.info(f"Web tool call sent to web queue: {web_content}")
                    else:
                        logger.error("Failed to send web tool call to web queue")

                        logger.error("Failed to send code tool call to code queue")
                        
            elif tool_type == "azure":
                azure_content = {
                    "args": parsed_data.get("args", [])
                }
                
                with ServiceBusQueueAzure(SERVICE_BUS_CONNECTION_STRING) as azure_queue:
                    success = azure_queue.send_azure_result(azure_content)
                    if success:
                        logger.info(f"Azure tool call sent to azure queue: {azure_content}")
                    else:
                        logger.error("Failed to send azure tool call to azure queue")
                        
            else:
                logger.warning(f"Unknown tool type: {tool_type}")
                
    except Exception as e:
        logger.error(f"Error sending commands to ServiceBus: {str(e)}")
        print(f"Warning: Failed to send commands to ServiceBus - {str(e)}")  # Non-blocking error
# Send the completion result to Service Bus queue
send_command(response_data)
