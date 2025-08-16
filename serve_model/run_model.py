from openai import OpenAI
import os
from dotenv import load_dotenv
import json
from parser import extract_all_content

# Load environment variables from .env file
load_dotenv()

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)
task='Find the official Microsoft doc that shows how to print the current Azure subscription name using the CLI, remember the exact command from that doc, then in /workspace create a file hello.txt containing hello world and read it back, then run the Azure command to print the subscription name. Finally, report (1) the file’s contents, (2) the subscription name you retrieved, and (3) the URL of the doc you used.'
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

print(json.dumps(response_data, indent=2, ensure_ascii=False))