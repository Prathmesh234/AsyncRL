#!/usr/bin/env python3
"""
Simple client for testing DisGenerator disaggregated serving.

This client sends a single prompt through the proxy server and
prints the streamed response.

Usage:
    uv run python simple_client.py
    uv run python simple_client.py "What is the capital of France?"
"""

import asyncio
import json
import os
import sys

import aiohttp
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configuration (matches DisTrainer)
PROXY_URL = os.getenv("PROXY_URL", "http://localhost:10001")
MODEL = os.getenv("MODEL", "Qwen/Qwen3-4B-Thinking-2507")
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "20000"))
SYSTEM_PROMPT = os.getenv("SYSTEM_PROMPT", "")


async def generate_simple(prompt: str, stream: bool = True) -> str:
    """
    Send a single prompt through disaggregated serving.
    
    Args:
        prompt: The user prompt
        stream: Whether to stream the response
        
    Returns:
        The complete response text
    """
    messages = []
    if SYSTEM_PROMPT:
        messages.append({"role": "system", "content": SYSTEM_PROMPT})
    messages.append({"role": "user", "content": prompt})

    payload = {
        "model": MODEL,
        "messages": messages,
        "max_tokens": MAX_TOKENS,
        "stream": stream,
        "temperature": 0.7,
    }
    
    url = f"{PROXY_URL}/v1/chat/completions"
    
    print(f"\n{'='*60}")
    print(f"📤 Sending request to: {url}")
    print(f"📝 Prompt: {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"{'='*60}\n")
    
    async with aiohttp.ClientSession() as session:
        async with session.post(url, json=payload) as resp:
            if resp.status != 200:
                error_text = await resp.text()
                print(f"❌ Error: {resp.status} - {error_text}")
                return ""
            
            if stream:
                # Handle SSE stream
                full_response = ""
                print("📥 Response: ", end="", flush=True)
                
                async for line in resp.content:
                    line_str = line.decode("utf-8").strip()
                    
                    if not line_str:
                        continue
                        
                    if line_str.startswith("data: "):
                        data_str = line_str[6:]  # Remove "data: " prefix
                        
                        if data_str == "[DONE]":
                            break
                            
                        try:
                            data = json.loads(data_str)
                            if "choices" in data and len(data["choices"]) > 0:
                                delta = data["choices"][0].get("delta", {})
                                content = delta.get("content", "")
                                if content:
                                    print(content, end="", flush=True)
                                    full_response += content
                        except json.JSONDecodeError:
                            pass  # Skip malformed lines
                
                print("\n")
                return full_response
            else:
                # Non-streaming response
                data = await resp.json()
                content = data["choices"][0]["message"]["content"]
                print(f"📥 Response: {content}")
                return content


async def test_health():
    """Check the health of the proxy server."""
    url = f"{PROXY_URL}/health"
    
    async with aiohttp.ClientSession() as session:
        try:
            async with session.get(url) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Proxy health: {data}")
                    return data
                else:
                    print(f"❌ Proxy unhealthy: {resp.status}")
                    return None
        except aiohttp.ClientError as e:
            print(f"❌ Cannot connect to proxy: {e}")
            return None


async def main():
    """Main entry point."""
    print("\n" + "="*60)
    print("DisGenerator Simple Client")
    print("="*60)
    print(f"  Proxy URL: {PROXY_URL}")
    print(f"  Model: {MODEL}")
    print(f"  Max tokens: {MAX_TOKENS}")
    print("="*60 + "\n")
    
    # Check health first
    health = await test_health()
    if not health:
        print("\n⚠️  Proxy server not available. Make sure to run:")
        print("    ./scripts/start_all.sh")
        return
    
    # Get prompt from command line or use default
    if len(sys.argv) > 1:
        prompt = " ".join(sys.argv[1:])
    else:
        prompt = "What is the capital of France? Answer in one sentence."
    
    # Generate response
    response = await generate_simple(prompt)
    
    print("="*60)
    print("✅ Test complete!")
    print(f"📏 Response length: {len(response)} characters")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
