#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Modified for DisGenerator by Prathmesh Bhatt
"""
Disaggregated Serving Proxy Server for P2P NCCL XpYd Architecture.

This proxy server handles:
1. Service Discovery - Prefill/Decode instances register via ZMQ
2. Request Routing - Routes prefill → decode based on X-Request-Id
3. Two-Phase Execution - First prefill (max_tokens=1), then decode (full request)

Based on vLLM's official disaggregated serving example:
https://docs.vllm.ai/en/latest/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/
"""

import os
import socket
import sys
import threading
import time
import uuid
from typing import Any

import aiohttp
import msgpack
import zmq
from quart import Quart, make_response, request

# Configuration
PROXY_IP = os.getenv("PROXY_IP", "0.0.0.0")
PROXY_ZMQ_PORT = int(os.getenv("PROXY_ZMQ_PORT", "30001"))
PROXY_HTTP_PORT = int(os.getenv("PROXY_HTTP_PORT", "10001"))
REQUEST_TIMEOUT_HOURS = int(os.getenv("REQUEST_TIMEOUT_HOURS", "6"))
DEFAULT_PING_SECONDS = 5

# Global state
count = 0
prefill_instances: dict[str, Any] = {}  # http_address: (zmq_address, stamp)
decode_instances: dict[str, Any] = {}   # http_address: (zmq_address, stamp)
prefill_cv = threading.Condition()
decode_cv = threading.Condition()

# Quart app
app = Quart(__name__)

# aiohttp timeout
AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_HOURS * 60 * 60)


def random_uuid() -> str:
    """Generate a random UUID hex string."""
    return str(uuid.uuid4().hex)


def _remove_oldest_instances(instances: dict[str, Any]) -> None:
    """Remove expired instances from the registry."""
    oldest_key = next(iter(instances), None)
    while oldest_key is not None:
        value = instances[oldest_key]
        if value[1] > time.time():
            break
        print(f"🔴 Remove [HTTP:{oldest_key}, ZMQ:{value[0]}, stamp:{value[1]}]")
        instances.pop(oldest_key, None)
        oldest_key = next(iter(instances), None)


def _listen_for_register(poller, router_socket):
    """
    Background thread that listens for instance registrations via ZMQ.
    
    Instances send heartbeat messages with format:
    {
        "type": "P" (prefill) or "D" (decode),
        "http_address": "ip:port",
        "zmq_address": "ip:port"
    }
    """
    while True:
        socks = dict(poller.poll())
        if router_socket in socks:
            remote_address, message = router_socket.recv_multipart()
            data = msgpack.loads(message)
            
            if data["type"] == "P":
                # Prefill instance registration
                with prefill_cv:
                    node = prefill_instances.get(data["http_address"], None)
                    prefill_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(prefill_instances)
                    
            elif data["type"] == "D":
                # Decode instance registration
                with decode_cv:
                    node = decode_instances.get(data["http_address"], None)
                    decode_instances[data["http_address"]] = (
                        data["zmq_address"],
                        time.time() + DEFAULT_PING_SECONDS,
                    )
                    _remove_oldest_instances(decode_instances)
            else:
                print(f"Unexpected message from {remote_address}: {data}")
                continue
                
            if node is None:
                print(f"🔵 Add [{data['type']}] HTTP:{data['http_address']}, ZMQ:{data['zmq_address']}")


def start_service_discovery(hostname: str, port: int) -> threading.Thread:
    """
    Start the ZMQ service discovery listener.
    
    Args:
        hostname: IP to bind to (0.0.0.0 for all interfaces)
        port: ZMQ port for instance registration
        
    Returns:
        The listener thread
    """
    if not hostname:
        hostname = socket.gethostname()
    if port == 0:
        raise ValueError("Port cannot be 0")
        
    context = zmq.Context()
    router_socket = context.socket(zmq.ROUTER)
    router_socket.bind(f"tcp://{hostname}:{port}")
    
    poller = zmq.Poller()
    poller.register(router_socket, zmq.POLLIN)
    
    listener_thread = threading.Thread(
        target=_listen_for_register,
        args=[poller, router_socket],
        daemon=True,
    )
    listener_thread.start()
    
    print(f"🚀 Service discovery started on tcp://{hostname}:{port}")
    return listener_thread


async def forward_request(url: str, data: dict, request_id: str):
    """
    Forward a request to a vLLM server and stream the response.
    
    The X-Request-Id header encodes routing information that vLLM uses
    to coordinate KV cache transfer between prefill and decode instances.
    """
    async with aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT) as session:
        headers = {
            "Authorization": f"Bearer {os.environ.get('OPENAI_API_KEY', '')}",
            "X-Request-Id": request_id,
            "Content-Type": "application/json",
        }
        
        async with session.post(url=url, json=data, headers=headers) as response:
            if response.status == 200:
                async for chunk_bytes in response.content.iter_chunked(1024):
                    yield chunk_bytes
            else:
                error_text = await response.text()
                print(f"Error from {url}: {response.status} - {error_text}")
                yield f"Error: {response.status}".encode()


@app.route("/v1/completions", methods=["POST"])
@app.route("/v1/chat/completions", methods=["POST"])
async def handle_request():
    """
    Main request handler implementing two-phase disaggregated serving.
    
    Phase 1 (Prefill):
        - Send request to prefill instance with max_tokens=1
        - Prefill computes KV cache and sends to decode via NCCL
        - Response is discarded
        
    Phase 2 (Decode):
        - Send original request to decode instance
        - Decode uses pre-transferred KV cache
        - Stream response back to client
    """
    global count
    
    try:
        original_request_data = await request.get_json()
        
        # Phase 1: Create prefill request with max_tokens=1
        prefill_request = original_request_data.copy()
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        
        # Select prefill instance (round-robin)
        with prefill_cv:
            if not prefill_instances:
                return {"error": "No prefill instances available"}, 503
            prefill_list = list(prefill_instances.items())
            ##this is for round robin 
            prefill_addr, (prefill_zmq_addr, _) = prefill_list[count % len(prefill_list)]
        
        # Select decode instance (round-robin)
        ##consider a case where we have havd  count of 2, then we first try this with  0 % 2 gpu index 0 
        ## then if it is busy we try with ! 0 % 2 gpu index 1
        ## if that is busy again we try with 2 % 2 gpu index 0 ....
        ##this goes on however we keep trying back and forth until we find an available gpu
        with decode_cv:
            if not decode_instances:
                return {"error": "No decode instances available"}, 503
            decode_list = list(decode_instances.items())
            decode_addr, (decode_zmq_addr, _) = decode_list[count % len(decode_list)]
        
        print(
            f"📝 Request #{count}: "
            f"[P] {prefill_addr}:{prefill_zmq_addr} 👉 "
            f"[D] {decode_addr}:{decode_zmq_addr}"
        )
        count += 1
        
        # Create request ID encoding routing information
        # Format: ___prefill_addr_{P_ZMQ}___decode_addr_{D_ZMQ}_{UUID}
        request_id = (
            f"___prefill_addr_{prefill_zmq_addr}___decode_addr_"
            f"{decode_zmq_addr}_{random_uuid()}"
        )
        
        # Phase 1: Execute prefill (discard response)
        prefill_url = f"http://{prefill_addr}{request.path}"
        async for _ in forward_request(prefill_url, prefill_request, request_id):
            pass  # Discard prefill response - only used for KV cache transfer
        
        # Phase 2: Execute decode and stream response
        decode_url = f"http://{decode_addr}{request.path}"
        generator = forward_request(decode_url, original_request_data, request_id)
        
        response = await make_response(generator)
        response.timeout = None
        response.headers["Content-Type"] = "text/event-stream"
        return response
        
    except Exception as e:
        import traceback
        print(f"❌ Error in proxy: {e}")
        print(traceback.format_exc())
        return {"error": str(e)}, 500


@app.route("/health", methods=["GET"])
async def health():
    """Health check endpoint."""
    with prefill_cv:
        num_prefill = len(prefill_instances)
    with decode_cv:
        num_decode = len(decode_instances)
        
    return {
        "status": "healthy" if num_prefill > 0 and num_decode > 0 else "degraded",
        "prefill_instances": num_prefill,
        "decode_instances": num_decode,
    }


@app.route("/", methods=["GET"])
async def root():
    """Root endpoint with service info."""
    return {
        "service": "DisGenerator Disaggregated Proxy",
        "version": "0.1.0",
        "endpoints": [
            "/v1/completions",
            "/v1/chat/completions",
            "/health",
        ],
    }


def main():
    """Main entry point."""
    print("=" * 60)
    print("DisGenerator Disaggregated Proxy Server")
    print("=" * 60)
    print(f"  ZMQ Port (service discovery): {PROXY_ZMQ_PORT}")
    print(f"  HTTP Port (API): {PROXY_HTTP_PORT}")
    print(f"  Request timeout: {REQUEST_TIMEOUT_HOURS} hours")
    print("=" * 60)
    
    # Start service discovery
    discovery_thread = start_service_discovery(PROXY_IP, PROXY_ZMQ_PORT)
    
    # Start HTTP server
    print(f"🌐 HTTP API starting on http://{PROXY_IP}:{PROXY_HTTP_PORT}")
    app.run(host=PROXY_IP, port=PROXY_HTTP_PORT)
    
    # Wait for discovery thread
    discovery_thread.join()


if __name__ == "__main__":
    main()
