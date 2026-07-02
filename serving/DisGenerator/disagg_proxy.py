#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Adapted from vLLM's NIXL disaggregated-serving proxy patterns for DisGenerator.
"""
Disaggregated Serving Proxy for the NixlConnector (XpYd) architecture.

Two-phase request flow (per vLLM's NIXL integration protocol):

  Phase 1 (Prefill):
    - Forward the request to a prefill instance with max_tokens=1, stream=False
      and kv_transfer_params={"do_remote_decode": true, ...}.
    - The prefill response's kv_transfer_params carries the remote engine id,
      block ids, host and port of the computed KV cache.

  Phase 2 (Decode):
    - Forward the ORIGINAL request to a decode instance with the
      kv_transfer_params returned by prefill. The decode instance pulls the KV
      cache directly from the prefill instance via NIXL (RDMA/UCX) and
      generates tokens, streaming back to the client.

Unlike the previous P2P NCCL design there is no ZMQ service discovery: the
prefill/decode instance lists are static, passed via CLI flags or environment
variables. The instance lists are also exposed on GET /servers so the
DisGenerator PolicyManager can discover every vLLM server it must push LoRA
adapters to.

Configuration (env or CLI):
  PREFILL_INSTANCES  - comma-separated host:port list (default: localhost:20001)
  DECODE_INSTANCES   - comma-separated host:port list (default: localhost:20002)
  PROXY_IP           - bind address                    (default: 0.0.0.0)
  PROXY_HTTP_PORT    - API port                        (default: 10001)
"""

import argparse
import itertools
import logging
import os
from typing import Optional

import aiohttp
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, Response, StreamingResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] [Proxy] %(message)s")
logger = logging.getLogger("DisaggProxy")

# Configuration
PROXY_IP = os.getenv("PROXY_IP", "0.0.0.0")
PROXY_HTTP_PORT = int(os.getenv("PROXY_HTTP_PORT", "10001"))
REQUEST_TIMEOUT_HOURS = int(os.getenv("REQUEST_TIMEOUT_HOURS", "6"))

AIOHTTP_TIMEOUT = aiohttp.ClientTimeout(total=REQUEST_TIMEOUT_HOURS * 60 * 60)

# kv_transfer_params sent with every prefill request. The prefill instance
# fills in remote_engine_id/block_ids/host/port in its response; the decode
# instance uses those to pull the KV cache via NIXL.
NIXL_PREFILL_KV_PARAMS = {
    "do_remote_decode": True,
    "do_remote_prefill": False,
    "remote_engine_id": None,
    "remote_block_ids": None,
    "remote_host": None,
    "remote_port": None,
}


def _parse_instances(value: str) -> list[str]:
    return [inst.strip() for inst in value.split(",") if inst.strip()]


class DisaggProxy:
    def __init__(self, prefill_instances: list[str], decode_instances: list[str]):
        if not prefill_instances or not decode_instances:
            raise ValueError("Need at least one prefill and one decode instance")
        self.prefill_instances = prefill_instances
        self.decode_instances = decode_instances
        self.prefill_cycler = itertools.cycle(prefill_instances)
        self.decode_cycler = itertools.cycle(decode_instances)
        self.count = 0
        self._session: Optional[aiohttp.ClientSession] = None

    async def session(self) -> aiohttp.ClientSession:
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=AIOHTTP_TIMEOUT)
        return self._session

    async def close(self):
        if self._session is not None and not self._session.closed:
            await self._session.close()

    async def handle(self, raw_request: Request, path: str):
        """Two-phase prefill → decode handling for one request."""
        request_data = await raw_request.json()
        stream = bool(request_data.get("stream", False))

        prefill_addr = next(self.prefill_cycler)
        decode_addr = next(self.decode_cycler)
        self.count += 1
        logger.info(f"Request #{self.count}: [P] {prefill_addr} -> [D] {decode_addr}")

        # ------------------------------------------------------------------
        # Phase 1: Prefill (non-streaming, 1 token, KV handshake)
        # ------------------------------------------------------------------
        prefill_request = dict(request_data)
        prefill_request["stream"] = False
        prefill_request.pop("stream_options", None)
        prefill_request["max_tokens"] = 1
        if "max_completion_tokens" in prefill_request:
            prefill_request["max_completion_tokens"] = 1
        prefill_request["kv_transfer_params"] = dict(NIXL_PREFILL_KV_PARAMS)

        session = await self.session()
        try:
            async with session.post(
                f"http://{prefill_addr}{path}", json=prefill_request
            ) as resp:
                if resp.status != 200:
                    # Propagate prefill errors to the client instead of
                    # silently discarding them (old NCCL proxy behavior).
                    body = await resp.read()
                    logger.error(
                        f"Prefill error from {prefill_addr}: {resp.status} - {body[:300]!r}"
                    )
                    return Response(
                        content=body,
                        status_code=resp.status,
                        media_type=resp.content_type or "application/json",
                    )
                prefill_json = await resp.json()
        except aiohttp.ClientError as e:
            logger.error(f"Prefill connection error ({prefill_addr}): {e}")
            return JSONResponse(
                status_code=502,
                content={"error": f"Prefill instance {prefill_addr} unreachable: {e}"},
            )

        kv_transfer_params = prefill_json.get("kv_transfer_params")
        if not kv_transfer_params:
            # Without the handshake the decode instance would recompute the
            # prefill from scratch — functional, but worth flagging loudly.
            logger.warning(
                f"Prefill response from {prefill_addr} carried no kv_transfer_params; "
                f"decode will recompute the full prompt. Is the NixlConnector configured?"
            )

        # ------------------------------------------------------------------
        # Phase 2: Decode (original request + KV handshake params)
        # ------------------------------------------------------------------
        decode_request = dict(request_data)
        if kv_transfer_params:
            decode_request["kv_transfer_params"] = kv_transfer_params

        try:
            resp = await session.post(
                f"http://{decode_addr}{path}", json=decode_request
            )
        except aiohttp.ClientError as e:
            logger.error(f"Decode connection error ({decode_addr}): {e}")
            return JSONResponse(
                status_code=502,
                content={"error": f"Decode instance {decode_addr} unreachable: {e}"},
            )

        if resp.status != 200:
            body = await resp.read()
            resp.release()
            logger.error(f"Decode error from {decode_addr}: {resp.status} - {body[:300]!r}")
            return Response(
                content=body,
                status_code=resp.status,
                media_type=resp.content_type or "application/json",
            )

        if stream:
            async def stream_body():
                try:
                    async for chunk in resp.content.iter_chunked(1024):
                        yield chunk
                finally:
                    resp.release()

            return StreamingResponse(stream_body(), media_type="text/event-stream")

        body = await resp.read()
        media_type = resp.content_type or "application/json"
        resp.release()
        return Response(content=body, status_code=200, media_type=media_type)


def build_app(proxy: DisaggProxy) -> FastAPI:
    app = FastAPI(title="DisGenerator Disaggregated Proxy (NIXL)")

    @app.post("/v1/completions")
    async def completions(raw_request: Request):
        return await proxy.handle(raw_request, "/v1/completions")

    @app.post("/v1/chat/completions")
    async def chat_completions(raw_request: Request):
        return await proxy.handle(raw_request, "/v1/chat/completions")

    @app.get("/health")
    async def health():
        return {
            "status": "healthy",
            "prefill_instances": len(proxy.prefill_instances),
            "decode_instances": len(proxy.decode_instances),
        }

    @app.get("/servers")
    async def servers():
        """Instance lists for PolicyManager LoRA hot-swap fan-out."""
        prefill = [f"http://{addr}" for addr in proxy.prefill_instances]
        decode = [f"http://{addr}" for addr in proxy.decode_instances]
        return {"prefill": prefill, "decode": decode, "all": prefill + decode}

    @app.get("/")
    async def root():
        return {
            "service": "DisGenerator Disaggregated Proxy",
            "kv_connector": "NixlConnector",
            "endpoints": ["/v1/completions", "/v1/chat/completions", "/health", "/servers"],
        }

    @app.on_event("shutdown")
    async def shutdown():
        await proxy.close()

    return app


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser("DisGenerator NIXL disaggregated proxy")
    parser.add_argument(
        "--prefill",
        type=str,
        default=os.getenv("PREFILL_INSTANCES", "localhost:20001"),
        help="Comma-separated prefill instances (host:port)",
    )
    parser.add_argument(
        "--decode",
        type=str,
        default=os.getenv("DECODE_INSTANCES", "localhost:20002"),
        help="Comma-separated decode instances (host:port)",
    )
    parser.add_argument("--host", type=str, default=PROXY_IP)
    parser.add_argument("--port", type=int, default=PROXY_HTTP_PORT)
    return parser.parse_args()


def main():
    args = parse_args()
    prefill_instances = _parse_instances(args.prefill)
    decode_instances = _parse_instances(args.decode)

    print("=" * 60)
    print("DisGenerator Disaggregated Proxy (NixlConnector)")
    print("=" * 60)
    print(f"  Prefill instances: {prefill_instances}")
    print(f"  Decode instances:  {decode_instances}")
    print(f"  HTTP API:          http://{args.host}:{args.port}")
    print(f"  Request timeout:   {REQUEST_TIMEOUT_HOURS} hours")
    print("=" * 60)

    proxy = DisaggProxy(prefill_instances, decode_instances)
    app = build_app(proxy)
    uvicorn.run(app, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
