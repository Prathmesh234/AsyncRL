#!/bin/bash

# Load environment variables
set -a
source "$(dirname "$0")/../.env"
set +a

echo "Starting Decode Server on GPU ${DECODE_GPU}..."

# Launch Decoder with LMCache
UCX_TLS="${UCX_TLS}" \
    LMCACHE_CONFIG_FILE="$(dirname "$0")/../lmcache-decoder-config.yaml" \
    CUDA_VISIBLE_DEVICES="${DECODE_GPU}" \
    HF_TOKEN="${HF_TOKEN}" \
    vllm serve "${MODEL_NAME}" \
    --port "${DECODE_PORT}" \
    --disable-log-requests \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_consumer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "consumer1"}}'
