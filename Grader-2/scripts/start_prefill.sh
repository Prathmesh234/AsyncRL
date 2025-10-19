#!/bin/bash

# Load environment variables
set -a
source "$(dirname "$0")/../.env"
set +a

echo "Starting Prefill Server on GPU ${PREFILL_GPU}..."

# Remove any cached Harmony vocab that may be corrupted.
rm -rf "${HOME}/.cache/tiktoken"

# Launch Prefiller with LMCache
UCX_TLS="${UCX_TLS}" \
    LMCACHE_CONFIG_FILE="$(dirname "$0")/../lmcache-prefiller-config.yaml" \
    CUDA_VISIBLE_DEVICES="${PREFILL_GPU}" \
    HF_TOKEN="${HF_TOKEN}" \
    vllm serve "${MODEL_NAME}" \
    --port "${PREFILL_PORT}" \
    --disable-log-requests \
    --kv-transfer-config \
    '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_producer","kv_connector_extra_config": {"discard_partial_chunks": false, "lmcache_rpc_port": "producer1"}}'
