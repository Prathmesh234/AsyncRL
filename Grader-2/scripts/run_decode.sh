#!/usr/bin/env bash
# run_decode.sh - helper script to start the disaggregated decode worker.
#
# Usage:
#   ./scripts/run_decode.sh --cache-uri <prefill-cache-uri> [extra-args]
# The helper sets up CUDA visibility and the LMCache client defaults before
# invoking ``decode_disaggregated.py`` with ``uv run``. Additional CLI arguments
# are forwarded directly to the Python script.
#
# To call the decode entrypoint manually instead of this helper, run:
#   uv run python decode_disaggregated.py --cache-uri <prefill-cache-uri>

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# GPU configuration for the decode stage. Default to the first available GPU
# but allow overrides by exporting ``CUDA_VISIBLE_DEVICES`` before running.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CACHE_NAMESPACE="${CACHE_NAMESPACE:-grader/prefill}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-2}"

exec uv run python decode_disaggregated.py \
  --model "${MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --cache-namespace "${CACHE_NAMESPACE}" \
  "$@"
