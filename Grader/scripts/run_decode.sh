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
#   uv run python decode_disaggregated.py --cache-uri <prefill-cache-uri> \
#       --control-endpoint tcp://127.0.0.1:5555 --data-endpoint tcp://127.0.0.1:6000

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# GPU configuration for the decode stage. Default to the first available GPU
# but allow overrides by exporting ``CUDA_VISIBLE_DEVICES`` before running.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

CONTROL_ENDPOINT="${CONTROL_ENDPOINT:-tcp://127.0.0.1:5555}"
DATA_ENDPOINT="${DATA_ENDPOINT:-tcp://127.0.0.1:6000}"

exec uv run python decode_disaggregated.py \
  --control-endpoint "${CONTROL_ENDPOINT}" \
  --data-endpoint "${DATA_ENDPOINT}" \
  "$@"
