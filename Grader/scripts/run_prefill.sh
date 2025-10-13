#!/usr/bin/env bash
# run_prefill.sh - helper script to launch the disaggregated prefill worker.
#
# Usage:
#   ./scripts/run_prefill.sh [extra-args]
# This will configure the CUDA devices, set sensible LMCache defaults, and
# execute ``prefill_disaggregated.py`` via ``uv run``. Any extra CLI arguments
# are forwarded to the Python entrypoint, allowing overrides such as
# ``./scripts/run_prefill.sh --control-endpoint tcp://0.0.0.0:7000``.
#
# To run the Python module directly without the helper script, execute:
#   uv run python prefill_disaggregated.py --control-endpoint tcp://0.0.0.0:5555 \
#       --data-endpoint tcp://0.0.0.0:6000 --cache-namespace grader/prefill

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
cd "${REPO_ROOT}"

# GPU configuration: default to both local GPUs (0 and 1) matching the
# 2xA100 topology assumed by the prefill worker. Override by exporting the
# variable before invoking the script, e.g. ``CUDA_VISIBLE_DEVICES=3 ./scripts/run_prefill.sh``.
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

# Ensure the repo is importable when the Python script runs.
export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"

# Default LMCache / ZeroMQ endpoints. These can be overridden with env vars
# or via CLI flags when invoking the helper script.
CONTROL_ENDPOINT="${CONTROL_ENDPOINT:-tcp://0.0.0.0:5555}"
DATA_ENDPOINT="${DATA_ENDPOINT:-tcp://0.0.0.0:6000}"
CACHE_NAMESPACE="${CACHE_NAMESPACE:-grader/prefill}"

# Launch the prefill worker via uv to ensure dependencies from pyproject.toml
# are resolved with the project's virtual environment.
exec uv run python prefill_disaggregated.py \
  --control-endpoint "${CONTROL_ENDPOINT}" \
  --data-endpoint "${DATA_ENDPOINT}" \
  --cache-namespace "${CACHE_NAMESPACE}" \
  "$@"
