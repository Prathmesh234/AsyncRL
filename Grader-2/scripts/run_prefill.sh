#!/usr/bin/env bash
# run_prefill.sh - helper script to launch the disaggregated prefill worker.
#
# Usage:
#   ./scripts/run_prefill.sh [extra-args]
# This will configure the CUDA devices, set sensible LMCache defaults, and
# execute ``prefill_disaggregated.py`` via ``uv run``. Any extra CLI arguments
# are forwarded to the Python entrypoint, allowing overrides such as
# ``./scripts/run_prefill.sh --cache-namespace grader/custom``.
#
# To run the Python module directly without the helper script, execute:
#   uv run python prefill_disaggregated.py --cache-namespace grader/prefill

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

CACHE_NAMESPACE="${CACHE_NAMESPACE:-grader/prefill}"
MODEL_NAME="${MODEL_NAME:-openai/gpt-oss-120b}"
TP_SIZE="${TP_SIZE:-2}"

# Launch the prefill worker via uv to ensure dependencies from pyproject.toml
# are resolved with the project's virtual environment.
exec uv run python prefill_disaggregated.py \
  --model "${MODEL_NAME}" \
  --tensor-parallel-size "${TP_SIZE}" \
  --cache-namespace "${CACHE_NAMESPACE}" \
  "$@"
