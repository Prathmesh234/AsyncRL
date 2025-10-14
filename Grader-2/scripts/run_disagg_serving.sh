#!/usr/bin/env bash
# run_disagg_serving.sh - helper script to run the disaggregated prefill service
# without requiring ZeroMQ endpoints. This is effectively a thin wrapper around
# ``prefill_disaggregated.py`` that keeps the command-line identical to the
# original grader configuration while relying on LMCache's NIXL transport.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"

exec "${SCRIPT_DIR}/run_prefill.sh" "$@"
