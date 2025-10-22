#!/usr/bin/env bash
set -euo pipefail

if [ -f .env ]; then
  echo "Loading environment variables from .env"
  set -a
  # shellcheck source=/dev/null
  source .env
  set +a
else
  echo "No .env file found at container startup"
fi

export CUDA_DEVICE_ORDER=${CUDA_DEVICE_ORDER:-PCI_BUS_ID}
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}
export PREFILL_GPU=${PREFILL_GPU:-0}
export DECODE_GPU=${DECODE_GPU:-1}

echo "Configured GPUs -> Prefill: GPU ${PREFILL_GPU}, Decode: GPU ${DECODE_GPU}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

cd scripts
exec ./start_all.sh
