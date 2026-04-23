#!/usr/bin/env bash

set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
METRIC="${1:-energy}"

POOLINGS=("max" "p95" "median")

for pooling in "${POOLINGS[@]}"; do
  echo
  echo "============================================================"
  echo "Running logit_baselines.py global --metric ${METRIC} --pooling ${pooling}"
  echo "============================================================"
  "${PYTHON_BIN}" logit_baselines.py global \
    --metric "${METRIC}" \
    --pooling "${pooling}" \
    --save-json
done

echo
echo "Finished all logit-baseline pooling runs for metric: ${METRIC}"
