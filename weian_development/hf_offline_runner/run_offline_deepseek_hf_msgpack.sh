#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

RID="${RID:-0}"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/offline_hf_single"
DATASET_PATH="/data/rbg/users/weian/project/rl/deepconf/aime25.jsonl"
ACCURACY_LOG="${OUTPUT_DIR}/accuracy_rid${RID}.json"

python3 "${PROJECT_ROOT}/weian_development/hf_offline_runner/run_dispatch_hf_serialized.py" \
  --mode offline \
  --config "${PROJECT_ROOT}/scripts/configs/deepseek_r1_qwen3_8b_64trace.yaml" \
  --rid "${RID}" \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-dir "${OUTPUT_DIR}" \
  --serializer msgpack_gzip "$@"

python3 "${PROJECT_ROOT}/weian_development/hf_offline_runner/offline_accuracy_report.py" \
  --output-dir "${OUTPUT_DIR}" \
  --rid "${RID}" \
  --dataset "${DATASET_PATH}" \
  --log-path "${ACCURACY_LOG}"
