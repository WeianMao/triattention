#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

CONFIG_PATH="${PROJECT_ROOT}/scripts/configs/streaming/deepseek_r1_qwen3_8b_streaming_64.yaml"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/streaming_hf_64trace"

ATTN_BACKEND="${ATTENTION_BACKEND:-flash_attn2}"
EXTRA_ARGS=()
if [[ "${ATTN_BACKEND}" != "auto" ]]; then
  EXTRA_ARGS+=(--attention-backend "${ATTN_BACKEND}")
fi

python3 "${PROJECT_ROOT}/scripts/yaml_runs_streaming/run_dispatch_streaming.py" \
  --config "${CONFIG_PATH}" \
  --rid 0 \
  --output-dir "${OUTPUT_DIR}" \
  --gpus 0,1,2,3,4,5,6,7 \
  "${EXTRA_ARGS[@]}" \
  "$@"
