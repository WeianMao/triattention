#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
INPUT_ROOT="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/attention_maps_full2"
DEVICE="cuda:0"
PATCH_SIZE=32
HEAD_BATCH=8
Q_TILE=4096
TARGET_SIZE=4096
DTYPE="float32"

mkdir -p "${OUTPUT_ROOT}"

python "${PROJECT_ROOT}/weian_development/attention_qk_analysis/visualize_attention_maps.py" \
  "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  --patch-size "${PATCH_SIZE}" \
  --head-batch "${HEAD_BATCH}" \
  --q-tile "${Q_TILE}" \
  --target-size "${TARGET_SIZE}" \
  --dtype "${DTYPE}" \
  --verbose "$@"
