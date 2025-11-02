#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

INPUT_ROOT="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces"
OUTPUT_ROOT="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/freq_magnitude_plots"
MODEL_PATH="${MODEL_PATH:-/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B}"
DEVICE="cuda:1"
DTYPE="float32"

python "${PROJECT_ROOT}/weian_development/attention_qk_analysis/freq_magnitude_plots.py" \
  "${INPUT_ROOT}" \
  --output-root "${OUTPUT_ROOT}" \
  --device "${DEVICE}" \
  --dtype "${DTYPE}" \
  --model-path "${MODEL_PATH}" \
  "$@"
