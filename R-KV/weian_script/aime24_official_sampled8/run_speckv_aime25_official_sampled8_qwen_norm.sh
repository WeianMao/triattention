#!/usr/bin/env bash
set -euo pipefail

# Sharded AIME25 SpeckV on DeepSeek-R1-Distill-Qwen-7B (flash_attn2 + bfloat16, no reset/fp32_topk), 8 draws, with score normalization enabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/sample8_speckv_aime25_official_qwen.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/sample8_speckv_aime25_official_qwen_norm" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/sample8_speckv_aime25_official_qwen_norm" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/sample8_speckv_aime25_official_qwen_norm/shards" \
  --sparse-normalize-scores \
  "$@"
