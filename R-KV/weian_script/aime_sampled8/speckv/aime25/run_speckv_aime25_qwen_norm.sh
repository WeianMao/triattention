#!/usr/bin/env bash
set -euo pipefail

# Sharded AIME25 SpeckV on DeepSeek-R1-Distill-Qwen-7B (flash_attn2 + bfloat16), 8 draws, seed=888, with score normalization enabled.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/aime_sampled8_speckv_aime25_qwen_norm.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime25/norm" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/aime_sampled8/speckv/aime25/norm" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime25/norm/shards" \
  --sparse-normalize-scores \
  "$@"
