#!/usr/bin/env bash
set -euo pipefail

# Sharded AIME24 R-KV on DeepSeek-R1-Distill-Llama-8B (flash_attn2 + bfloat16), 8 draws, seed=888.
# kv_budget=500 for low-budget comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/aime_sampled8_rkv_aime24_llama_budget500.yaml" \
  --dataset aime24 \
  "$@"
