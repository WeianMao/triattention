#!/usr/bin/env bash
set -euo pipefail

# Ablation: R-KV with prefill protection on DeepSeek-R1-Distill-Qwen-7B (flash_attn2 + bfloat16), 8 draws, seed=888.
# Purpose: Test impact of protecting prefill (question) tokens from compression.
# Difference from run_rkv_aime24_qwen.sh: enables --protect-prefill flag to preserve question tokens.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/aime_sampled8_rkv_aime24_qwen.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/rkv/aime24/protect_prefill" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/aime_sampled8/rkv/aime24/protect_prefill" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/rkv/aime24/protect_prefill/shards" \
  --gpu-memory-threshold 700 \
  --protect-prefill \
  "$@"
