#!/usr/bin/env bash
set -euo pipefail

# Sharded AIME25 SpeckV ALIGNED on DeepSeek-R1-Distill-Qwen-7B (flash_attn2 + bfloat16), 8 draws, seed=888.
# Aligned version: cross-dataset stats, reduced round_window, prefill in budget, R-KV style compression.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/aime_sampled8_speckv_aime25_qwen_norm_aligned.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime25/norm_aligned" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/aime_sampled8/speckv/aime25/norm_aligned" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime25/norm_aligned/shards" \
  --sparse-normalize-scores \
  --include-prefill-in-budget \
  --rkv-style-compression \
  --rkv-style-slack-trigger \
  --divide-length 128 \
  "$@"
