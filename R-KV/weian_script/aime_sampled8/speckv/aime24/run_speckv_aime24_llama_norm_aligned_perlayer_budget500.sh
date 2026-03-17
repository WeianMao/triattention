#!/usr/bin/env bash
set -euo pipefail

# Sharded AIME24 SpeckV ALIGNED + PER_LAYER on DeepSeek-R1-Distill-Llama-8B (flash_attn2 + bfloat16), 8 draws, seed=888.
# kv_budget=500 for low-budget comparison with R-KV.
# Purpose: per-layer pruning (all KV heads in same layer share same tokens, different layers select independently).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/aime_sampled8_speckv_aime24_llama_norm_aligned_budget500.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perlayer_llama_budget500" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/aime_sampled8/speckv/aime24/norm_aligned_perlayer_llama_budget500" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perlayer_llama_budget500/shards" \
  --dataset aime24 \
  --sparse-normalize-scores \
  --include-prefill-in-budget \
  --rkv-style-compression \
  --rkv-style-slack-trigger \
  --divide-length 128 \
  --per-layer-pruning \
  "$@"
