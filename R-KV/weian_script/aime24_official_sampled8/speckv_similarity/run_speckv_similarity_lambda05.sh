#!/usr/bin/env bash
set -euo pipefail

# SpeckV + Similarity Deduplication for AIME24, 8 draws, lambda=0.5

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/sample8_speckv_similarity_lambda05_aime24_official.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/sample8_speckv_similarity_lambda05_aime24_official" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/sample8_speckv_similarity_lambda05_aime24_official" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/sample8_speckv_similarity_lambda05_aime24_official/shards" \
  --sparse-normalize-scores \
  "$@"
