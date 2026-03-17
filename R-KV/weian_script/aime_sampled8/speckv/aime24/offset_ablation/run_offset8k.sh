#!/usr/bin/env bash
set -euo pipefail

# Ablation: offset_max_length=8192 (14 elements)
# offsets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime24/offset_ablation/offset8k" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/aime_sampled8/speckv/aime24/offset_ablation/offset8k" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime24/offset_ablation/offset8k/shards" \
  --sparse-normalize-scores \
  --include-prefill-in-budget \
  --rkv-aligned-budget \
  --divide-length 128 \
  --sparse-offset-max-length 8192 \
  "$@"
