#!/usr/bin/env bash
set -euo pipefail

# Sharded AIME24 FullKV + SIMULATED ATTENTION POSITION OFFSET on DeepSeek-R1-Distill-Qwen-7B.
#
# This experiment simulates bug 896cbca6's attention position offset effect on fullkv baseline
# to verify if the position encoding shift improves performance independently of KV compression.
#
# Bug 896cbca6 effect:
#   - Prefill K positions: [X, X+1, ..., X+P-1]  (X = accumulated value)
#   - Decode Q positions:  [X, X+1, ...]         (starts from same X, NOT X+P)
#   - Relative positions:  Q - K = -k            (instead of normal P-k)
#   - Effect: Decode "sees" prefill tokens as much closer (at position 0, -1, -2, ... instead of P, P-1, ...)
#
# Simulation method:
#   - Prefill: normal positions [0, 1, ..., P-1]
#   - Decode: subtract prefill_len, so positions become [0, 1, ...] instead of [P, P+1, ...]
#   - This makes relative positions = 0 - k = -k, exactly matching the bug effect
#
# If this experiment improves performance, it confirms that the "closer prefill" attention effect
# works independently of KV compression (validated on fullkv baseline).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/aime_sampled8_fullkv_aime24_qwen.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/fullkv/aime24/simulated_pos_offset" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/aime_sampled8/fullkv/aime24/simulated_pos_offset" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/fullkv/aime24/simulated_pos_offset/shards" \
  --simulate-attention-position-offset 1 \
  "$@"
