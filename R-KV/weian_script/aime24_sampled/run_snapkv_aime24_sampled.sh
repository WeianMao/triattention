#!/usr/bin/env bash
set -euo pipefail

# Sharded AIME24 SnapKV with sampling (default 64 draws; set SAMPLES=8 for quick run).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

CONFIG_BASENAME="sample64_snapkv_aime24"
if [[ "${SAMPLES:-64}" == "8" ]]; then
  CONFIG_BASENAME="sample8_snapkv_aime24"
fi

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/${CONFIG_BASENAME}.yaml" \
  "$@"
