#!/usr/bin/env bash
set -euo pipefail

# Run Qwen SpeckV AIME24 then AIME25 sequentially (flash_attn2 + bfloat16, plain prompt).

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

CONFIG_AIME24="${PROJECT_ROOT}/R-KV/weian_script/configs/sample8_speckv_aime24_official_qwen.yaml"
CONFIG_AIME25="${PROJECT_ROOT}/R-KV/weian_script/configs/sample8_speckv_aime25_official_qwen.yaml"

echo "[run] SpeckV Qwen AIME24 -> ${CONFIG_AIME24}"
python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${CONFIG_AIME24}" \
  "$@"

echo "[run] SpeckV Qwen AIME25 -> ${CONFIG_AIME25}"
python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${CONFIG_AIME25}" \
  "$@"
