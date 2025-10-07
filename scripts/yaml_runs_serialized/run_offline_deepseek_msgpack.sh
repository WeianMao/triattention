#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export VLLM_TORCH_COMPILE_CACHE_PATH="${VLLM_TORCH_COMPILE_CACHE_PATH:-/data/rbg/users/weian/project/rl/cache/vllm}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PROJECT_ROOT}/scripts/yaml_runs_serialized/run_dispatch_serialized.py" \
  --mode offline \
  --config "${PROJECT_ROOT}/scripts/configs/deepseek_r1_qwen3_8b_32trace.yaml" \
  --rid 0 \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-dir "${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/offline_msgpack" \
  --serializer msgpack_gzip "$@"
