#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PROJECT_ROOT}/weian_development/hf_offline_runner_sparse/run_dispatch_hf_serialized.py" \
  --mode offline \
  --config "${PROJECT_ROOT}/scripts/configs/deepseek_r1_qwen3_8b_64trace.yaml" \
  --rid sparse_2048 \
  --gpus 0,1,2,3,4,5,6,7 \
  --output-dir "${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/offline_hf_sparse" \
  --serializer msgpack_gzip "${@:-}"
