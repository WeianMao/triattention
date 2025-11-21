#!/usr/bin/env bash
set -euo pipefail

# Third sharded run (second duplicate) pointing to separate dirs for comparison.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/rkv_aime24_sharded.yaml" \
  --log-dir "R-KV/logs/rkv_aime24_sharded_run3" \
  --method-output-dir "R-KV/outputs/rkv_aime24_sharded_run3" \
  --output-dir "R-KV/outputs/rkv_aime24_sharded_run3/shards" \
  "$@"
