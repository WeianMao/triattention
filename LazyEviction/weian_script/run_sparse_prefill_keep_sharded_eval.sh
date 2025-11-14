#!/usr/bin/env bash
set -euo pipefail

# Thin wrapper that mirrors scripts/yaml_runs_serialized/run_offline_* style.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"

python3 "${PROJECT_ROOT}/weian_development/lazy_eviction_sparse_prefill_keep_dispatch.py" \
  --config "${PROJECT_ROOT}/LazyEviction/weian_script/configs/sparse_prefill_keep_aime.yaml" \
  "$@"
