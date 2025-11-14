#!/usr/bin/env bash
set -euo pipefail

# Config-driven Window_LAZY launch using the shared dispatcher.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"

python3 "${PROJECT_ROOT}/weian_development/lazy_eviction_sparse_prefill_keep_dispatch.py" \
  --config "${PROJECT_ROOT}/LazyEviction/weian_script/configs/window_lazy_aime.yaml" \
  "$@"
