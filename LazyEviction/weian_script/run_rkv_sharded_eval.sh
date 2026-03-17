#!/usr/bin/env bash
set -euo pipefail

# Placeholder entrypoint for the LazyEviction-native RKV rebuild.
# Mirrors the thin-wrapper style of run_sparse_prefill_keep_sharded_eval.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"

CONFIG_PATH="${PROJECT_ROOT}/LazyEviction/weian_script/configs/rkv_lazy_aime.yaml"
DISPATCH_PATH="${PROJECT_ROOT}/weian_development/rkv_lazy_dispatch.py"

if [[ ! -f "${DISPATCH_PATH}" ]] || [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "RKV rebuild dispatch or config missing. See LazyEviction/docs/rkv_rebuild/status_overview.md for the plan and TODOs." >&2
  exit 1
fi

python3 "${DISPATCH_PATH}" \
  --config "${CONFIG_PATH}" \
  "$@"
