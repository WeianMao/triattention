#!/usr/bin/env bash
set -euo pipefail

# Fair comparison R-KV script (aligned with sparse_prefill_keep)
# Uses round-based compression: round_window=363, round_base_budget=1129
# This ensures average decode KV ~1310, matching sparse_prefill_keep

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"

# Use the fair comparison config
CONFIG_PATH="${PROJECT_ROOT}/LazyEviction/weian_script/configs/rkv_lazy_aime_fair.yaml"
DISPATCH_PATH="${PROJECT_ROOT}/weian_development/rkv_lazy_dispatch.py"

if [[ ! -f "${DISPATCH_PATH}" ]] || [[ ! -f "${CONFIG_PATH}" ]]; then
  echo "RKV dispatch or fair config missing." >&2
  exit 1
fi

echo "Running R-KV with fair comparison settings (aligned with sparse_prefill_keep)"
echo "Config: ${CONFIG_PATH}"
echo "Key params: round_window=363, round_base_budget=1129, reset_cache_each_batch=true"
echo ""

python3 "${DISPATCH_PATH}" \
  --config "${CONFIG_PATH}" \
  "$@"
