#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RKV_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"

export PYTHONPATH="${RKV_ROOT}:${PYTHONPATH:-}"

DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=("--dry-run")
fi

python "${RKV_ROOT}/weian_development/speckv_experiments_cli_v2.py" "${EXTRA_ARGS[@]}" build-stats
