#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RKV_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"

export PYTHONPATH="${RKV_ROOT}:${PYTHONPATH:-}"

DRY_RUN="${DRY_RUN:-0}"
JOB_PARALLEL="${JOB_PARALLEL:-1}"
if ! [[ "${JOB_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${JOB_PARALLEL}" -lt 1 ]]; then
  echo "JOB_PARALLEL must be a positive integer (got: ${JOB_PARALLEL})" >&2
  exit 1
fi

GLOBAL_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  GLOBAL_ARGS+=("--dry-run")
fi

DATASETS=(aime24 aime25 math500)
MODEL="DeepSeek-R1-Distill-Qwen-7B"
SUBCOMMAND_ARGS=(--job-parallel "${JOB_PARALLEL}" --model "${MODEL}")
for dataset in "${DATASETS[@]}"; do
  SUBCOMMAND_ARGS+=("--dataset" "${dataset}")
done

python "${RKV_ROOT}/weian_development/speckv_experiments_cli_v2.py" "${GLOBAL_ARGS[@]}" \
  build-stats "${SUBCOMMAND_ARGS[@]}"
