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

EXTRA_CONFIG="${EXTRA_CONFIG:-${EXP_ROOT}/configs/extra_config/speckv_disable_mlr.yaml}"

DATASETS=(aime24 aime25 math500)
MODELS=("Qwen3-8B")

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    python "${RKV_ROOT}/weian_development/speckv_experiments_cli_v2.py" "${EXTRA_ARGS[@]}" run-one \
      --dataset "$dataset" \
      --model "$model" \
      --method speckv \
      --extra-config "${EXTRA_CONFIG}"
  done
done
