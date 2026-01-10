#!/usr/bin/env bash
set -euo pipefail

# Editable defaults (override with flags).
DATASET="aime24"
MODEL="DeepSeek-R1-Distill-Qwen-7B"
METHOD="rkv"
BUDGET=""
DRY_RUN="0"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RKV_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"

usage() {
  cat <<USAGE
Usage: bash scripts/run_one.sh [--dataset name] [--model name] [--method fullkv|rkv|speckv] [--budget N] [--dry-run]

If --budget is omitted for rkv/speckv, default_budget from configs/shared/defaults.yaml is used.
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --model)
      MODEL="$2"
      shift 2
      ;;
    --method)
      METHOD="$2"
      shift 2
      ;;
    --budget)
      BUDGET="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN="1"
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage
      exit 1
      ;;
  esac
done

export PYTHONPATH="${RKV_ROOT}:${PYTHONPATH:-}"

ARGS=("--dataset" "$DATASET" "--model" "$MODEL" "--method" "$METHOD")
if [[ -n "$BUDGET" ]]; then
  ARGS+=("--budget" "$BUDGET")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=("--dry-run")
fi

python "${RKV_ROOT}/weian_development/speckv_experiments_cli_v2.py" run-one "${ARGS[@]}"
