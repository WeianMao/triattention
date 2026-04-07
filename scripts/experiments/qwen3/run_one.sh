#!/usr/bin/env bash
set -euo pipefail

DATASET="aime24"
MODEL="Qwen3-8B"
METHOD="r1kv"
BUDGET=""
DRY_RUN="0"
JOB_PARALLEL="${JOB_PARALLEL:-1}"
if ! [[ "${JOB_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${JOB_PARALLEL}" -lt 1 ]]; then
  echo "JOB_PARALLEL must be a positive integer (got: ${JOB_PARALLEL})" >&2
  exit 1
fi
if [[ "${JOB_PARALLEL}" -ne 1 ]]; then
  echo "[warn] run_one.sh runs a single setting; JOB_PARALLEL=${JOB_PARALLEL} is ignored." >&2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RKV_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"

usage() {
  cat <<USAGE
Usage: bash scripts/qwen3/run_one.sh [--dataset name] [--model name] [--method fullkv|r1kv|snapkv|triattention] [--budget N] [--dry-run]

Dataset defaults to aime24 and model defaults to Qwen3-8B.
If --budget is omitted for r1kv/snapkv/triattention, default_budget from triattention/configs/shared/defaults.yaml is used.
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

python "${RKV_ROOT}/scripts/cli.py" run-one "${ARGS[@]}"
