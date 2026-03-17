#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
RKV_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"

BASE_DIR=""       # Path to the merged output directory
DATASET="aime24"  # Default dataset name, can be changed to aime25 or math500
OUTPUT_DIR=""     # Just keep it empty to use the default output directory
NUM_SAMPLES="1"   # Modify this to change the number of samples
DRY_RUN="0"

usage() {
  cat <<USAGE
Usage: bash scripts/qwen3/run_eval_only.sh --base-dir PATH --dataset NAME [--output-dir PATH] [--num-samples N] [--dry-run]

Examples:
  bash scripts/qwen3/run_eval_only.sh --base-dir R-KV/speckv_experiments/outputs/aime25/Qwen3-8B/sample64/rkv/budget_256/merged --dataset aime25
  bash scripts/qwen3/run_eval_only.sh --base-dir /abs/path/to/merged --dataset math500 --num-samples 1
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --base-dir)
      BASE_DIR="$2"
      shift 2
      ;;
    --dataset)
      DATASET="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --num-samples)
      NUM_SAMPLES="$2"
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

if [[ -z "${BASE_DIR}" || -z "${DATASET}" ]]; then
  echo "Missing required --base-dir or --dataset." >&2
  usage
  exit 1
fi

export PYTHONPATH="${RKV_ROOT}:${PYTHONPATH:-}"

ARGS=(--base-dir "${BASE_DIR}" --dataset "${DATASET}")
if [[ -n "${OUTPUT_DIR}" ]]; then
  ARGS+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${NUM_SAMPLES}" ]]; then
  ARGS+=(--num-samples "${NUM_SAMPLES}")
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  ARGS+=(--dry-run)
fi

python "${RKV_ROOT}/weian_development/run_eval_only.py" "${ARGS[@]}"
