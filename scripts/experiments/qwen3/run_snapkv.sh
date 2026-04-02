#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
EXP_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
RKV_ROOT="$(cd "${EXP_ROOT}/.." && pwd)"

export PYTHONPATH="${RKV_ROOT}:${PYTHONPATH:-}"

usage() {
  cat <<USAGE
Usage: bash scripts/qwen3/run_snapkv.sh [--budget N]

Runs the Qwen3-8B SnapKV sweep for aime24/aime25/math500. When --budget is
provided, the underlying CLI receives the supplied budget. Omitting the flag
falls back to the default budget defined in configs/shared/defaults.yaml.
USAGE
}

BUDGET=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --budget)
      BUDGET="$2"
      shift 2
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

if [[ -n "${BUDGET}" ]] && ! [[ "${BUDGET}" =~ ^[0-9]+$ ]]; then
  echo "--budget expects a positive integer (got: ${BUDGET})" >&2
  exit 1
fi

DRY_RUN="${DRY_RUN:-0}"
EXTRA_ARGS=()
if [[ "${DRY_RUN}" == "1" ]]; then
  EXTRA_ARGS+=("--dry-run")
fi

BUDGET_ARGS=()
if [[ -n "${BUDGET}" ]]; then
  BUDGET_ARGS+=("--budget" "${BUDGET}")
fi

JOB_PARALLEL="${JOB_PARALLEL:-1}"
if ! [[ "${JOB_PARALLEL}" =~ ^[0-9]+$ ]] || [[ "${JOB_PARALLEL}" -lt 1 ]]; then
  echo "JOB_PARALLEL must be a positive integer (got: ${JOB_PARALLEL})" >&2
  exit 1
fi

JOB_PIDS=()
JOB_LABELS=()

wait_for_oldest_job() {
  if [[ "${#JOB_PIDS[@]}" -eq 0 ]]; then
    return
  fi
  local pid="${JOB_PIDS[0]}"
  local label="${JOB_LABELS[0]}"
  wait "${pid}"
  local status=$?
  if [[ "${status}" -ne 0 ]]; then
    echo "[error] Job ${label} failed with status ${status}" >&2
    exit "${status}"
  fi
  JOB_PIDS=("${JOB_PIDS[@]:1}")
  JOB_LABELS=("${JOB_LABELS[@]:1}")
}

launch_job() {
  local dataset="$1"
  local model="$2"
  (
    python "${RKV_ROOT}/scripts/cli.py" "${EXTRA_ARGS[@]}" run-one \
      --dataset "$dataset" \
      --model "$model" \
      --method snapkv \
      "${BUDGET_ARGS[@]}"
  ) &
  local pid=$!
  JOB_PIDS+=("${pid}")
  JOB_LABELS+=("${dataset}/${model}")
  if [[ "${#JOB_PIDS[@]}" -ge "${JOB_PARALLEL}" ]]; then
    wait_for_oldest_job
  fi
}

DATASETS=(aime24 aime25 math500)
MODELS=("Qwen3-8B")

declare -a JOB_QUEUE=()
for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    JOB_QUEUE+=("${dataset}/${model}")
  done
done

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "[dry-run] JOB_PARALLEL=${JOB_PARALLEL}; budget=${BUDGET:-default}; planned batches:"
  batch=1
  idx=0
  total=${#JOB_QUEUE[@]}
  while (( idx < total )); do
    chunk=("${JOB_QUEUE[@]:idx:JOB_PARALLEL}")
    echo "  batch ${batch}: ${chunk[*]}"
    idx=$((idx + JOB_PARALLEL))
    batch=$((batch + 1))
  done
fi

for dataset in "${DATASETS[@]}"; do
  for model in "${MODELS[@]}"; do
    launch_job "$dataset" "$model"
  done
done

while [[ "${#JOB_PIDS[@]}" -gt 0 ]]; do
  wait_for_oldest_job
done
