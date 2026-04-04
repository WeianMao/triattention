#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DFS_DIR="${SCRIPT_DIR}/.."

PYTHON_BIN="python"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-8B}"
DATASET_PATH="${DFS_DIR}/datasets/dfs_state_query_100.json"
OUTPUT_PATH="${DFS_DIR}/analysis/dfs_state_query_eval_100.json"
MAX_SAMPLES=""
MAX_NEW_TOKENS="32768"
VERBOSE="1"

ARGS=(
  "${SCRIPT_DIR}/eval_dfs_state_query.py"
  --dataset "${DATASET_PATH}"
  --model-path "${MODEL_PATH}"
  --output "${OUTPUT_PATH}"
  --max-new-tokens "${MAX_NEW_TOKENS}"
)

if [[ -n "${MAX_SAMPLES}" ]]; then
  ARGS+=(--max-samples "${MAX_SAMPLES}")
fi

if [[ "${VERBOSE}" == "1" ]]; then
  ARGS+=(--verbose)
fi

mkdir -p "$(dirname "${OUTPUT_PATH}")"
exec "${PYTHON_BIN}" "${ARGS[@]}"
