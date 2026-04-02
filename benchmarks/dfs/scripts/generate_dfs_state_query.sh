#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DFS_DIR="${SCRIPT_DIR}/.."

PYTHON_BIN="python"
NUM_SAMPLES="300"
MIN_NODES="10"
MAX_NODES="30"
MIN_STEPS="1"
MAX_STEPS="30"
GRAPH_TYPES=("tree" "sparse")
SEED="42"
OUTPUT_PATH="${DFS_DIR}/datasets/dfs_state_query_300.json"
SHOW_SAMPLE="0"

ARGS=(
  "${SCRIPT_DIR}/generate_dfs_state_dataset.py"
  --num-samples "${NUM_SAMPLES}"
  --min-nodes "${MIN_NODES}"
  --max-nodes "${MAX_NODES}"
  --min-steps "${MIN_STEPS}"
  --max-steps "${MAX_STEPS}"
  --seed "${SEED}"
  --output "${OUTPUT_PATH}"
)

if [[ "${#GRAPH_TYPES[@]}" -gt 0 ]]; then
  ARGS+=(--graph-types "${GRAPH_TYPES[@]}")
fi

if [[ "${SHOW_SAMPLE}" == "1" ]]; then
  ARGS+=(--show-sample)
fi

exec "${PYTHON_BIN}" "${ARGS[@]}"
