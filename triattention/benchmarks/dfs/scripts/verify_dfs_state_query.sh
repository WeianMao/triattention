#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DFS_DIR="${SCRIPT_DIR}/.."

PYTHON_BIN="python"
DATASET_PATH="${DFS_DIR}/datasets/dfs_state_query_100.json"
VERIFY_ALL="1"
SAMPLE_IDS=("0")
VERBOSE="0"
FILTER_INVALID="0"
FILTER_OUTPUT_PATH="${DFS_DIR}/datasets/dfs_state_query_100_clean.json"
VISUALIZE_ID=""

ARGS=("${SCRIPT_DIR}/verify_dfs_dataset.py" "${DATASET_PATH}")

if [[ -n "${VISUALIZE_ID}" ]]; then
  ARGS+=(--visualize "${VISUALIZE_ID}")
elif [[ "${FILTER_INVALID}" == "1" ]]; then
  ARGS+=(--filter --output "${FILTER_OUTPUT_PATH}")
elif [[ "${VERIFY_ALL}" == "1" ]]; then
  ARGS+=(--verify-all)
else
  if [[ "${#SAMPLE_IDS[@]}" -gt 0 ]]; then
    ARGS+=(--sample-id "${SAMPLE_IDS[@]}")
  fi
fi

if [[ "${VERBOSE}" == "1" ]]; then
  ARGS+=(--verbose)
fi

exec "${PYTHON_BIN}" "${ARGS[@]}"
