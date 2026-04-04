#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

INPUT_PATH="${INPUT_PATH:-}"
BINS="${BINS:-0-3,4-6,7-9,10-12,13-15,16-18,19-21,22-24,25-27,28-30,31-9999}"

if [[ -z "${INPUT_PATH}" ]]; then
  echo "Usage: INPUT_PATH=/path/to/raw.jsonl bash $0" >&2
  exit 1
fi

ARGS=(
  "${SCRIPT_DIR}/analyze_dfs_state_query_raw.py"
  --input "${INPUT_PATH}"
  --bins "${BINS}"
)

exec python "${ARGS[@]}"
