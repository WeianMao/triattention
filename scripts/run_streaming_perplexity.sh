#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

JSON_DIR="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/offline_reasoning_json"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/perplexity_stream_test"

"${PYTHON_BIN}" "${PROJECT_ROOT}/weian_development/streaming_perplexity/run_streaming_perplexity_distributed.py" \
  "${JSON_DIR}" "${OUTPUT_DIR}" \
  --gpus "0" \
  --limit-files 1 \
  --limit-traces 1 \
  --chunk-size 128 \
  --stream-window 2048 \
  --verbose
