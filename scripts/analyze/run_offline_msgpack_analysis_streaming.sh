#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

OUTPUT_DIR="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/streaming_hf_64trace"

python3 "${PROJECT_ROOT}/development/example_analyze_offline_serialized.py" \
  --output_dir "${OUTPUT_DIR}" \
  "$@"
