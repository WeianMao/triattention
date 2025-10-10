#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PROJECT_ROOT}/development/example_analyze_offline_serialized.py" \
  --output_dir "${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/offline_msgpack_64trace" \
  --max_qid 29 \
  --rids 0 \
  --detailed_timing "$@"
