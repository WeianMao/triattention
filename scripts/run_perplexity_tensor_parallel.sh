#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

JSON_DIR="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/offline_reasoning_json"
OUTPUT_DIR="${PROJECT_ROOT}/outputs/deepseek_r1_qwen3_8b/perplexity_full"
GPUS="${PERPLEXITY_GPUS:-0,1,2,3,4,5,6,7}"
CHUNK_SIZE="${PERPLEXITY_CHUNK_SIZE:-256}"

python "${PROJECT_ROOT}/weian_development/run_perplexity_distributed.py" \
  "${JSON_DIR}" \
  "${OUTPUT_DIR}" \
  --gpus "${GPUS}" \
  --chunk-size "${CHUNK_SIZE}" \
  --model-type deepseek \
  --tensor-parallel \
  --verbose "$@"
