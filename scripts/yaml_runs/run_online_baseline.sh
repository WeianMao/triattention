#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export VLLM_TORCH_COMPILE_CACHE_PATH="${VLLM_TORCH_COMPILE_CACHE_PATH:-.cache/vllm}"
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

python3 "${PROJECT_ROOT}/scripts/yaml_runs/run_dispatch.py" --mode baseline "$@"
