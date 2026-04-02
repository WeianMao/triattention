#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
EXP_ROOT="$(cd "${QWEN3_ROOT}/../.." && pwd)"

EXTRA_CONFIG="${EXP_ROOT}/configs/extra_config/triattention_per_head_pruning_allow_prefill.yaml"
export EXTRA_CONFIG

exec "${QWEN3_ROOT}/run_triattention_per_head.sh" --budget 512 "$@"
