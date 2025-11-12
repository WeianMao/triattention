#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LAZY_DIR="$( dirname "${SCRIPT_DIR}" )"
REPO_DIR="$( dirname "${LAZY_DIR}" )"

python "${REPO_DIR}/weian_development/merge_lazy_eviction_shards.py" \
    --method-output-dir "${REPO_DIR}/outputs/DeepSeek-R1-Distill-Qwen-7B/aime/7b/Original/test/Window_LAZY"
