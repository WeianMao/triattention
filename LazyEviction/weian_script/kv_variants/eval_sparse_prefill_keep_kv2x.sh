#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WEIAN_DIR="$( dirname "${SCRIPT_DIR}" )"
LAZY_DIR="$( dirname "${WEIAN_DIR}" )"
REPO_DIR="$( dirname "${LAZY_DIR}" )"

python "${REPO_DIR}/weian_development/merge_lazy_eviction_shards.py" \
    --method-output-dir "${REPO_DIR}/outputs/DeepSeek-R1-Distill-Qwen-7B/aime_prefill_keep_kv2x/7b/Original/test/sparse_round_prefill"
