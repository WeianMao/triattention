#!/usr/bin/env bash
set -euo pipefail

# Wrapper: runs the sdpa+fp16+fp32_topk+reset single-GPU baseline (with auto-eval).

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
"${SCRIPT_DIR}/run_rkv_aime24_single.sh" "$@"
