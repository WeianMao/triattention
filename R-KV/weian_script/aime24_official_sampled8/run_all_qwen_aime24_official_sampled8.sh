#!/usr/bin/env bash
set -euo pipefail

# Sequentially run FullKV → R-KV → SnapKV on DeepSeek-R1-Distill-Qwen-7B (8 draws).
# Additional args are forwarded to each sub-script.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

bash "${SCRIPT_DIR}/run_fullkv_aime24_official_sampled8_qwen.sh" "$@"
bash "${SCRIPT_DIR}/run_rkv_aime24_official_sampled8_qwen.sh" "$@"
bash "${SCRIPT_DIR}/run_snapkv_aime24_official_sampled8_qwen.sh" "$@"
