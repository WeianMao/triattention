#!/usr/bin/env bash
set -euo pipefail

# Ablation: simulate_bug_phase_offset=156
# Simulate Bug 896cbca6 phase offset effect (Δ=156 tokens, average prefill length)
#
# Background:
#   Bug 896cbca6 caused phi to be offset by -Δ×ω in scoring function.
#   This experiment verifies if the phase offset is responsible for the bug's
#   unexpected performance improvement.
#
# Expected: If phase offset is the cause, this should replicate the bug's
#   performance improvement on the fixed codebase.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../../../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}/R-KV:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

python3 "${PROJECT_ROOT}/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "${PROJECT_ROOT}/R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml" \
  --method-output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime24/phase_offset_ablation/offset_156" \
  --log-dir "${PROJECT_ROOT}/R-KV/logs/aime_sampled8/speckv/aime24/phase_offset_ablation/offset_156" \
  --output-dir "${PROJECT_ROOT}/R-KV/outputs/aime_sampled8/speckv/aime24/phase_offset_ablation/offset_156/shards" \
  --sparse-normalize-scores \
  --include-prefill-in-budget \
  --rkv-aligned-budget \
  --divide-length 128 \
  --simulate-bug-phase-offset 156 \
  "$@"
