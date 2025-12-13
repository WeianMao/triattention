#!/usr/bin/env bash
set -euo pipefail

# NMS-Only Isolated Test: Baseline (NMS Disabled)
# Pure NMS ablation study - no score-based drop interference.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/weian_development/online_k_pruning_viz"
INPUT_ROOT="/data/rbg/users/weian/project/rl/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces"
TRACE="qid0003_trace34"
STATS_TRACE="$INPUT_ROOT/$TRACE"
OUTPUT_ROOT="$SCRIPT_DIR/results/nms_only_baseline"

# Use GPU 1 (baseline control case)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-1} \
conda run -n dc python "$SCRIPT_DIR/attention_pruning_case_study_nms_variance_isolated.py" \
    "$INPUT_ROOT" \
    --trace "$TRACE" \
    --stats-trace "$STATS_TRACE" \
    --head-sample-file "$SCRIPT_DIR/hybrid_sample_heads_lowret_top10.json" \
    --output-root "$OUTPUT_ROOT" \
    --round-window 64 \
    --verbose

echo "NMS-Only baseline (NMS disabled) completed. Results: $OUTPUT_ROOT"
