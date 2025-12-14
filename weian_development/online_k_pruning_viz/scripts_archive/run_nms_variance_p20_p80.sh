#!/usr/bin/env bash
set -euo pipefail

# Variance-Aware NMS Experiment: Q-Magnitude Percentile Weights (20/80)
# This script runs the pruning experiment WITH variance-aware NMS using
# conservative weight selection based on Q-magnitude percentiles.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/weian_development/online_k_pruning_viz"
INPUT_ROOT="/data/rbg/users/weian/project/rl/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces"
TRACE="qid0003_trace34"
STATS_TRACE="$INPUT_ROOT/$TRACE"
OUTPUT_ROOT="$SCRIPT_DIR/results/nms_variance_p20_p80"

# Use GPU 2
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} \
conda run -n dc python "$SCRIPT_DIR/attention_pruning_case_study_hybrid_rounds_xtrace_nms_variance.py" \
    "$INPUT_ROOT" \
    --trace "$TRACE" \
    --stats-trace "$STATS_TRACE" \
    --head-sample-file "$SCRIPT_DIR/hybrid_sample_heads_lowret_top10.json" \
    --output-root "$OUTPUT_ROOT" \
    --max-keys 2048 \
    --round-window 64 \
    --nms-enabled \
    --low-percentile 20.0 \
    --high-percentile 80.0 \
    --verbose

echo "Variance-aware NMS (p20/p80) experiment completed. Results saved to: $OUTPUT_ROOT"
