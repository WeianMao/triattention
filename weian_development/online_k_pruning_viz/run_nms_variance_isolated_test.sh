#!/usr/bin/env bash
set -euo pipefail

# Variance-Aware NMS Isolated Test: NMS Enabled (p50/p50)
# This script runs the isolated variance NMS experiment WITH NMS enabled
# using median percentiles (p50/p50). Uses identical input parameters as
# baseline except for NMS flags.

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/weian_development/online_k_pruning_viz"
INPUT_ROOT="/data/rbg/users/weian/project/rl/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces"
TRACE="qid0003_trace34"
STATS_TRACE="$INPUT_ROOT/$TRACE"
OUTPUT_ROOT="$SCRIPT_DIR/results/isolated_nms_p50_p50"

# Use GPU 2 (test case with NMS)
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-2} \
conda run -n dc python "$SCRIPT_DIR/attention_pruning_case_study_nms_variance_isolated.py" \
    "$INPUT_ROOT" \
    --trace "$TRACE" \
    --stats-trace "$STATS_TRACE" \
    --head-sample-file "$SCRIPT_DIR/hybrid_sample_heads_lowret_top10.json" \
    --output-root "$OUTPUT_ROOT" \
    --max-keys 2048 \
    --round-window 64 \
    --nms-enabled \
    --low-percentile 50.0 \
    --high-percentile 50.0 \
    --verbose

echo "Isolated NMS test (p50/p50) experiment completed. Results saved to: $OUTPUT_ROOT"
