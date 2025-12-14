#!/usr/bin/env bash
set -euo pipefail

# Spectrum-Aware NMS Experiment: Meanvec Energy Method
# This script runs the pruning experiment WITH NMS using meanvec energy weights
# Energy formula: E_f = |E[q_f]| * |E[k_f]| (mean vector product)
# Reference: freq_magnitude_single_plot_meanvec_scatter.py mean_vector_product()

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/weian_development/online_k_pruning_viz"
INPUT_ROOT="/data/rbg/users/weian/project/rl/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces"
TRACE="qid0003_trace34"
STATS_TRACE="$INPUT_ROOT/$TRACE"
OUTPUT_ROOT="$SCRIPT_DIR/results/nms_meanvec"

# Use GPU 3
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-3} \
conda run -n dc python "$SCRIPT_DIR/attention_pruning_case_study_hybrid_rounds_xtrace_nms.py" \
    "$INPUT_ROOT" \
    --trace "$TRACE" \
    --stats-trace "$STATS_TRACE" \
    --head-sample-file "$SCRIPT_DIR/hybrid_sample_heads_lowret_top10.json" \
    --output-root "$OUTPUT_ROOT" \
    --max-keys 2048 \
    --round-window 64 \
    --nms-enabled \
    --energy-method meanvec \
    --verbose

echo "NMS Meanvec experiment completed. Results saved to: $OUTPUT_ROOT"
