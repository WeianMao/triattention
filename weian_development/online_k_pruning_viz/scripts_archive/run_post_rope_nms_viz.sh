#!/usr/bin/env bash
set -euo pipefail

# Post-RoPE Key NMS Visualization
# This script runs the post-RoPE key magnitude NMS scatter plot visualization
# for sampled attention heads using variance-aware percentile thresholds.

# Configuration
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0}
TRACE="qid0003_trace34"
INPUT_DIR="/data/rbg/users/weian/project/rl/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces"
HEAD_SAMPLE_FILE="hybrid_sample_heads_lowret_top10.json"
LOW_PERCENTILE=20
HIGH_PERCENTILE=80
OUTPUT_DIR="post_rope_nms_viz_output"

# Get script directory for proper path resolution
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/weian_development/online_k_pruning_viz"

# Construct full trace path
TRACE_PATH="${INPUT_DIR}/${TRACE}"

echo "Running post-RoPE NMS visualization..."
echo "Trace: $TRACE_PATH"
echo "Head sample file: $HEAD_SAMPLE_FILE"
echo "Percentile range: ${LOW_PERCENTILE}-${HIGH_PERCENTILE}"
echo "Output directory: $OUTPUT_DIR"

# Execute visualization script with conda environment
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES \
conda run -n dc python "$SCRIPT_DIR/post_rope_key_nms_scatter.py" \
    --trace "$TRACE_PATH" \
    --head-sample-file "$SCRIPT_DIR/$HEAD_SAMPLE_FILE" \
    --low-percentile $LOW_PERCENTILE \
    --high-percentile $HIGH_PERCENTILE \
    --output-dir "$OUTPUT_DIR" \
    --input-root "$INPUT_DIR"

echo "Visualization complete. Output saved to: $OUTPUT_DIR"
