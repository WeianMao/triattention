#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SCRIPT_DIR="$ROOT_DIR/weian_development/hf_offline_runner_sparse"
MODEL_PATH="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"
DATA_PATH="$ROOT_DIR/aime25.jsonl"
STATS_PATH="$SCRIPT_DIR/stats/qid0008_trace46_stats.pt"
STREAM_LOG="$ROOT_DIR/tmp_eval/stream_qid0.txt"

mkdir -p "$ROOT_DIR/tmp_eval"

CUDA_VISIBLE_DEVICES=1 \
conda run -n dc python "$SCRIPT_DIR/example_offline_hf_serialized_streaming.py" \
  --model "$MODEL_PATH" \
  --dataset "$DATA_PATH" \
  --qid 0 \
  --rid stream_default \
  --model_type deepseek \
  --max_tokens -1 \
  --temperature 0 \
  --top_p 1 \
  --top_k 0 \
  --enable_sparse_pruning \
  --sparse-stats-path "$STATS_PATH" \
  --sparse-max-keys 2048 \
  --sparse-round-window 64 \
  --sparse-offset-max-length 2048 \
  --output_dir "$ROOT_DIR/tmp_eval" \
  --stream-log-path "$STREAM_LOG" \
  --no_store_logprobs
