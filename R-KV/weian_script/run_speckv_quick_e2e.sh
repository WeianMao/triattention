#!/usr/bin/env bash
# End-to-end runner: calibrate SpecKV stats (Llama3) and launch the quick 6x8 AIME24 experiment.
set -euo pipefail

ROOT="/data/rbg/users/weian/project/rl/dc"
EXP_NAME="sample8_speckv_aime24_quick_clean"
MODEL_PATH="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B"
TRACE_ROOT="$ROOT/R-KV/outputs/sample8_fullkv_aime24_official"
STATS_OUT="$TRACE_ROOT/stats/deepseek_r1_llama8b_chat_stats.pt"
CONFIG_PATH="$ROOT/R-KV/weian_script/configs/sample8_speckv_aime24_quick.yaml"
OUT_DIR="$ROOT/R-KV/outputs/${EXP_NAME}"
LOG_DIR="$ROOT/R-KV/logs/${EXP_NAME}"
SHARD_DIR="${OUT_DIR}/shards"

# Optional overrides
GPUS="${GPUS:-0}"
NUM_SHARDS="${NUM_SHARDS:-1}"

export PYTHONPATH="$ROOT/R-KV"

echo "[1/3] Calibrating sparse stats for Llama3 (SpecKV)…"
conda run -n rkv python "$ROOT/R-KV/weian_development/rkv_sparse_round_calibrate.py" \
  --trace-root "$TRACE_ROOT" \
  --model-path "$MODEL_PATH" \
  --output-path "$STATS_OUT" \
  --num-traces 3 \
  --use-chat-template \
  --dtype float16

echo "[2/3] Cleaning previous outputs/logs…"
rm -rf "$OUT_DIR" "$LOG_DIR"

echo "[3/3] Launching SpecKV quick run…"
conda run -n rkv python "$ROOT/R-KV/weian_development/rkv_sharded_dispatch.py" \
  --config "$CONFIG_PATH" \
  --gpus "$GPUS" \
  --num-shards "$NUM_SHARDS" \
  --skip-existing \
  --method-output-dir "$OUT_DIR" \
  --log-dir "$LOG_DIR" \
  --output-dir "$SHARD_DIR" \
  --dataset aime24

echo "Done. Outputs: $OUT_DIR ; Logs: $LOG_DIR"
