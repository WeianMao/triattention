#!/usr/bin/env bash
set -euo pipefail

# TriAttention vLLM benchmark: AIME24 with PER_HEAD pruning mode
# Purpose: Per-head pruning (each head selects its own tokens independently)
# Mirrors: R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Environment setup
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

# Activate trivllm conda environment
# Note: This requires conda to be initialized in the shell
if command -v conda &> /dev/null; then
    eval "$(conda shell.bash hook)"
    conda activate trivllm
else
    echo "Warning: conda not found. Please activate trivllm environment manually."
fi

# Model and dataset paths
MODEL_PATH="${MODEL_PATH:-/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B}"
DATASET_PATH="${DATASET_PATH:-${PROJECT_ROOT}/R-KV/HuggingFace/data/aime24.jsonl}"
STATS_PATH="${STATS_PATH:-${PROJECT_ROOT}/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt}"

# Output paths
OUTPUT_DIR="${PROJECT_ROOT}/TriAttention_vLLM/outputs/aime24/perhead"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_DIR="${PROJECT_ROOT}/TriAttention_vLLM/logs/aime24/perhead"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

# Run benchmark
python3 "${SCRIPT_DIR}/run_math_vllm.py" \
  --model-path "${MODEL_PATH}" \
  --dataset-path "${DATASET_PATH}" \
  --output-path "${OUTPUT_FILE}" \
  --kv-budget 2048 \
  --window-size 128 \
  --divide-length 128 \
  --sparse-round-window 32 \
  --sparse-offset-max-length 65536 \
  --sparse-score-aggregation mean \
  --pruning-mode per_head \
  --sparse-normalize-scores \
  --include-prefill-in-budget \
  --rkv-style-compression \
  --rkv-style-slack-trigger \
  --sparse-stats-path "${STATS_PATH}" \
  --num-samples 8 \
  --temperature 0.6 \
  --top-p 0.95 \
  --seed 888 \
  --load-dtype bfloat16 \
  --max-tokens 32768 \
  "$@" 2>&1 | tee "${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

echo "Benchmark complete! Results saved to: ${OUTPUT_FILE}"
