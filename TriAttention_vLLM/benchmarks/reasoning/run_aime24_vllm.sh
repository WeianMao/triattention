#!/usr/bin/env bash
set -euo pipefail

# Production script for running AIME24 benchmark with TriAttention + vLLM
# Uses DeepSeek-R1-Distill-Qwen-7B with tensor parallelism on 2 GPUs

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

# Environment setup
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export VLLM_PROCESS_NAME_PREFIX="${VLLM_PROCESS_NAME_PREFIX:-PD-L1_binder}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"
export CUDA_VISIBLE_DEVICES=1,2

# Activate trivllm conda environment
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate trivllm

# Model and dataset paths
MODEL_PATH="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B"
DATASET_PATH="${PROJECT_ROOT}/R-KV/HuggingFace/data/aime24.jsonl"
STATS_PATH="${PROJECT_ROOT}/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt"

# Output paths
OUTPUT_DIR="${PROJECT_ROOT}/TriAttention_vLLM/outputs/aime24_vllm_perhead"
OUTPUT_FILE="${OUTPUT_DIR}/results.jsonl"
LOG_DIR="${PROJECT_ROOT}/TriAttention_vLLM/logs/aime24_vllm_perhead"

# Create output directories
mkdir -p "${OUTPUT_DIR}"
mkdir -p "${LOG_DIR}"

echo "=================================================="
echo "AIME24 Benchmark: vLLM + TriAttention"
echo "=================================================="
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET_PATH}"
echo "Stats: ${STATS_PATH}"
echo "Output: ${OUTPUT_FILE}"
echo "GPUs: ${CUDA_VISIBLE_DEVICES}"
echo "Tensor Parallel Size: 2"
echo "=================================================="

# Run benchmark with tensor parallelism
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
  --tensor-parallel-size 2 \
  --gpu-memory-utilization 0.85 \
  --max-tokens 32768 \
  2>&1 | tee "${LOG_DIR}/run_$(date +%Y%m%d_%H%M%S).log"

echo ""
echo "=================================================="
echo "Benchmark complete!"
echo "Results: ${OUTPUT_FILE}"
echo "=================================================="
