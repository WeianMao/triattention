#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${REPO_ROOT}/HuggingFace/data/aime24.jsonl"
MODEL_PATH="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_PATH="${REPO_ROOT}/R-KV/outputs/aime24_rkv.jsonl"

mkdir -p "${REPO_ROOT}/R-KV/outputs"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

conda run -n rkv CUDA_HOME=/usr/local/cuda-12.4 python "${REPO_ROOT}/HuggingFace/run_math.py" \
  --dataset_path "${DATA_PATH}" \
  --save_path "${OUTPUT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --max_length 16384 \
  --eval_batch_size 1 \
  --method rkv \
  --kv_budget 128
