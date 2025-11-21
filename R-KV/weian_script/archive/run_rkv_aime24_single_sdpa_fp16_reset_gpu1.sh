#!/usr/bin/env bash
set -euo pipefail

# Second SDPA fp16+reset single-GPU run on a different device (avoid GPU 0).

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_PATH="${REPO_ROOT}/HuggingFace/data/aime24.jsonl"
MODEL_PATH="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_PATH="${REPO_ROOT}/HuggingFace/outputs/output_sdpa_fp16_reset_gpu1.jsonl"

mkdir -p "$(dirname "${OUTPUT_PATH}")"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

conda run -n rkv --no-capture-output python "${REPO_ROOT}/HuggingFace/run_math.py" \
  --dataset_path "${DATA_PATH}" \
  --save_path "${OUTPUT_PATH}" \
  --model_path "${MODEL_PATH}" \
  --max_length 16384 \
  --eval_batch_size 1 \
  --method rkv \
  --kv_budget 2048 \
  --attn_implementation sdpa \
  --load_dtype float16 \
  --fp32_topk True \
  --reset_cache_each_batch True
