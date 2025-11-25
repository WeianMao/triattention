#!/usr/bin/env bash
set -euo pipefail

# Single-GPU AIME24 baseline: sdpa + fp16 + fp32_topk + reset (outputs under R-KV/outputs, auto-eval).
# Process name is masked via run_math.py (PD-L1_binder).

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
DATA_PATH="${REPO_ROOT}/HuggingFace/data/aime24.jsonl"
MODEL_PATH="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B"
OUTPUT_DIR="${REPO_ROOT}/R-KV/outputs/rkv_aime24_single_sdpa_fp16_reset"
OUTPUT_PATH="${OUTPUT_DIR}/output.jsonl"
EVAL_DIR="${REPO_ROOT}/R-KV/HuggingFace/outputs/output_sdpa_fp16_reset_eval"

mkdir -p "${OUTPUT_DIR}" "${EVAL_DIR}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export HF_HOME="${HF_HOME:-/data/rbg/users/weian/.cache/huggingface}"
export PIP_CACHE_DIR="${PIP_CACHE_DIR:-/data/rbg/users/weian/.cache/pip}"

# Inference
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

# Evaluation (writes to ${EVAL_DIR})
conda run -n rkv --no-capture-output python "${REPO_ROOT}/HuggingFace/evaluation/eval_math.py" \
  --base_dir "${OUTPUT_DIR}" \
  --dataset aime24 \
  --exp_name output_sdpa_fp16_reset \
  --output_dir "${EVAL_DIR}"
