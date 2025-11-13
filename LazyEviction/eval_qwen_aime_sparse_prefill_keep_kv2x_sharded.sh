#!/usr/bin/env bash
set -euo pipefail

BENCHMARK="${BENCHMARK:-aime}"
MODEL_PATH="${MODEL_PATH:-/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B}"
MODEL_SIZE="${MODEL_SIZE:-7b}"
MODEL_TYPE="${MODEL_TYPE:-qwen}"
DATA_TYPE="${DATA_TYPE:-test}"

MAX_NUM_EXAMPLES=${MAX_NUM_EXAMPLES:-100000000000000}
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-16384}
EVAL_BATCH_SIZE=${EVAL_BATCH_SIZE:-1}
TEMPERATURE=${TEMPERATURE:-0.0}
SEED=${SEED:-42}

method="sparse_round_prefill"
max_kv_capacity=${MAX_KV_CAPACITY:-2984}
attn_implementation="${ATTN_IMPLEMENTATION:-sdpa}"
decoding_recent_size=${DECODING_RECENT_SIZE:-363}
SPARSE_OFFSET_MAX_LENGTH=${SPARSE_OFFSET_MAX_LENGTH:-65536}
SPARSE_SCORE_AGGREGATION="${SPARSE_SCORE_AGGREGATION:-mean}"
SPARSE_HEAD_LIMIT=${SPARSE_HEAD_LIMIT:--1}
SPARSE_SEED=${SPARSE_SEED:-0}
SPARSE_ROUND_WINDOW=${SPARSE_ROUND_WINDOW:-${decoding_recent_size}}

NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SHARD_ID:-0}

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_DIR="$( dirname "${SCRIPT_DIR}" )"
PYTHON_RUNNER="${REPO_DIR}/weian_development/lazy_eviction_sparse_prefill_keep_runner.py"
DEFAULT_STATS="${REPO_DIR}/weian_development/hf_offline_runner_sparse/stats/distill_qwen7b_qid9001_trace00_stats.pt"
SPARSE_STATS_PATH="${SPARSE_STATS_PATH:-${DEFAULT_STATS}}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${REPO_DIR}/outputs/DeepSeek-R1-Distill-Qwen-7B/${BENCHMARK}_prefill_keep_kv2x}"

python "${PYTHON_RUNNER}" --output-dir "${OUTPUT_ROOT}" \
    --model-path "${MODEL_PATH}" --tokenizer-path "${MODEL_PATH}" \
    --model-size "${MODEL_SIZE}" --model-type "${MODEL_TYPE}" --data-type "${DATA_TYPE}" \
    --max_num_examples "${MAX_NUM_EXAMPLES}" --max_new_tokens "${MAX_NEW_TOKENS}" \
    --eval_batch_size "${EVAL_BATCH_SIZE}" --temperature "${TEMPERATURE}" --seed "${SEED}" \
    --benchmark "${BENCHMARK}" --method "${method}" --use_cache True \
    --max_kv_capacity "${max_kv_capacity}" --decoding_recent_size "${decoding_recent_size}" \
    --attn_implementation "${attn_implementation}" --alpha 0.0001 \
    --num_shards "${NUM_SHARDS}" --shard_id "${SHARD_ID}" \
    --sparse_stats_path "${SPARSE_STATS_PATH}" \
    --sparse_offset_max_length "${SPARSE_OFFSET_MAX_LENGTH}" \
    --sparse_score_aggregation "${SPARSE_SCORE_AGGREGATION}" \
    --sparse_head_limit "${SPARSE_HEAD_LIMIT}" \
    --sparse_seed "${SPARSE_SEED}" \
    --sparse_round_window "${SPARSE_ROUND_WINDOW}"
