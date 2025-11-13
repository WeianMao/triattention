BENCHMARK="aime" # "gsm8k", "math", "aime"
MODEL_PATH="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B"
MODEL_SIZE="7b"
MODEL_TYPE="qwen" # "llama3", "qwen"
DATA_TYPE="test"

# Generation Settings
MAX_NUM_EXAMPLES=100000000000000
MAX_NEW_TOKENS=16384
EVAL_BATCH_SIZE=1
TEMPERATURE=0.0
SEED=42

# KV Compression Settings
method=Window_LAZY
max_kv_capacity=2984
attn_implementation="sdpa"
decoding_recent_size=363

# Shard Controls
NUM_SHARDS=${NUM_SHARDS:-1}
SHARD_ID=${SHARD_ID:-0}

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
REPO_DIR="$( dirname "${SCRIPT_DIR}" )"
PYTHON_RUNNER="${REPO_DIR}/weian_development/lazy_eviction_sharded_runner.py"

OUPTUT_DIR="${REPO_DIR}/outputs/DeepSeek-R1-Distill-Qwen-7B/${BENCHMARK}_kv2x"

python "${PYTHON_RUNNER}" --output-dir ${OUPTUT_DIR} \
    --model-path ${MODEL_PATH} --tokenizer-path ${MODEL_PATH} \
    --model-size ${MODEL_SIZE} --model-type ${MODEL_TYPE} --data-type ${DATA_TYPE} \
    --max_num_examples ${MAX_NUM_EXAMPLES} --max_new_tokens ${MAX_NEW_TOKENS} \
    --eval_batch_size ${EVAL_BATCH_SIZE} --temperature ${TEMPERATURE} --seed ${SEED} \
    --benchmark ${BENCHMARK} --method ${method} --use_cache True \
    --max_kv_capacity ${max_kv_capacity} --decoding_recent_size ${decoding_recent_size} \
    --attn_implementation ${attn_implementation} --alpha 0.0001 \
    --num_shards ${NUM_SHARDS} --shard_id ${SHARD_ID}
