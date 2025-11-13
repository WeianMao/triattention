#!/usr/bin/env bash
set -euo pipefail

# Launch SparseRound (prefill-keep, KV×5) AIME evaluation across multiple GPUs.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
WEIAN_DIR="$( dirname "${SCRIPT_DIR}" )"
LAZY_DIR="$( dirname "${WEIAN_DIR}" )"
REPO_DIR="$( dirname "${LAZY_DIR}" )"
LOG_DIR="${REPO_DIR}/logs/lazy_eviction_sparse_round_prefill_kv5x"
mkdir -p "${LOG_DIR}"

if [ -z "${NUM_SHARDS+x}" ]; then
    NUM_SHARDS=8
    NUM_SHARDS_WAS_DEFAULT=1
else
    NUM_SHARDS_WAS_DEFAULT=0
fi

if [ -z "${GPUS+x}" ]; then
    GPUS="0 1 2 3 4 5 6 7"
    GPUS_WAS_DEFAULT=1
else
    GPUS_WAS_DEFAULT=0
fi

OUTPUT_ROOT_ENV="${OUTPUT_ROOT:-}"
EVAL_SCRIPT="${LAZY_DIR}/kv_variants/eval_qwen_aime_sparse_prefill_keep_kv5x_sharded.sh"

if [ "${GPUS_WAS_DEFAULT}" -eq 1 ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        GPU_MEMORY_THRESHOLD=${GPU_MEMORY_THRESHOLD:-200}
        AVAILABLE_GPUS=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | python -c "import os, sys; threshold=int(os.environ.get('GPU_MEMORY_THRESHOLD','200')); choices=[]; [choices.append(str(int(parts[0].strip()))) for line in sys.stdin if (parts:=line.split(',')) and len(parts)>=2 and parts[0].strip().isdigit() and parts[1].strip().isdigit() and int(parts[1])<=threshold]; sys.stdout.write(' '.join(choices))")
        if [ -n "${AVAILABLE_GPUS}" ]; then
            GPUS="${AVAILABLE_GPUS}"
            AVAILABLE_COUNT=$(echo "${AVAILABLE_GPUS}" | wc -w | awk '{print $1}')
            if [ "${AVAILABLE_COUNT}" -gt 0 ] && [ "${NUM_SHARDS_WAS_DEFAULT}" -eq 1 ]; then
                NUM_SHARDS=${AVAILABLE_COUNT}
            fi
        fi
    fi
fi

echo "Using eval script: ${EVAL_SCRIPT}"

shard_id=0
for gpu in ${GPUS}; do
    if [ "${shard_id}" -ge "${NUM_SHARDS}" ]; then
        break
    fi
    log_path="${LOG_DIR}/sparse_prefill_kv5x_shard${shard_id}.log"
    echo "Launching SparseRound-Prefix-KV5x shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu}, logging to ${log_path}"
    if [ -n "${OUTPUT_ROOT_ENV}" ]; then
        CONDA_CMD="nohup env OUTPUT_ROOT=${OUTPUT_ROOT_ENV} CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${shard_id} bash ${EVAL_SCRIPT} > ${log_path} 2>&1 &"
    else
        CONDA_CMD="nohup env CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${shard_id} bash ${EVAL_SCRIPT} > ${log_path} 2>&1 &"
    fi
    conda run -n lazy_evict bash -lc "${CONDA_CMD}"
    shard_id=$((shard_id + 1))
done

if [ "${shard_id}" -lt "${NUM_SHARDS}" ]; then
    echo "Warning: only launched ${shard_id} of ${NUM_SHARDS} requested shards (not enough GPUs listed)." >&2
fi
