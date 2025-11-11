#!/usr/bin/env bash
set -euo pipefail

# Launches the LazyEviction AIME evaluation across multiple GPUs using the
# sharded helper we just set up. Mirrors the manual commands we have been running.

SCRIPT_DIR="$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
LAZY_DIR="$( dirname "${SCRIPT_DIR}" )"
REPO_DIR="$( dirname "${LAZY_DIR}" )"
LOG_DIR="${REPO_DIR}/logs/lazy_eviction"
mkdir -p "${LOG_DIR}"

NUM_SHARDS=${NUM_SHARDS:-8}
GPUS=${GPUS:-"0 1 2 3 4 5 6 7"}

shard_id=0
for gpu in ${GPUS}; do
    if [ "${shard_id}" -ge "${NUM_SHARDS}" ]; then
        break
    fi
    log_path="${LOG_DIR}/shard${shard_id}.log"
    echo "Launching shard ${shard_id}/${NUM_SHARDS} on GPU ${gpu}, logging to ${log_path}"
    conda run -n lazy_evict bash -lc "nohup env CUDA_VISIBLE_DEVICES=${gpu} NUM_SHARDS=${NUM_SHARDS} SHARD_ID=${shard_id} bash ${LAZY_DIR}/eval_qwen_aime_sharded.sh > ${log_path} 2>&1 &"
    shard_id=$((shard_id + 1))
done
