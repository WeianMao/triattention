#!/usr/bin/env bash
set -euo pipefail

# Reproduce the "32B long benchmark speaks normal language" run.
#
# Defaults match the known-good test setup:
# - model: JunHowie/Qwen3-32B-GPTQ-Int4
# - stats: qwen3_32b_int4_speckv_stats.pt
# - dataset: /tmp/tri_diag/openclaw_like_dataset.jsonl
# - kv_budget: 12000

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
RUNNER="${ROOT_DIR}/TriAttention_vLLM/evaluation/runner/vllm_triattention_runtime_runner.py"

CUDA_DEVICE="${CUDA_DEVICE:-0}"
MODEL_PATH="${MODEL_PATH:-/data/rbg/users/weian/env/huggingface/hub/models--JunHowie--Qwen3-32B-GPTQ-Int4/snapshots/275d13ed8617787bde259624a8ab2f5527266465}"
STATS_PATH="${STATS_PATH:-${ROOT_DIR}/demo/openclaw-demo/stats/qwen3_32b_int4_speckv_stats.pt}"
DATASET_PATH="${DATASET_PATH:-/tmp/tri_diag/openclaw_like_dataset.jsonl}"
OUTPUT_DIR="${OUTPUT_DIR:-/tmp/tri_diag/$(date +%Y%m%d_%H%M%S)_32b_openclaw_like_tri12k_out}"

KV_BUDGET="${KV_BUDGET:-12000}"
MAX_LENGTH="${MAX_LENGTH:-12600}"
MAX_EXAMPLES="${MAX_EXAMPLES:-1}"

if [[ ! -f "${RUNNER}" ]]; then
  echo "Runner not found: ${RUNNER}" >&2
  exit 1
fi
if [[ ! -f "${STATS_PATH}" ]]; then
  echo "Stats not found: ${STATS_PATH}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_PATH}" ]]; then
  echo "Dataset not found: ${DATASET_PATH}" >&2
  exit 1
fi

mkdir -p "${OUTPUT_DIR}"

echo "[run] output_dir=${OUTPUT_DIR}"
echo "[run] model=${MODEL_PATH}"
echo "[run] stats=${STATS_PATH}"
echo "[run] dataset=${DATASET_PATH}"
echo "[run] kv_budget=${KV_BUDGET} max_length=${MAX_LENGTH}"

CUDA_VISIBLE_DEVICES="${CUDA_DEVICE}" \
conda run -n trivllm python "${RUNNER}" \
  --seed 42 \
  --dataset-path "${DATASET_PATH}" \
  --output-dir "${OUTPUT_DIR}" \
  --model-path "${MODEL_PATH}" \
  --shard-id 0 \
  --num-shards 1 \
  --num-samples 1 \
  --max-examples "${MAX_EXAMPLES}" \
  --max-length "${MAX_LENGTH}" \
  --load-dtype float16 \
  --kv-budget "${KV_BUDGET}" \
  --divide-length 128 \
  --protect-prefill false \
  --window-size 128 \
  --sparse-stats-path "${STATS_PATH}" \
  --pruning-mode per_head \
  --per-head-selection-semantics hf_aligned_global_per_head \
  --layer-perhead-aggregation max \
  --per-layer-aggregation max \
  --allow-per-layer-mode true \
  --enable-experimental-kv-compaction true \
  --enable-experimental-block-reclaim true \
  --require-triton-scoring true \
  --require-physical-reclaim true \
  --disable-compression false \
  --tensor-parallel-size 1 \
  --gpu-memory-utilization 0.92 \
  --enforce-eager true

RESULT_JSONL="${OUTPUT_DIR}/shard00/run000.jsonl"
if [[ ! -f "${RESULT_JSONL}" ]]; then
  echo "Missing result file: ${RESULT_JSONL}" >&2
  exit 1
fi

echo "[summary] ${RESULT_JSONL}"
RESULT_JSONL="${RESULT_JSONL}" conda run -n trivllm python - <<'PY'
import itertools
import json
import os
import re

path = os.environ["RESULT_JSONL"]
obj = json.loads(open(path, "r", encoding="utf-8").readline())
text = obj.get("output") or obj.get("answer") or ""

ws = [x for x in re.split(r"\s+", text) if x]
max_ws = 1
cur = 1
prev = None
for tok in ws:
    if tok == prev:
        cur += 1
    else:
        cur = 1
        prev = tok
    if cur > max_ws:
        max_ws = cur

max_char = 1
max_char_c = ""
for c, g in itertools.groupby(text):
    n = sum(1 for _ in g)
    if n > max_char:
        max_char = n
        max_char_c = c

print(
    {
        "prefill_tokens": obj.get("prefill_tokens"),
        "output_tokens": obj.get("output_tokens"),
        "total_tokens": obj.get("total_tokens"),
        "kv_budget": obj.get("kv_budget"),
        "text_chars": len(text),
        "max_same_ws_run": max_ws,
        "max_same_char_run": max_char,
        "max_same_char": max_char_c,
    }
)
print("HEAD:\n" + text[:1000])
print("TAIL:\n" + text[-1000:])
PY
