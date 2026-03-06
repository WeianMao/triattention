#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PIXI_BIN="${PIXI_BIN:-/home/wayne/.pixi/bin/pixi}"
TRACE_ROOT="${TRACE_ROOT:-/home/wayne/linxi/qwen3-8b_aime24_sample8_chat_merged.jsonl}"
MODEL_PATH="${MODEL_PATH:-/home/wayne/linxi/Qwen3-32B-INT4}"
OUTPUT_PATH="${OUTPUT_PATH:-/home/wayne/linxi/speckv/R-KV/outputs/qwen3-8b_aime24_sample8_chat/stats/qwen3_32b_int4_speckv_stats.pt}"

NUM_TRACES="${NUM_TRACES:-3}"
DTYPE="${DTYPE:-bfloat16}"
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION:-flash_attention_2}"
KV_BUDGET="${KV_BUDGET:-2048}"
FREE_MEM_THRESHOLD_MIB="${FREE_MEM_THRESHOLD_MIB:-5000}"
POLL_SECONDS="${POLL_SECONDS:-30}"

if [[ ! -x "${PIXI_BIN}" ]]; then
  echo "ERROR: pixi binary not found or not executable: ${PIXI_BIN}" >&2
  exit 1
fi

if [[ ! -f "${TRACE_ROOT}" ]]; then
  echo "ERROR: trace file not found: ${TRACE_ROOT}" >&2
  exit 1
fi

if [[ ! -d "${MODEL_PATH}" ]]; then
  echo "ERROR: model path not found: ${MODEL_PATH}" >&2
  exit 1
fi

pick_free_gpu() {
  nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
    | awk -F, -v threshold="${FREE_MEM_THRESHOLD_MIB}" '$2 + 0 < threshold { gsub(/ /, "", $1); print $1; exit }'
}

while true; do
  GPU="$(pick_free_gpu || true)"
  if [[ -n "${GPU}" ]]; then
    break
  fi
  echo "[wait] no free gpu (memory.used < ${FREE_MEM_THRESHOLD_MIB} MiB), sleep ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

mkdir -p "$(dirname "${OUTPUT_PATH}")"

echo "========================================"
echo "Generating SpeckV stats with GPTQ patch"
echo "========================================"
echo "Project root:        ${PROJECT_ROOT}"
echo "GPU:                 ${GPU}"
echo "Trace root:          ${TRACE_ROOT}"
echo "Model path:          ${MODEL_PATH}"
echo "Output path:         ${OUTPUT_PATH}"
echo "Num traces:          ${NUM_TRACES}"
echo "DType:               ${DTYPE}"
echo "Attn implementation: ${ATTN_IMPLEMENTATION}"
echo "KV budget:           ${KV_BUDGET}"
echo "========================================"

cd "${PROJECT_ROOT}"

CUDA_VISIBLE_DEVICES="${GPU}" \
TRACE_ROOT="${TRACE_ROOT}" \
MODEL_PATH="${MODEL_PATH}" \
OUTPUT_PATH="${OUTPUT_PATH}" \
NUM_TRACES="${NUM_TRACES}" \
DTYPE="${DTYPE}" \
ATTN_IMPLEMENTATION="${ATTN_IMPLEMENTATION}" \
KV_BUDGET="${KV_BUDGET}" \
"${PIXI_BIN}" run python - <<'PY'
import os
import runpy
import sys

import torch
import transformers

# Fix 1: avoid Marlin post-init in-place grad error in this calibration path.
torch.set_grad_enabled(False)

# Fix 2: force GPTQ backend selection on CUDA instead of CPU.
_orig_from_pretrained = transformers.AutoModelForCausalLM.from_pretrained


def _patched_from_pretrained(*args, **kwargs):
    kwargs.setdefault("device_map", "cuda")
    return _orig_from_pretrained(*args, **kwargs)


transformers.AutoModelForCausalLM.from_pretrained = _patched_from_pretrained

# Fix 3: default logits_to_keep=1 for Qwen3 forward when not provided.
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3ForCausalLM

    _orig_qwen3_forward = Qwen3ForCausalLM.forward

    def _patched_qwen3_forward(self, *args, **kwargs):
        if "logits_to_keep" not in kwargs:
            kwargs["logits_to_keep"] = 1
        return _orig_qwen3_forward(self, *args, **kwargs)

    Qwen3ForCausalLM.forward = _patched_qwen3_forward
except Exception:
    pass

sys.argv = [
    "rkv_sparse_round_calibrate.py",
    "--trace-root",
    os.environ["TRACE_ROOT"],
    "--model-path",
    os.environ["MODEL_PATH"],
    "--output-path",
    os.environ["OUTPUT_PATH"],
    "--num-traces",
    os.environ["NUM_TRACES"],
    "--dtype",
    os.environ["DTYPE"],
    "--attn-implementation",
    os.environ["ATTN_IMPLEMENTATION"],
    "--kv-budget",
    os.environ["KV_BUDGET"],
]

runpy.run_path("R-KV/weian_development/rkv_sparse_round_calibrate.py", run_name="__main__")
PY

echo "Done."
ls -lh "${OUTPUT_PATH}"
