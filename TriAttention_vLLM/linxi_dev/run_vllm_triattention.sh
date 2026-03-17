#!/usr/bin/env bash
set -euo pipefail

# Quick-start runner for TriAttention vLLM integration.
# Usage:
#   conda activate trivllm
#   ./linxi_dev/run_vllm_triattention.sh
#   MODEL=/path/to/model STATS_PATH=/path/to/stats.pt ./linxi_dev/run_vllm_triattention.sh
#   KV_BUDGET=128 PROMPT_1="Hello" PROMPT_2="World" ./linxi_dev/run_vllm_triattention.sh
#   ./linxi_dev/run_vllm_triattention.sh --kv-budget 512 --prompt "CLI override"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# ----------------------------
# Editable env vars (defaults)
# ----------------------------

: "${MODEL:=/home/linxi/projects/speckv/R-KV/speckv_experiments/models/Qwen3-8B}"
: "${STATS_PATH:=/home/linxi/projects/speckv/R-KV/speckv_experiments/stats/aime24/Qwen3-8B/stats_budget_512.pt}"
: "${KV_BUDGET:=2048}"
: "${DIVIDE_LENGTH:=128}"
: "${PRUNING_MODE:=per_head}"  # per_head | per_layer | per_layer_head
: "${MAX_MODEL_LEN:=32768}"
: "${GPU_MEMORY_UTILIZATION:=0.8}"
: "${DTYPE:=bfloat16}"
: "${VLLM_ENABLE_V1_MULTIPROCESSING:=0}"

# Booleans: true/false (also accepts 1/0, yes/no, on/off)
: "${ENFORCE_EAGER:=false}"
: "${TRUST_REMOTE_CODE:=true}"

: "${TEMPERATURE:=0.6}"
: "${TOP_P:=0.95}"
: "${MAX_TOKENS:=16384}"
: "${PROMPT_1:=The quick brown fox}"
: "${PROMPT_2:=Once upon a time}"

is_truthy() {
  case "${1,,}" in
    1|true|yes|on) return 0 ;;
    0|false|no|off) return 1 ;;
    *)
      echo "Invalid boolean value: ${1}" >&2
      echo "Use one of: true/false, 1/0, yes/no, on/off" >&2
      exit 1
      ;;
  esac
}

ARGS=(
  --model "${MODEL}"
  --stats-path "${STATS_PATH}"
  --kv-budget "${KV_BUDGET}"
  --divide-length "${DIVIDE_LENGTH}"
  --pruning-mode "${PRUNING_MODE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --dtype "${DTYPE}"
  --temperature "${TEMPERATURE}"
  --top-p "${TOP_P}"
  --max-tokens "${MAX_TOKENS}"
  --prompt "${PROMPT_1}"
  --prompt "${PROMPT_2}"
)

if is_truthy "${ENFORCE_EAGER}"; then
  ARGS+=(--enforce-eager)
else
  ARGS+=(--no-enforce-eager)
fi

if is_truthy "${TRUST_REMOTE_CODE}"; then
  ARGS+=(--trust-remote-code)
else
  ARGS+=(--no-trust-remote-code)
fi

VLLM_ENABLE_V1_MULTIPROCESSING="${VLLM_ENABLE_V1_MULTIPROCESSING}" \
  pixi run python "${PROJECT_ROOT}/examples/quick_vllm_triattention.py" "${ARGS[@]}" "$@"
