#!/usr/bin/env bash
set -euo pipefail

# Quick-start runner for vLLM serve with optional TriAttention backend.
# Usage:
#   conda activate trivllm
#   ./linxi_dev/run_vllm_serve.sh
#   ENABLE_TRIATTENTION=false ./linxi_dev/run_vllm_serve.sh
#   ./linxi_dev/run_vllm_serve.sh --no-triattention --port 9000
#   MODEL=/path/to/model STATS_PATH=/path/to/stats.pt ./linxi_dev/run_vllm_serve.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# ----------------------------
# Editable env vars (defaults)
# ----------------------------
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
: "${MODEL:=/home/linxi/models/Qwen3-0.6B}"
: "${HOST:=0.0.0.0}"
: "${PORT:=8000}"
: "${MAX_MODEL_LEN:=32768}"
: "${GPU_MEMORY_UTILIZATION:=0.2}"
: "${DTYPE:=bfloat16}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${TRUST_REMOTE_CODE:=true}"
: "${VLLM_ENABLE_V1_MULTIPROCESSING:=0}"
: "${ENFORCE_EAGER:=true}"

# TriAttention-related variables (only used when ENABLE_TRIATTENTION=true)
: "${ENABLE_TRIATTENTION:=true}"
: "${STATS_PATH:=/home/linxi/projects/speckv/R-KV/speckv_experiments/stats/aime24/Qwen3-8B/stats_budget_512.pt}"
: "${KV_BUDGET:=2048}"
: "${DIVIDE_LENGTH:=128}"
: "${WINDOW_SIZE:=128}"
: "${PRUNING_MODE:=per_head}"  # per_head | per_layer | per_layer_head
: "${TRIATTENTION_QUIET:=0}"
: "${TRIATTENTION_LOG_TRIGGER:=1}"
: "${TRIATTENTION_LOG_DECISIONS:=0}"
# Interface mode:
# - runtime (default): V2 integration via plugin monkeypatch + TRIATTN_RUNTIME_* envs
# - legacy_custom: retired V1 CUSTOM backend path (kept only for compatibility)
: "${TRIATTENTION_INTERFACE:=runtime}"
: "${TRIATTENTION_BACKEND:=CUSTOM}"  # legacy mode only

# Optional process title override for long-running jobs.
: "${PROCESS_NAME:=PD-L1_binder}"

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

# Parse script-only toggles and forward everything else to vllm serve.
PASSTHROUGH_ARGS=()
USER_ATTENTION_BACKEND=""
while (($# > 0)); do
  case "$1" in
    --triattention)
      ENABLE_TRIATTENTION=true
      shift
      ;;
    --no-triattention)
      ENABLE_TRIATTENTION=false
      shift
      ;;
    --attention-backend)
      if (($# < 2)); then
        echo "Missing value for --attention-backend" >&2
        exit 1
      fi
      USER_ATTENTION_BACKEND="$2"
      shift 2
      ;;
    --attention-backend=*)
      USER_ATTENTION_BACKEND="${1#*=}"
      shift
      ;;
    *)
      PASSTHROUGH_ARGS+=("$1")
      shift
      ;;
  esac
done

# Safety: strip any attention-backend override from passthrough args so it
# cannot override the backend selected by this script.
CLEAN_PASSTHROUGH_ARGS=()
i=0
while ((i < ${#PASSTHROUGH_ARGS[@]})); do
  arg="${PASSTHROUGH_ARGS[$i]}"
  if [[ "${arg}" == "--attention-backend" ]]; then
    echo "Ignoring passthrough --attention-backend override: ${PASSTHROUGH_ARGS[$((i + 1))]:-<missing>}" >&2
    i=$((i + 2))
    continue
  fi
  if [[ "${arg}" == --attention-backend=* ]]; then
    echo "Ignoring passthrough ${arg}" >&2
    i=$((i + 1))
    continue
  fi
  CLEAN_PASSTHROUGH_ARGS+=("${arg}")
  i=$((i + 1))
done
PASSTHROUGH_ARGS=("${CLEAN_PASSTHROUGH_ARGS[@]}")

SERVE_ARGS=(
  serve "${MODEL}"
  --host "${HOST}"
  --port "${PORT}"
  --dtype "${DTYPE}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}"
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --enable-auto-tool-choice
  --tool-call-parser hermes
)

if is_truthy "${TRUST_REMOTE_CODE}"; then
  SERVE_ARGS+=(--trust-remote-code)
fi

USER_SET_ENFORCE_EAGER=0
for arg in "${PASSTHROUGH_ARGS[@]}"; do
  if [[ "${arg}" == "--enforce-eager" || "${arg}" == "--no-enforce-eager" ]]; then
    USER_SET_ENFORCE_EAGER=1
    break
  fi
done

if is_truthy "${ENABLE_TRIATTENTION}"; then
  if [[ ! -f "${STATS_PATH}" ]]; then
    echo "TRIATTENTION is enabled, but stats file not found: ${STATS_PATH}" >&2
    echo "Set STATS_PATH to a valid .pt file, or disable with --no-triattention." >&2
    exit 1
  fi

  if ! pixi run python -c "import importlib.metadata as m, sys; sys.exit(0 if any(ep.name=='triattention' for ep in m.entry_points(group='vllm.general_plugins')) else 1)"; then
    echo "TriAttention plugin is not installed in current pixi environment." >&2
    echo "Run: pixi run python -m pip install -e ${PROJECT_ROOT}" >&2
    exit 1
  fi

  # Keep legacy env vars for backward compatibility.
  export TRIATTENTION_STATS_PATH="${STATS_PATH}"
  export TRIATTENTION_KV_BUDGET="${KV_BUDGET}"
  export TRIATTENTION_DIVIDE_LENGTH="${DIVIDE_LENGTH}"
  export TRIATTENTION_WINDOW_SIZE="${WINDOW_SIZE}"
  export TRIATTENTION_PRUNING_MODE="${PRUNING_MODE}"
  export TRIATTENTION_QUIET="${TRIATTENTION_QUIET}"
  export TRIATTENTION_LOG_TRIGGER="${TRIATTENTION_LOG_TRIGGER}"
  export TRIATTENTION_LOG_DECISIONS="${TRIATTENTION_LOG_DECISIONS}"
  export TRIATTENTION_INTERFACE="${TRIATTENTION_INTERFACE}"

  # Runtime (V2) env vars.
  export TRIATTN_RUNTIME_SPARSE_STATS_PATH="${STATS_PATH}"
  export TRIATTN_RUNTIME_KV_BUDGET="${KV_BUDGET}"
  export TRIATTN_RUNTIME_DIVIDE_LENGTH="${DIVIDE_LENGTH}"
  export TRIATTN_RUNTIME_WINDOW_SIZE="${WINDOW_SIZE}"
  case "${PRUNING_MODE}" in
    per_layer_head) export TRIATTN_RUNTIME_PRUNING_MODE="per_layer_per_head" ;;
    *) export TRIATTN_RUNTIME_PRUNING_MODE="${PRUNING_MODE}" ;;
  esac
  export TRIATTN_RUNTIME_LOG_DECISIONS="${TRIATTENTION_LOG_DECISIONS}"
  export TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION="${TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_KV_COMPACTION:-true}"
  export TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM="${TRIATTN_RUNTIME_ENABLE_EXPERIMENTAL_BLOCK_RECLAIM:-true}"
  export TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING="${TRIATTN_RUNTIME_REQUIRE_TRITON_SCORING:-true}"
  export TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM="${TRIATTN_RUNTIME_REQUIRE_PHYSICAL_RECLAIM:-true}"
  export TRIATTN_RUNTIME_PATCH_WORKER="${TRIATTN_RUNTIME_PATCH_WORKER:-true}"
  export TRIATTN_RUNTIME_PATCH_SCHEDULER="${TRIATTN_RUNTIME_PATCH_SCHEDULER:-true}"

  # Ensure plugin loading is not accidentally filtered out.
  if [[ -z "${VLLM_PLUGINS:-}" ]]; then
    export VLLM_PLUGINS="triattention"
  elif [[ ",${VLLM_PLUGINS// /}," != *",triattention,"* ]]; then
    export VLLM_PLUGINS="${VLLM_PLUGINS},triattention"
  fi

  interface_mode="${TRIATTENTION_INTERFACE,,}"
  if [[ "${interface_mode}" == "legacy_custom" || "${interface_mode}" == "legacy" || "${interface_mode}" == "v1" || "${interface_mode}" == "custom" ]]; then
    FINAL_ATTENTION_BACKEND="${TRIATTENTION_BACKEND}"
    if [[ -n "${USER_ATTENTION_BACKEND}" ]]; then
      FINAL_ATTENTION_BACKEND="${USER_ATTENTION_BACKEND}"
    fi
    SERVE_ARGS+=(--attention-backend "${FINAL_ATTENTION_BACKEND}")
    echo "[run_vllm_serve] TRIATTENTION_INTERFACE=${TRIATTENTION_INTERFACE} (legacy) backend=${FINAL_ATTENTION_BACKEND}" >&2
  else
    if [[ -n "${USER_ATTENTION_BACKEND}" ]]; then
      echo "Ignoring user --attention-backend (${USER_ATTENTION_BACKEND}) in runtime interface mode." >&2
    fi
    echo "[run_vllm_serve] TRIATTENTION_INTERFACE=${TRIATTENTION_INTERFACE} (runtime/v2) via plugin monkeypatch" >&2
  fi
  if is_truthy "${ENFORCE_EAGER}" && (( USER_SET_ENFORCE_EAGER == 0 )); then
    SERVE_ARGS+=(--enforce-eager)
  fi
else
  unset TRIATTENTION_STATS_PATH TRIATTENTION_KV_BUDGET TRIATTENTION_DIVIDE_LENGTH
  unset TRIATTENTION_WINDOW_SIZE TRIATTENTION_PRUNING_MODE TRIATTENTION_QUIET
  unset TRIATTENTION_LOG_TRIGGER TRIATTENTION_LOG_DECISIONS
  if [[ -n "${USER_ATTENTION_BACKEND}" ]]; then
    SERVE_ARGS+=(--attention-backend "${USER_ATTENTION_BACKEND}")
  fi
fi

export VLLM_ENABLE_V1_MULTIPROCESSING

echo "[run_vllm_serve] command: pixi run vllm ${SERVE_ARGS[*]} ${PASSTHROUGH_ARGS[*]}" >&2
exec -a "${PROCESS_NAME}" pixi run vllm "${SERVE_ARGS[@]}" "${PASSTHROUGH_ARGS[@]}"
