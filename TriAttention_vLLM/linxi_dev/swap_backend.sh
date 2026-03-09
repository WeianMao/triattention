#!/usr/bin/env bash
set -euo pipefail

# Hot-swap between TriAttention and baseline vLLM backends.
# Usage:
#   ./swap_backend.sh baseline    # switch to vanilla vLLM
#   ./swap_backend.sh triattention # switch to TriAttention
#   ./swap_backend.sh status      # show which backend is running

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Must match current deployment
VLLM_PORT="${VLLM_PORT:-8002}"
MODEL="${MODEL:-/home/wayne/linxi/Qwen3-32B-INT4}"
STATS_PATH="${STATS_PATH:-/home/wayne/linxi/speckv/R-KV/outputs/qwen3-8b_aime24_sample8_chat/stats/qwen3_32b_int4_speckv_stats.pt}"

_find_vllm_pid() {
  pgrep -f "vllm serve.*--port ${VLLM_PORT}" 2>/dev/null | head -1 || true
}

_current_mode() {
  local pid
  pid=$(_find_vllm_pid)
  if [[ -z "${pid}" ]]; then
    echo "stopped"
    return
  fi
  if tr '\0' '\n' < "/proc/${pid}/environ" 2>/dev/null | grep -q "ENABLE_TRIATTENTION=true"; then
    echo "triattention"
  else
    echo "baseline"
  fi
}

_wait_healthy() {
  local url="http://127.0.0.1:${VLLM_PORT}/health"
  echo -n "Waiting for vLLM health"
  for i in $(seq 1 60); do
    if curl -sf "${url}" >/dev/null 2>&1; then
      echo " ready (${i}s)"
      return 0
    fi
    echo -n "."
    sleep 1
  done
  echo " TIMEOUT"
  return 1
}

_kill_vllm() {
  local pid
  pid=$(_find_vllm_pid)
  if [[ -z "${pid}" ]]; then
    echo "No vLLM process on port ${VLLM_PORT}"
    return
  fi
  echo "Killing vLLM (PID ${pid})..."
  kill "${pid}" 2>/dev/null || true
  # wait for process to exit
  for _ in $(seq 1 15); do
    if ! kill -0 "${pid}" 2>/dev/null; then
      echo "Process stopped."
      return
    fi
    sleep 1
  done
  echo "Force killing..."
  kill -9 "${pid}" 2>/dev/null || true
  sleep 1
}

_start() {
  local mode="$1"
  local flag
  if [[ "${mode}" == "triattention" ]]; then
    flag=""
    export STATS_PATH
  else
    flag="--no-triattention"
  fi

  echo "Starting vLLM in ${mode} mode..."
  export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
  export MODEL HOST="${HOST:-127.0.0.1}" PORT="${VLLM_PORT}"
  export MAX_MODEL_LEN="${MAX_MODEL_LEN:-16384}"
  export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.96}"
  export TRIATTENTION_LOG_DECISIONS="${TRIATTENTION_LOG_DECISIONS:-1}"

  nohup "${SCRIPT_DIR}/run_vllm_serve.sh" ${flag} \
    --max-num-seqs 32 \
    > "/tmp/vllm_${mode}.log" 2>&1 &

  _wait_healthy
  echo "Backend: ${mode} | Port: ${VLLM_PORT} | PID: $(_find_vllm_pid)"
}

cmd="${1:-status}"
case "${cmd}" in
  status)
    mode=$(_current_mode)
    pid=$(_find_vllm_pid)
    echo "Backend: ${mode} | Port: ${VLLM_PORT} | PID: ${pid:-none}"
    ;;
  baseline|vanilla)
    current=$(_current_mode)
    if [[ "${current}" == "baseline" ]]; then
      echo "Already running baseline."
      exit 0
    fi
    _kill_vllm
    _start baseline
    ;;
  triattention|tri)
    current=$(_current_mode)
    if [[ "${current}" == "triattention" ]]; then
      echo "Already running triattention."
      exit 0
    fi
    _kill_vllm
    _start triattention
    ;;
  *)
    echo "Usage: $0 {baseline|triattention|status}"
    exit 1
    ;;
esac
