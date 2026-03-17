#!/usr/bin/env bash
set -euo pipefail

# Ensure pixi is on PATH for non-interactive SSH sessions
export PATH="$HOME/.pixi/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRI_ROOT="${REPO_ROOT}/TriAttention_vLLM"
RUN_VLLM_SERVE_SH="${TRI_ROOT}/linxi_dev/run_vllm_serve.sh"
PID_DIR="${SCRIPT_DIR}/run"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${PID_DIR}" "${LOG_DIR}"

: "${MODEL:=/home/wayne/linxi/Qwen3-32B-INT4}"
: "${STATS_PATH:=${REPO_ROOT}/demo/openclaw-demo/stats/qwen3_32b_int4_speckv_stats.pt}"

: "${BASELINE_CUDA:=3}"
: "${TRIATTENTION_CUDA:=4}"

: "${BASELINE_PORT:=8001}"
: "${TRIATTENTION_PORT:=8002}"
: "${DEMO_HOST:=127.0.0.1}"
: "${DEMO_PORT:=8010}"

: "${MAX_MODEL_LEN:=16384}"
: "${GPU_MEMORY_UTILIZATION:=0.75}"
: "${MAX_NUM_SEQS:=32}"
: "${DTYPE:=bfloat16}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${TRUST_REMOTE_CODE:=true}"
: "${ENFORCE_EAGER:=true}"
: "${KV_BUDGET:=14336}"
: "${VLLM_RELAXED_KV_CHECK:=1}"
: "${TRIATTN_RUNTIME_PROTECT_PREFILL:=false}"

if [[ ! -x "${RUN_VLLM_SERVE_SH}" ]]; then
  echo "Missing executable: ${RUN_VLLM_SERVE_SH}" >&2
  exit 1
fi

BASELINE_PID_FILE="${PID_DIR}/baseline.pid"
TRIATTENTION_PID_FILE="${PID_DIR}/triattention.pid"
DEMO_PID_FILE="${PID_DIR}/demo.pid"

if [[ -f "${BASELINE_PID_FILE}" || -f "${TRIATTENTION_PID_FILE}" || -f "${DEMO_PID_FILE}" ]]; then
  echo "Existing pid files detected under ${PID_DIR}. Run stop_remote_demo.sh first." >&2
  exit 1
fi

echo "[start] baseline vLLM on :${BASELINE_PORT} (GPU ${BASELINE_CUDA})"
(
  cd "${TRI_ROOT}"
  exec -a PD-L1_binder env \
    CUDA_VISIBLE_DEVICES="${BASELINE_CUDA}" \
    VLLM_RELAXED_KV_CHECK="${VLLM_RELAXED_KV_CHECK}" \
    MODEL="${MODEL}" \
    PORT="${BASELINE_PORT}" \
    HOST="127.0.0.1" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    DTYPE="${DTYPE}" \
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" \
    ENFORCE_EAGER="${ENFORCE_EAGER}" \
    "${RUN_VLLM_SERVE_SH}" --no-triattention --max-num-seqs "${MAX_NUM_SEQS}"
) >"${LOG_DIR}/baseline.log" 2>&1 &
echo $! > "${BASELINE_PID_FILE}"

sleep 1

echo "[start] triattention vLLM on :${TRIATTENTION_PORT} (GPU ${TRIATTENTION_CUDA})"
(
  cd "${TRI_ROOT}"
  exec -a PD-L1_binder env \
    CUDA_VISIBLE_DEVICES="${TRIATTENTION_CUDA}" \
    VLLM_RELAXED_KV_CHECK="${VLLM_RELAXED_KV_CHECK}" \
    MODEL="${MODEL}" \
    PORT="${TRIATTENTION_PORT}" \
    HOST="127.0.0.1" \
    MAX_MODEL_LEN="${MAX_MODEL_LEN}" \
    GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION}" \
    DTYPE="${DTYPE}" \
    TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE}" \
    TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE}" \
    ENFORCE_EAGER="${ENFORCE_EAGER}" \
    TRIATTN_RUNTIME_PROTECT_PREFILL="${TRIATTN_RUNTIME_PROTECT_PREFILL}" \
    STATS_PATH="${STATS_PATH}" \
    KV_BUDGET="${KV_BUDGET}" \
    "${RUN_VLLM_SERVE_SH}" --max-num-seqs "${MAX_NUM_SEQS}"
) >"${LOG_DIR}/triattention.log" 2>&1 &
echo $! > "${TRIATTENTION_PID_FILE}"

sleep 1

echo "[start] demo FastAPI on ${DEMO_HOST}:${DEMO_PORT}"
(
  cd "${REPO_ROOT}"
  exec -a PD-L1_binder env \
    VLLM_BACKEND_URL="http://127.0.0.1:${BASELINE_PORT}" \
    TRIATTENTION_VLLM_URL="http://127.0.0.1:${TRIATTENTION_PORT}" \
    DEMO_HOST="${DEMO_HOST}" \
    DEMO_PORT="${DEMO_PORT}" \
    pixi run python -m uvicorn demo.vllm.server:app --host "${DEMO_HOST}" --port "${DEMO_PORT}"
) >"${LOG_DIR}/demo.log" 2>&1 &
echo $! > "${DEMO_PID_FILE}"

cat <<EOF

Started processes:
  baseline pid:       $(cat "${BASELINE_PID_FILE}")   (GPU ${BASELINE_CUDA}, port ${BASELINE_PORT})
  triattention pid:   $(cat "${TRIATTENTION_PID_FILE}")   (GPU ${TRIATTENTION_CUDA}, port ${TRIATTENTION_PORT})
  demo pid:           $(cat "${DEMO_PID_FILE}")   (port ${DEMO_PORT})

Logs:
  ${LOG_DIR}/baseline.log
  ${LOG_DIR}/triattention.log
  ${LOG_DIR}/demo.log

Open in browser (after ssh -L):
  http://127.0.0.1:${DEMO_PORT}
EOF
