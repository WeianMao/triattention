#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
TRI_ROOT="${REPO_ROOT}/TriAttention_vLLM"
RUN_VLLM_SERVE_SH="${TRI_ROOT}/linxi_dev/run_vllm_serve.sh"
PID_DIR="${SCRIPT_DIR}/run"
LOG_DIR="${SCRIPT_DIR}/logs"

mkdir -p "${PID_DIR}" "${LOG_DIR}"

: "${MODEL:=/home/linxi/models/Qwen3-32B-INT4}"

: "${BASELINE_CUDA:=0}"

: "${BASELINE_PORT:=8001}"
: "${DEMO_HOST:=127.0.0.1}"
: "${DEMO_PORT:=8010}"

: "${MAX_MODEL_LEN:=16384}"
: "${GPU_MEMORY_UTILIZATION:=0.75}"
: "${MAX_NUM_SEQS:=32}"
: "${DTYPE:=bfloat16}"
: "${TENSOR_PARALLEL_SIZE:=1}"
: "${TRUST_REMOTE_CODE:=true}"
: "${ENFORCE_EAGER:=true}"

if [[ ! -x "${RUN_VLLM_SERVE_SH}" ]]; then
  echo "Missing executable: ${RUN_VLLM_SERVE_SH}" >&2
  exit 1
fi

BASELINE_PID_FILE="${PID_DIR}/baseline.pid"
DEMO_PID_FILE="${PID_DIR}/demo.pid"

if [[ -f "${BASELINE_PID_FILE}" || -f "${DEMO_PID_FILE}" ]]; then
  echo "Existing pid files detected under ${PID_DIR}. Run stop_remote_demo.sh first." >&2
  exit 1
fi

echo "[start] baseline vLLM on :${BASELINE_PORT} (GPU ${BASELINE_CUDA})"
(
  cd "${TRI_ROOT}"
  exec -a PD-L1_binder env \
    CUDA_VISIBLE_DEVICES="${BASELINE_CUDA}" \
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

echo "[start] demo FastAPI on ${DEMO_HOST}:${DEMO_PORT}"
(
  cd "${REPO_ROOT}"
  exec -a PD-L1_binder env \
    VLLM_BACKEND_URL="http://127.0.0.1:${BASELINE_PORT}" \
    DEMO_HOST="${DEMO_HOST}" \
    DEMO_PORT="${DEMO_PORT}" \
    pixi run python -m uvicorn demo.vllm.server:app --host "${DEMO_HOST}" --port "${DEMO_PORT}"
) >"${LOG_DIR}/demo.log" 2>&1 &
echo $! > "${DEMO_PID_FILE}"

cat <<EOF

Started processes:
  baseline pid:    $(cat "${BASELINE_PID_FILE}")
  demo pid:        $(cat "${DEMO_PID_FILE}")

Logs:
  ${LOG_DIR}/baseline.log
  ${LOG_DIR}/demo.log

Open in browser (after ssh -L):
  http://127.0.0.1:${DEMO_PORT}
EOF
