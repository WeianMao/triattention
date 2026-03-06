#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PID_DIR="${SCRIPT_DIR}/run"

stop_by_pid_file() {
  local name="$1"
  local pid_file="$2"

  if [[ ! -f "${pid_file}" ]]; then
    echo "[stop] ${name}: pid file not found, skip"
    return 0
  fi

  local pid
  pid="$(cat "${pid_file}")"
  if [[ -n "${pid}" ]] && kill -0 "${pid}" 2>/dev/null; then
    echo "[stop] ${name}: sending TERM to pid ${pid}"
    kill "${pid}" || true
    sleep 1
    if kill -0 "${pid}" 2>/dev/null; then
      echo "[stop] ${name}: still alive, sending KILL to pid ${pid}"
      kill -9 "${pid}" || true
    fi
  else
    echo "[stop] ${name}: process not running"
  fi

  rm -f "${pid_file}"
}

stop_by_pid_file "demo" "${PID_DIR}/demo.pid"
stop_by_pid_file "baseline" "${PID_DIR}/baseline.pid"

echo "[stop] done"
