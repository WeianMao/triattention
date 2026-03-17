#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
QWEN3_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

exec "${QWEN3_ROOT}/run_rkv.sh" --budget 1024 "$@"
