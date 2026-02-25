#!/usr/bin/env bash
set -euo pipefail

# Default/current entrypoint (compat wrapper).
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${SCRIPT_DIR}/run_v2_hf_alignment_quick.sh" "$@"
