#!/usr/bin/env bash
set -euo pipefail

# Quick V2 alignment run:
# 1) run TriAttention V2 on a tiny AIME24 slice;
# 2) optionally compare against HF results if provided.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG_PATH="${PROJECT_ROOT}/TriAttention_vLLM/evaluation/dispatch/configs/triattention_v2_aime24_quick.yaml"
DISPATCH_SCRIPT="${PROJECT_ROOT}/TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py"
COMPARE_SCRIPT="${PROJECT_ROOT}/TriAttention_vLLM/benchmarks/reasoning/compare_results.py"

HF_RESULTS_PATH=""
DRY_RUN_ARGS=()
for arg in "$@"; do
  if [[ "${arg}" == "--dry-run" ]]; then
    DRY_RUN_ARGS+=("--dry-run")
  else
    HF_RESULTS_PATH="${arg}"
  fi
done
V2_MERGED_PATH="${PROJECT_ROOT}/TriAttention_vLLM/evaluation/outputs/triattention_v2_aime24_quick/merged/merged.jsonl"
REPORT_DIR="${PROJECT_ROOT}/TriAttention_vLLM/evaluation/outputs/triattention_v2_aime24_quick"
REPORT_PATH="${REPORT_DIR}/hf_alignment_report.txt"

mkdir -p "${REPORT_DIR}"

COMPARE_PY=(python)
if command -v conda >/dev/null 2>&1; then
  # `rkv` env contains strict-math-eval dependencies (latex2sympy2, etc.).
  COMPARE_PY=(conda run -n rkv python)
fi

echo "=================================================="
echo "TriAttention V2 quick alignment run"
echo "Config: ${CONFIG_PATH}"
echo "=================================================="

python "${DISPATCH_SCRIPT}" \
  --config "${CONFIG_PATH}" \
  --no-eval \
  --no-skip-existing \
  "${DRY_RUN_ARGS[@]}"

echo "V2 merged output: ${V2_MERGED_PATH}"

if [[ ${#DRY_RUN_ARGS[@]} -gt 0 ]]; then
  echo "Dry-run mode; skipped HF comparison."
elif [[ -n "${HF_RESULTS_PATH}" ]]; then
  if [[ ! -f "${HF_RESULTS_PATH}" ]]; then
    echo "HF results file not found: ${HF_RESULTS_PATH}"
    exit 1
  fi
  echo "Comparing against HF results: ${HF_RESULTS_PATH}"
  "${COMPARE_PY[@]}" "${COMPARE_SCRIPT}" \
    --hf-results "${HF_RESULTS_PATH}" \
    --vllm-results "${V2_MERGED_PATH}" \
    --dataset-name aime24 \
    --output-report "${REPORT_PATH}" \
    --detailed
  echo "Comparison report: ${REPORT_PATH}"
else
  echo "HF results path not provided; skipped comparison."
  echo "Usage with comparison:"
  echo "  ${0} /path/to/hf_merged.jsonl"
fi
