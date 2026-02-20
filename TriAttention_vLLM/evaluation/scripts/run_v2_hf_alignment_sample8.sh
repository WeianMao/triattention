#!/usr/bin/env bash
set -euo pipefail

# HF-aligned V2 run:
# 1) use HF-style sharded task scheduling (num_shards tasks over draw_ids);
# 2) generate sample8 outputs with V2 runner;
# 3) optionally compare with HF merged results.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

CONFIG_PATH="${PROJECT_ROOT}/TriAttention_vLLM/evaluation/dispatch/configs/triattention_v2_aime24_hf_strict.yaml"
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

V2_MERGED_PATH="${PROJECT_ROOT}/TriAttention_vLLM/evaluation/outputs/triattention_v2_aime24_hf_strict/merged/merged.jsonl"
REPORT_DIR="${PROJECT_ROOT}/TriAttention_vLLM/evaluation/outputs/triattention_v2_aime24_hf_strict"
REPORT_PATH="${REPORT_DIR}/hf_alignment_report_strict.txt"

mkdir -p "${REPORT_DIR}"

COMPARE_PY=(python)
if command -v conda >/dev/null 2>&1; then
  COMPARE_PY=(conda run -n rkv python)
fi

echo "=================================================="
echo "TriAttention V2 HF-aligned sample8 run"
echo "Config: ${CONFIG_PATH}"
echo "=================================================="
echo "Note: rerun should use a clean output directory to avoid resume skip."

python "${DISPATCH_SCRIPT}" \
  --config "${CONFIG_PATH}" \
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
