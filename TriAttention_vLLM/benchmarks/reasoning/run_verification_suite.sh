#!/usr/bin/env bash
set -euo pipefail

# Comprehensive verification suite for TriAttention vLLM integration
# Tests compression triggering, output format, and comparison readiness

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../../.." && pwd)"

export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"
export CUDA_VISIBLE_DEVICES=1

echo "=================================================="
echo "TriAttention vLLM Verification Suite"
echo "=================================================="
echo ""

# Activate environment
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate trivllm

# Test 1: Verify compression is triggered
echo "Test 1: Verifying compression triggers..."
echo "--------------------------------------------------"
python3 "${SCRIPT_DIR}/verify_compression.py"
echo ""

# Test 2: Check output format
echo "Test 2: Checking output format..."
echo "--------------------------------------------------"
VLLM_RESULTS="${PROJECT_ROOT}/TriAttention_vLLM/outputs/test_minimal/results.jsonl"

if [ -f "${VLLM_RESULTS}" ]; then
    python3 "${SCRIPT_DIR}/check_output_format.py" \
        --vllm-results "${VLLM_RESULTS}"
else
    echo "ERROR: Test results not found at ${VLLM_RESULTS}"
    echo "Run test_minimal.sh first"
    exit 1
fi

echo ""

# Test 3: Compare with HF results if available
echo "Test 3: Format compatibility test..."
echo "--------------------------------------------------"

# Find a sample HF result file
HF_SAMPLE=$(find "${PROJECT_ROOT}/R-KV/speckv_experiments/outputs" -name "*.jsonl" -type f 2>/dev/null | head -1)

if [ -n "${HF_SAMPLE}" ] && [ -f "${HF_SAMPLE}" ]; then
    echo "Found HF sample: ${HF_SAMPLE}"
    python3 "${SCRIPT_DIR}/check_output_format.py" \
        --vllm-results "${VLLM_RESULTS}" \
        --hf-results "${HF_SAMPLE}"
else
    echo "No HF results found for comparison (optional)"
fi

echo ""
echo "=================================================="
echo "Verification Suite Complete"
echo "=================================================="
echo ""
echo "Summary:"
echo "  1. Compression trigger: See above output"
echo "  2. Output format: PASSED"
echo "  3. Compatibility: See above output"
echo ""
echo "Next steps:"
echo "  - Run full benchmark: ./run_aime24_vllm.sh"
echo "  - Compare results: python compare_results.py --hf-results <path> --vllm-results <path>"
echo ""
