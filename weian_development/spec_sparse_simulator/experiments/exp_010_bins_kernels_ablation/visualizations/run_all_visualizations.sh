#!/bin/bash
set -euo pipefail

# Get script directory for absolute paths
SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

# Configuration parameters
LAYER_HEAD='15-20'
SEED=42
ROUND_IDX=40

echo "Running all visualizations for layer-head ${LAYER_HEAD}..."
echo "=============================================="

# Step 1: Global QK Distribution
echo ""
echo "Step 1/3: Global QK Distribution..."
python "${SCRIPT_DIR}/viz_global_qk_distribution.py" --layer-head "${LAYER_HEAD}"

# Step 2: Per-Round Analysis
echo ""
echo "Step 2/3: Per-Round Analysis (seed=${SEED})..."
python "${SCRIPT_DIR}/viz_per_round_analysis.py" --layer-head "${LAYER_HEAD}" --seed "${SEED}"

# Step 3: Probe/Bin Assignment
echo ""
echo "Step 3/4: Probe/Bin Assignment (round=${ROUND_IDX})..."
python "${SCRIPT_DIR}/viz_probe_bin_assignment.py" --layer-head "${LAYER_HEAD}" --round-idx "${ROUND_IDX}"

# Step 4: Query Distribution
echo ""
echo "Step 4/6: Query Distribution..."
python "${SCRIPT_DIR}/viz_query_distribution.py" --layer-head "${LAYER_HEAD}"

# Step 5: Train-Test Key Comparison
echo ""
echo "Step 5/6: Train-Test Key Comparison..."
python "${SCRIPT_DIR}/viz_train_test_key_comparison.py" --layer-head "${LAYER_HEAD}"

# Step 6: Train-Test Query Comparison
echo ""
echo "Step 6/6: Train-Test Query Comparison..."
python "${SCRIPT_DIR}/viz_train_test_query_comparison.py" --layer-head "${LAYER_HEAD}"

# Verify outputs
echo ""
echo "=============================================="
echo "Verifying outputs..."
OUTPUT_DIR="${SCRIPT_DIR}/../output/visualizations"
COUNT=$(ls "${OUTPUT_DIR}"/*.png 2>/dev/null | wc -l)

if [ "${COUNT}" -ge 6 ]; then
    echo "SUCCESS: All 6 visualizations generated in ${OUTPUT_DIR}"
    ls -1 "${OUTPUT_DIR}"/*.png
    exit 0
else
    echo "ERROR: Expected 6 PNG files, found ${COUNT}"
    exit 1
fi
