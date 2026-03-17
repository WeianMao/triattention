#!/bin/bash
# Verification script for seq_len synchronization fix

set -e

echo "================================================================================"
echo "VERIFICATION: seq_len Synchronization Fix"
echo "================================================================================"
echo ""

# Activate conda environment
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate rkv

echo "Running verification tests..."
echo ""

# Test 1: State tracking logic
echo "TEST 1: State Tracking Logic"
echo "--------------------------------------------------------------------------------"
python test_seq_len_sync.py
echo ""

# Test 2: Minimal simulation
echo ""
echo "TEST 2: Minimal Simulation"
echo "--------------------------------------------------------------------------------"
python test_compression_fix_minimal.py
echo ""

echo "================================================================================"
echo "VERIFICATION COMPLETE"
echo "================================================================================"
echo ""
echo "Expected Results:"
echo "  ✓ Compression triggers at seq_len=320 (prefill)"
echo "  ✓ Next compression at seq_len=384 (320 + 64)"
echo "  ✓ Subsequent compressions every 64 tokens"
echo "  ✓ NO compression on every decode step"
echo ""
echo "The fix ensures compression follows R-KV slack mode:"
echo "  - Cache fluctuates between [256, 320]"
echo "  - Compression overhead reduced by ~64x"
echo ""
