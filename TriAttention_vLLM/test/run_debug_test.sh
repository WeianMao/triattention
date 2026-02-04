#!/bin/bash
# Simple test script to debug TriAttention compression trigger issue
#
# This script runs a minimal vLLM test to observe:
# 1. Whether the hook is called (_apply_triattention_compression)
# 2. What seq_len values are seen during generation
# 3. Whether should_compress returns True/False
#
# Expected output with kv_budget=256, divide_length=64:
# - seq_len should grow from prefill length to 512
# - should_compress should return True when seq_len >= 320

set -e

cd /data/rbg/users/weian/project/rl/dc/TriAttention_vLLM

# Setup environment
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate rkv

export LD_LIBRARY_PATH=/data/rbg/users/weian/env/miniconda3/envs/rkv/lib/python3.10/site-packages/nvidia/cudnn/lib:$LD_LIBRARY_PATH
export VLLM_ENABLE_V1_MULTIPROCESSING=0
export CUDA_VISIBLE_DEVICES=1
export PYTHONPATH="/data/rbg/users/weian/project/rl/dc:$PYTHONPATH"

echo "=========================================="
echo "TriAttention Compression Trigger Debug"
echo "=========================================="
echo ""
echo "Test configuration:"
echo "  kv_budget: 256"
echo "  divide_length: 64"
echo "  trigger_threshold: 320 tokens"
echo "  generating: 512 tokens"
echo ""
echo "Expected behavior:"
echo "  1. [DEBUG HOOK] logs appear when hook is called"
echo "  2. [DEBUG] logs show seq_len values"
echo "  3. should_compress=True when seq_len >= 320"
echo ""
echo "=========================================="
echo ""

# Run test and filter for debug output
exec -a PD-L1_binder timeout 90 python test/test_vllm_compression_trigger.py 2>&1 | grep -E "DEBUG|should_compress|Compression triggered|ERROR|Traceback" || echo "Test completed or timed out"
