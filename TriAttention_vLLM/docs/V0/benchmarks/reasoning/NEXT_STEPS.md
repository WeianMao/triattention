# Next Steps for TriAttention vLLM Integration

## Current Status Summary

### ✅ What's Working

1. **Inference Pipeline**
   - vLLM engine initializes correctly
   - Model loads and runs inference
   - Generation completes without errors
   - Output written to JSONL

2. **TriAttention Integration**
   - Hook successfully patches 28 attention layers
   - Wrapper reports `compression_enabled: true`
   - Config properly initialized

3. **Output Format**
   - Compatible with comparison tools
   - `compare_results.py` handles both HF and vLLM formats
   - All required fields present

### ⚠️ What Needs Verification

1. **Compression Actually Triggers**
   - Hook is installed ✅
   - Need to verify compression events occur during generation
   - Should see `[TriAttention] Compressing: seq_len=...` in logs

2. **Accuracy Matches HF Baseline**
   - Need to run full AIME24 benchmark
   - Compare accuracy with HF SpeckV implementation
   - Expected: within 1-2% difference

## Immediate Action Items

### Step 1: Verify Compression Triggers (5 minutes)

Run the compression verification script:

```bash
cd /data/rbg/users/weian/project/rl/dc/TriAttention_vLLM
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate trivllm
export CUDA_VISIBLE_DEVICES=1

# Quick verification
python3 benchmarks/reasoning/verify_compression.py
```

**Expected output**:
```
✓ Compression TRIGGERED: X times
Compression details:
  1. Layer 0: seq_len=2176
  2. Layer 1: seq_len=2176
  ...
```

**If compression NOT triggered**:
- Check sequence length (must exceed `budget + divide_length`)
- Verify `should_compress()` logic
- Check state initialization

### Step 2: Run Full AIME24 Benchmark (30-60 minutes)

Run the production script:

```bash
cd /data/rbg/users/weian/project/rl/dc/TriAttention_vLLM

# Make sure output directory is clean
rm -rf outputs/aime24_vllm_perhead/

# Run benchmark (uses 2 GPUs with tensor parallelism)
./benchmarks/reasoning/run_aime24_vllm.sh
```

**Monitor progress**:
- Watch for compression messages: `[TriAttention] Compressing: seq_len=...`
- Check GPU utilization: `nvidia-smi`
- Monitor output file: `tail -f outputs/aime24_vllm_perhead/results.jsonl`

**Success criteria**:
- All questions processed
- No errors in log
- Compression events visible
- Output file has correct number of entries

### Step 3: Compare with HF Baseline (5 minutes)

```bash
# Find HF baseline results
HF_RESULTS="R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/results.jsonl"
VLLM_RESULTS="TriAttention_vLLM/outputs/aime24_vllm_perhead/results.jsonl"

# Run comparison
python3 benchmarks/reasoning/compare_results.py \
    --hf-results "${HF_RESULTS}" \
    --vllm-results "${VLLM_RESULTS}" \
    --output-report comparison_report.txt \
    --detailed
```

**Expected results**:
- Accuracy difference: <2%
- Token match ratio: >80% (due to sampling variance)
- No systematic differences

### Step 4: Investigate Any Issues

If accuracy differs significantly:

1. **Check configuration alignment**:
   ```bash
   # Compare parameters in scripts
   diff -u \
       R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh \
       TriAttention_vLLM/benchmarks/reasoning/run_aime24_vllm.sh
   ```

2. **Check compression behavior**:
   ```bash
   # Count compression events in logs
   grep "Compressing:" logs/aime24_vllm_perhead/*.log | wc -l
   ```

3. **Verify stats loading**:
   ```bash
   # Check if stats file loads correctly
   ls -lh R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt
   ```

## Quick Reference Commands

### Run Minimal Test
```bash
./benchmarks/reasoning/test_minimal.sh
```

### Check Output Format
```bash
python3 benchmarks/reasoning/check_output_format.py \
    --vllm-results outputs/test_minimal/results.jsonl
```

### Run Verification Suite
```bash
./benchmarks/reasoning/run_verification_suite.sh
```

### View Recent Logs
```bash
tail -f logs/test_minimal/*.log
```

## Troubleshooting Guide

### Issue: Compression Not Triggering

**Symptoms**: No `[TriAttention] Compressing:` messages in logs

**Diagnosis**:
```python
# Check trigger threshold
budget = 2048
divide_length = 128
threshold = budget + divide_length  # 2176

# Check sequence lengths in generation
# Should exceed 2176 for compression to trigger
```

**Solutions**:
1. Lower budget to trigger faster: `--kv-budget 512`
2. Increase generation length: `--max-tokens 8192`
3. Check `should_compress()` logic in `state.py`

### Issue: Accuracy Significantly Different

**Symptoms**: >5% accuracy difference from HF baseline

**Possible causes**:
1. Stats file not loading correctly
2. Parameter mismatch (check configs)
3. Compression logic error
4. Random seed difference

**Solutions**:
1. Verify stats path: `--sparse-stats-path <path>`
2. Check parameter alignment
3. Run equivalence test: `test/test_rkv_triattention_equivalence.py`
4. Set same seed: `--seed 888`

### Issue: Out of Memory

**Symptoms**: CUDA OOM errors

**Solutions**:
1. Reduce batch size (already 1)
2. Lower GPU memory utilization: `--gpu-memory-utilization 0.7`
3. Reduce max model length: `--max-tokens 16384`
4. Use smaller model for testing: Qwen2.5-1.5B

### Issue: Format Compatibility Error

**Symptoms**: `compare_results.py` fails to parse results

**Solutions**:
1. Check JSONL format: `jq . results.jsonl | head`
2. Verify fields present: `check_output_format.py`
3. Check for empty lines or invalid JSON

## Success Checklist

Before considering production-ready:

- [ ] Compression triggers verified (see events in logs)
- [ ] Full AIME24 benchmark completes
- [ ] Accuracy within 2% of HF baseline
- [ ] Output format validated
- [ ] No memory leaks (check with long runs)
- [ ] Tensor parallelism works (2+ GPUs)
- [ ] Stats file loads correctly

## Contact / References

**Key Files**:
- Integration: `triattention/vllm_integration.py`
- Benchmark: `benchmarks/reasoning/run_math_vllm.py`
- Production: `benchmarks/reasoning/run_aime24_vllm.sh`

**Reference Implementation**:
- HF script: `R-KV/HuggingFace/run_math.py`
- HF runner: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`

**Documentation**:
- Verification status: `VERIFICATION_STATUS.md`
- Ready to run guide: `READY_TO_RUN.md`
- README: `README.md`
