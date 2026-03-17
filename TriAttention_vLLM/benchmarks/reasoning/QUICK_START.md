# Quick Start Guide: TriAttention vLLM Benchmark

## TL;DR

```bash
cd /data/rbg/users/weian/project/rl/dc/TriAttention_vLLM
conda activate trivllm

# 1. Verify compression works (5 min)
python3 benchmarks/reasoning/verify_compression.py

# 2. Run full benchmark (30-60 min)
./benchmarks/reasoning/run_aime24_vllm.sh

# 3. Compare with HF baseline (5 min)
python3 benchmarks/reasoning/compare_results.py \
    --hf-results /path/to/hf/results.jsonl \
    --vllm-results outputs/aime24_vllm_perhead/results.jsonl \
    --detailed
```

## Status: Ready for Testing

✅ **Infrastructure**: Complete
- Inference works
- Hook integrated
- Parameters aligned
- Output format compatible

⏳ **Validation**: Needs testing
- Compression trigger (ready to verify)
- Full benchmark (ready to run)
- Accuracy comparison (ready to compare)

## Quick Commands

### Verify Setup
```bash
# Run all verification tests
./benchmarks/reasoning/run_verification_suite.sh
```

### Check Output Format
```bash
# Check vLLM output
python3 benchmarks/reasoning/check_output_format.py \
    --vllm-results outputs/test_minimal/results.jsonl
```

### Test Minimal
```bash
# Quick test on 1 question (2 samples)
./benchmarks/reasoning/test_minimal.sh
```

### Production Run
```bash
# Full AIME24 benchmark (8 samples per question)
./benchmarks/reasoning/run_aime24_vllm.sh
```

## What's Different from HF

| Aspect | HF | vLLM |
|--------|-----|------|
| Backend | Transformers | vLLM |
| Execution | Sequential batches | Request-based |
| Parallelism | Single GPU | 2 GPUs (tensor parallel) |
| Cache | Manual reset | Auto per-request |
| Speed | Slower | Faster |
| Output | `output` field | `generated_answers` field |

**Good news**: All parameters aligned, output format compatible!

## Current Configuration

**Model**: DeepSeek-R1-Distill-Qwen-7B
**Dataset**: AIME24 (math problems)
**Stats**: AIME25 stats (cross-dataset, no leakage)
**Compression**:
- Budget: 2048 tokens
- Window: 128 tokens
- Trigger: 2176 tokens (budget + divide_length)
- Mode: per_head

**Generation**:
- Samples: 8 per question
- Temperature: 0.6
- Top-p: 0.95
- Seed: 888

**System**:
- GPUs: 2 (tensor parallel)
- Dtype: bfloat16
- Max tokens: 32768

## Files to Know

**Run scripts**:
- `run_aime24_vllm.sh` - Production benchmark
- `test_minimal.sh` - Quick test
- `run_verification_suite.sh` - All verification tests

**Verification**:
- `verify_compression.py` - Check compression triggers
- `check_output_format.py` - Validate output format
- `compare_results.py` - Compare HF vs vLLM results

**Documentation**:
- `NEXT_STEPS.md` - Detailed action guide
- `VERIFICATION_STATUS.md` - Current status
- `PARAMETER_ALIGNMENT.md` - HF vs vLLM params
- `AGENT3_SUMMARY.md` - What was done

## Troubleshooting

**Compression not working?**
```bash
# Lower budget to trigger faster
python3 run_math_vllm.py ... --kv-budget 512
```

**Out of memory?**
```bash
# Reduce GPU memory utilization
--gpu-memory-utilization 0.7
```

**Format errors?**
```bash
# Check JSONL structure
jq . results.jsonl | head
```

## Expected Results

**Compression**:
- Should trigger at seq_len >= 2176
- Compress down to budget (2048)
- See: `[TriAttention] Compressing: seq_len=...`

**Accuracy**:
- Should match HF within 1-2%
- Some variation due to sampling
- Use --detailed for token comparison

**Performance**:
- vLLM should be faster (tensor parallel)
- Lower latency per token
- Higher throughput

## Success Checklist

- [ ] `verify_compression.py` shows compression events
- [ ] `run_aime24_vllm.sh` completes without errors
- [ ] Accuracy within 2% of HF baseline
- [ ] No memory leaks (check nvidia-smi)

## References

**HF baseline**: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`

**vLLM implementation**: `TriAttention_vLLM/triattention/vllm_integration.py`

**Comparison tool**: `benchmarks/reasoning/compare_results.py`
