# TriAttention vLLM Benchmark - Ready to Run

## Status: ✅ READY FOR TESTING

All validation checks have passed. The vLLM inference script is ready to run.

## What Was Fixed

### 1. Parameter Mapping Issue
**Problem**: The `setup_triattention_config()` function was missing critical parameters.

**Fixed**:
- Added `sparse_round_window` parameter mapping
- Added `include_prefill_in_budget` parameter mapping
- Fixed parameter name: `normalize_scores` → `sparse_normalize_scores`

### 2. Validation Infrastructure
Created comprehensive validation script (`validate_setup.py`) that checks:
- ✅ All required files exist (model, dataset, stats)
- ✅ All imports work correctly (PyTorch, vLLM, TriAttention)
- ✅ Configuration can be created without errors
- ✅ Triton kernels can be imported and compiled

## Configuration Alignment

The vLLM script now matches the HuggingFace SpeckV configuration:

| Parameter | HF Script | vLLM Script | Status |
|-----------|-----------|-------------|---------|
| kv_budget | 2048 | 2048 | ✅ |
| divide_length | 128 | 128 | ✅ |
| window_size | 128 | 128 | ✅ |
| sparse_round_window | 32 | 32 | ✅ |
| sparse_offset_max_length | 65536 | 65536 | ✅ |
| sparse_score_aggregation | mean | mean | ✅ |
| pruning_mode | per_head | per_head | ✅ |
| sparse_normalize_scores | true | true | ✅ |
| include_prefill_in_budget | true | true | ✅ |
| num_samples | 8 | 8 | ✅ |
| temperature | 0.6 | 0.6 | ✅ |
| top_p | 0.95 | 0.95 | ✅ |
| seed | 888 | 888 | ✅ |
| load_dtype | bfloat16 | bfloat16 | ✅ |
| max_length | 32768 | 32768 | ✅ |

## How to Run

### Quick Test (1 question, 2 samples)
```bash
cd /data/rbg/users/weian/project/rl/dc
./TriAttention_vLLM/benchmarks/reasoning/test_quick.sh
```

This will:
- Run on 1 question from AIME24
- Generate 2 samples per question
- Output: `TriAttention_vLLM/outputs/test_quick/results.jsonl`
- Logs: `TriAttention_vLLM/logs/test_quick/`

### Full Benchmark (all questions, 8 samples)
```bash
cd /data/rbg/users/weian/project/rl/dc
./TriAttention_vLLM/benchmarks/reasoning/run_triattention_aime24_perhead.sh
```

This will:
- Run on all 30 AIME24 questions
- Generate 8 samples per question
- Output: `TriAttention_vLLM/outputs/aime24/perhead/results.jsonl`
- Logs: `TriAttention_vLLM/logs/aime24/perhead/`

### Environment Requirements
- Conda environment: `trivllm` (vLLM 0.7.0)
- GPU: Tesla T4 or better
- Memory: ~16GB GPU RAM for DeepSeek-R1-Distill-Qwen-7B

## Expected Behavior

### During Initialization
```
[TriAttention] Patching vLLM attention layers...
[TriAttention] Model info: block_size=16, num_kv_heads=8, head_dim=128
[TriAttention] Successfully patched 28 attention layers
```

### During Inference
```
[DEBUG] Layer 0, Request decode_0:
  seq_len: 2176
  prefill_length: 0
  current_cache_len: 0
  compression_count: 0
  config.kv_budget: 2048
  config.divide_length: 128
  effective_size: 2176
  trigger_threshold: 2176
  should_compress: True

[TriAttention] Compressed layer 0: 2176 -> 2048 tokens (count: 1)
```

## File Locations

### Input Files
- Model: `/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B`
- Dataset: `/data/rbg/users/weian/project/rl/dc/R-KV/HuggingFace/data/aime24.jsonl`
- Stats: `/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt`

### Output Files
- Quick test: `TriAttention_vLLM/outputs/test_quick/results.jsonl`
- Full benchmark: `TriAttention_vLLM/outputs/aime24/perhead/results.jsonl`

### Scripts
- Main script: `TriAttention_vLLM/benchmarks/reasoning/run_math_vllm.py`
- Quick test: `TriAttention_vLLM/benchmarks/reasoning/test_quick.sh`
- Full benchmark: `TriAttention_vLLM/benchmarks/reasoning/run_triattention_aime24_perhead.sh`
- Validation: `TriAttention_vLLM/benchmarks/reasoning/validate_setup.py`

## Next Steps

1. **Run Quick Test**: Verify end-to-end functionality on 1 question
2. **Check Output Format**: Ensure JSONL output matches expected format
3. **Run Full Benchmark**: Execute on all 30 AIME24 questions
4. **Compare Results**: Use `compare_results.py` to compare with HF baseline

## Comparison with HF Baseline

To compare results with the HuggingFace SpeckV baseline:

```bash
python TriAttention_vLLM/benchmarks/reasoning/compare_results.py \
  --vllm-results TriAttention_vLLM/outputs/aime24/perhead/results.jsonl \
  --hf-results R-KV/outputs/aime_sampled8/speckv/aime24/norm_aligned_perhead/merged/results.jsonl \
  --output-dir TriAttention_vLLM/outputs/comparison/
```

Expected: Accuracy difference < 1%

## Technical Notes

### Why vLLM 0.7.0?
- TriAttention integration is designed for vLLM 0.7.0 API
- Different from vLLM 0.10.2 in `dc` environment
- Uses FlashAttentionImpl patching approach

### Per-Request State Isolation
- Each request maintains its own compression state
- State includes: prefill_length, absolute_position, compression_count
- Automatic cleanup when requests complete

### Compression Trigger Logic
- R-KV slack mode: trigger at `budget + divide_length`
- Example: With budget=2048, divide_length=128, triggers at 2176 tokens
- Compresses back down to budget (2048 tokens)

## Debugging

If something goes wrong, check:

1. **Validation**: `python TriAttention_vLLM/benchmarks/reasoning/validate_setup.py`
2. **Logs**: Check `TriAttention_vLLM/logs/` for detailed output
3. **GPU Memory**: Monitor with `nvidia-smi`
4. **Compression Stats**: Look for `[TriAttention] Compressed layer X: Y -> Z tokens` messages

## Contact

For issues or questions, refer to:
- Implementation docs: `TriAttention_vLLM/triattention/kernels/DELIVERY_SUMMARY.md`
- Test results: `TriAttention_vLLM/test/TEST_RESULTS_2026-02-01.md`
- Project roadmap: `TriAttention_vLLM/docs/project/roadmap.md`
