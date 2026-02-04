# Parameter Alignment: HF SpeckV vs vLLM TriAttention

## Reference Implementation

**HF Script**: `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`

**Config**: `R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml`

## Parameter Comparison

### Core Model Parameters

| Parameter | HF Value | vLLM Value | Status | Notes |
|-----------|----------|------------|--------|-------|
| `model_path` | DeepSeek-R1-Distill-Qwen-7B | DeepSeek-R1-Distill-Qwen-7B | ✅ | Same model |
| `load_dtype` | bfloat16 | float16 | ⚠️ | **Difference**: HF uses bfloat16, vLLM uses float16 |
| `attn_implementation` | flash_attention_2 | (auto) | ✅ | vLLM auto-selects FlashAttention |
| `max_length` | 32768 | 32768 | ✅ | Same |
| `seed` | 888 | 888 | ✅ | Same |

### Compression Parameters

| Parameter | HF Value | vLLM Value | Status | Notes |
|-----------|----------|------------|--------|-------|
| `kv_budget` | 2048 | 2048 | ✅ | Same |
| `window_size` | 128 | 128 | ✅ | Same |
| `divide_length` | 128 | 128 | ✅ | Same |
| `sparse_round_window` | 32 | 32 | ✅ | Same |
| `sparse_offset_max_length` | 65536 | 65536 | ✅ | Same |
| `sparse_score_aggregation` | mean | mean | ✅ | Same |
| `sparse_normalize_scores` | true | true | ✅ | Same |
| `include_prefill_in_budget` | true | true | ✅ | Same |
| `rkv_style_compression` | true | true | ✅ | Same |
| `rkv_style_slack_trigger` | true | true | ✅ | Same |

### Pruning Mode

| Parameter | HF Value | vLLM Value | Status | Notes |
|-----------|----------|------------|--------|-------|
| Pruning strategy | `--per-head-pruning` | `per_head` | ✅ | Equivalent |

### Stats Loading

| Parameter | HF Value | vLLM Value | Status | Notes |
|-----------|----------|------------|--------|-------|
| `sparse_stats_path` | `R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt` | Same | ✅ | AIME24 test uses AIME25 stats (cross-dataset) |

### Generation Parameters

| Parameter | HF Value | vLLM Value | Status | Notes |
|-----------|----------|------------|--------|-------|
| `num_samples` | 8 | 8 | ✅ | Same |
| `temperature` | 0.6 | 0.6 | ✅ | Same |
| `top_p` | 0.95 | 0.95 | ✅ | Same |
| `top_k` | (default) | -1 (disabled) | ✅ | Both disabled |

### Execution Parameters

| Parameter | HF Value | vLLM Value | Status | Notes |
|-----------|----------|------------|--------|-------|
| `eval_batch_size` | 1 | 1 (implicit) | ✅ | vLLM processes 1 prompt at a time |
| `reset_cache_each_batch` | false | (not applicable) | ✅ | vLLM manages cache per request |
| Tensor parallelism | N/A | 2 GPUs | ➕ | vLLM addition for faster inference |
| GPU memory util | N/A | 0.85 | ➕ | vLLM addition |

## Critical Differences

### 1. Data Type ⚠️

**HF**: `bfloat16`
**vLLM**: `float16`

**Impact**: May cause minor numerical differences in generation
**Recommendation**: Test with `bfloat16` in vLLM if accuracy differs significantly

**Fix**:
```bash
# In run_aime24_vllm.sh, change:
--load-dtype float16
# To:
--load-dtype bfloat16
```

### 2. Tensor Parallelism ➕

**HF**: Single GPU
**vLLM**: 2 GPUs with tensor parallelism

**Impact**: Faster inference, but different memory layout
**Recommendation**: Keep tensor parallelism for speed, verify accuracy is maintained

### 3. Execution Model 📋

**HF**: Sequential batch processing with explicit cache reset
**vLLM**: Request-based processing with automatic cache management

**Impact**: Different cache lifecycle management
**Recommendation**: Verify compression state is properly reset between requests

## Alignment Verification

### Before Running Production

1. **Check dtype compatibility**:
   ```python
   # Verify model can load with bfloat16
   llm = LLM(model=model_path, dtype="bfloat16", ...)
   ```

2. **Verify stats loading**:
   ```python
   # Check stats file exists and loads correctly
   import torch
   stats = torch.load(stats_path)
   print(stats.keys())
   ```

3. **Test with single GPU first**:
   ```bash
   # Remove tensor parallelism for exact comparison
   --tensor-parallel-size 1
   ```

## Expected Behavior Differences

### Acceptable Differences

1. **Token-level variation**: Due to sampling, exact outputs will differ
2. **Speed**: vLLM should be faster due to tensor parallelism
3. **Memory usage**: Different patterns due to paged attention

### Unacceptable Differences

1. **Accuracy**: Should be within 1-2%
2. **Compression frequency**: Should compress at same thresholds
3. **Output format**: Should be compatible with comparison tools

## Validation Commands

### Check Current vLLM Config
```bash
grep -A5 "python3.*run_math_vllm.py" benchmarks/reasoning/run_aime24_vllm.sh
```

### Check HF Config
```bash
cat R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml
```

### Compare Parameters
```bash
# Extract vLLM args
grep "\-\-" benchmarks/reasoning/run_aime24_vllm.sh | sort > /tmp/vllm_args.txt

# Extract HF args
cat R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml | grep ":" | sort > /tmp/hf_args.txt

# Compare
diff -u /tmp/hf_args.txt /tmp/vllm_args.txt
```

## Recommended Configuration

For exact alignment with HF baseline:

```bash
python3 run_math_vllm.py \
  --model-path "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B" \
  --dataset-path "data/aime24.jsonl" \
  --output-path "outputs/results.jsonl" \
  --kv-budget 2048 \
  --window-size 128 \
  --divide-length 128 \
  --sparse-round-window 32 \
  --sparse-offset-max-length 65536 \
  --sparse-score-aggregation mean \
  --pruning-mode per_head \
  --sparse-normalize-scores \
  --include-prefill-in-budget \
  --rkv-style-compression \
  --rkv-style-slack-trigger \
  --sparse-stats-path "stats/deepseek_r1_qwen7b_plain_stats.pt" \
  --num-samples 8 \
  --temperature 0.6 \
  --top-p 0.95 \
  --seed 888 \
  --load-dtype bfloat16 \  # Match HF dtype!
  --tensor-parallel-size 1 \  # For exact comparison
  --max-tokens 32768
```

## Notes on AIME24 vs AIME25

The HF config uses **AIME25 stats** for **AIME24 testing** to avoid data leakage:
- Stats file: from AIME25 dataset
- Test dataset: AIME24 questions
- This is intentional for fair evaluation

vLLM should use the **same stats file** to maintain alignment.

## References

- HF runner: `R-KV/weian_development/rkv_sharded_runner.py`
- vLLM runner: `TriAttention_vLLM/benchmarks/reasoning/run_math_vllm.py`
- Config loader: `R-KV/weian_development/rkv_sharded_dispatch.py`
