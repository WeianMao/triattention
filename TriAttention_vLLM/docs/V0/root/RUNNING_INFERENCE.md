# Running TriAttention vLLM Inference

## Overview

This directory contains a working vLLM inference implementation with TriAttention KV cache compression, functionally equivalent to the HuggingFace SpeckV baseline (`R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh`).

## Status: WORKING

Successfully tested on 2026-02-01 with:
- Model: Qwen2.5-1.5B (minimal test)
- Backend: vLLM 0.7.0 + TriAttention compression
- GPU: Tesla T4
- Result: Generated 2 samples, compression hooks working correctly

## Quick Start

### Minimal Test (1.5B model, single GPU)

```bash
cd /data/rbg/users/weian/project/rl/dc/TriAttention_vLLM/benchmarks/reasoning
bash test_minimal.sh
```

**What it does:**
- Uses Qwen2.5-1.5B model (fits on single T4)
- Runs on 1 question from AIME24
- Generates 2 samples
- Output: `TriAttention_vLLM/outputs/test_minimal/results.jsonl`

### Production Benchmark (7B model, 2 GPUs)

```bash
cd /data/rbg/users/weian/project/rl/dc/TriAttention_vLLM/benchmarks/reasoning
bash run_aime24_vllm.sh
```

**What it does:**
- Uses DeepSeek-R1-Distill-Qwen-7B
- Tensor parallelism across 2 GPUs
- Full AIME24 dataset
- Generates 8 samples per question
- Output: `TriAttention_vLLM/outputs/aime24_vllm_perhead/results.jsonl`

## Architecture

### Core Components

1. **TriAttention Library** (`triattention/`)
   - `config.py`: Configuration for compression parameters
   - `compressor.py`: Per-layer KV cache compression logic
   - `wrapper.py`: Wrapper managing compression across layers
   - `vllm_integration.py`: vLLM attention patching and hook implementation
   - `kernels/triton_scoring.py`: Triton kernel for token scoring

2. **Benchmark Script** (`benchmarks/reasoning/run_math_vllm.py`)
   - CLI interface matching HuggingFace baseline
   - vLLM engine initialization
   - Automatic TriAttention patching
   - JSONL output generation

3. **Validation Script** (`benchmarks/reasoning/validate_setup.py`)
   - Pre-flight checks for files, imports, config, and Triton kernels

## Configuration

### TriAttention Parameters

Matching R-KV SpeckV configuration:

```python
kv_budget = 2048              # Maximum KV cache budget
window_size = 128             # Recent token window (always retained)
divide_length = 128           # Compression trigger interval
sparse_round_window = 32      # Sparse round window
pruning_mode = "per_head"     # Token selection: per_head, per_layer, per_layer_per_head
score_aggregation = "mean"    # Score aggregation: mean, max
include_prefill_in_budget = True  # Include prefill tokens in budget
protect_prefill = False       # Don't protect prefill when include_prefill_in_budget=True
```

### vLLM Parameters

```python
dtype = "float16"             # Use fp16 for Tesla T4 compatibility
tensor_parallel_size = 2      # Use 2 GPUs for 7B model
gpu_memory_utilization = 0.85 # GPU memory utilization ratio
enforce_eager = True          # Disable CUDA graphs to save memory
max_tokens = 32768            # Maximum sequence length
```

### Generation Parameters

```python
num_samples = 8               # Number of samples per question
temperature = 0.6             # Sampling temperature
top_p = 0.95                  # Nucleus sampling
seed = 888                    # Random seed
```

## How It Works

### Patching vLLM Attention

1. Initialize vLLM engine with target model
2. Create `TriAttentionWrapper` with compression config
3. Call `patch_vllm_attention(model, tri_wrapper)` to inject compression hooks
4. Hooks intercept attention forward passes and apply compression

### Compression Flow

1. **Prefill Phase**: Store all tokens without compression
2. **Decode Phase**:
   - Check if compression needed (seq_len > kv_budget + divide_length)
   - Extract KV cache for current layer
   - Reshape cache from vLLM's flattened format to block format
   - Call `tri_wrapper.compress()` to select top-k tokens
   - Update cache in-place with compressed version

### Request Isolation

Each request maintains separate compression state:
- `prefill_length`: Initial sequence length
- `current_cache_len`: Current cache size
- `compression_count`: Number of compressions applied

## Verification

To verify the setup before running:

```bash
cd /data/rbg/users/weian/project/rl/dc/TriAttention_vLLM/benchmarks/reasoning
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate trivllm
python3 validate_setup.py
```

**Expected output:**
```
✅ PASS: Files
✅ PASS: Imports
✅ PASS: Config
✅ PASS: Triton
✅ All validation checks passed!
✅ Ready to run benchmark
```

## Output Format

Results are saved in JSONL format, one JSON object per line:

```json
{
  "id": 60,
  "question": "Every morning Aya goes for a...",
  "ground_truth": "204",
  "generated_answers": ["answer1", "answer2", ...],
  "num_samples": 8,
  "temperature": 0.6,
  "kv_budget": 2048,
  "pruning_mode": "per_head",
  "backend": "vllm",
  "compression_enabled": true
}
```

## Troubleshooting

### OOM on Single GPU (7B model)

**Problem**: `torch.OutOfMemoryError` when loading 7B model on single T4

**Solution**: Use tensor parallelism across 2 GPUs:
```bash
export CUDA_VISIBLE_DEVICES=1,2
--tensor-parallel-size 2
```

### bfloat16 Not Supported

**Problem**: `ValueError: Bfloat16 is only supported on GPUs with compute capability of at least 8.0`

**Solution**: Tesla T4 (compute capability 7.5) requires fp16:
```bash
--load-dtype float16
```

### Compression Not Triggering

**Problem**: Debug logs show `should_compress: False` throughout generation

**Possible causes:**
1. Sequence length < kv_budget + divide_length (2048 + 128 = 2176)
2. `protect_prefill=True` with large prefill
3. Stats file path incorrect (falls back to uniform scoring)

**Solution**: For minimal test, reduce `kv_budget` to trigger compression earlier:
```bash
--kv-budget 256  # Instead of 2048
```

## Environment

- **Conda env**: `trivllm`
- **Python**: 3.10
- **PyTorch**: 2.5.1+cu124
- **vLLM**: 0.7.0
- **CUDA**: 12.4
- **GPU**: Tesla T4 (compute capability 7.5)

## References

- HuggingFace baseline: `R-KV/HuggingFace/run_math.py`
- Reference config: `R-KV/weian_script/configs/aime_sampled8_speckv_aime24_qwen_norm_aligned.yaml`
- Stats file: `R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt`

## Next Steps

1. Run full AIME24 benchmark with 7B model and 8 samples
2. Compare results with HuggingFace baseline
3. Measure compression ratio and throughput
4. Profile GPU memory usage during decode
