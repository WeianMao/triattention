# TriAttention + vLLM Quick Start Guide

## Installation

```bash
# Activate environment
conda activate trivllm

# Verify installation
python -c "import vllm; from triattention import patch_vllm_attention; print('✓ Ready')"
```

## Basic Usage

### Python API

```python
from vllm import LLM, SamplingParams
from triattention import TriAttentionConfig, TriAttentionWrapper, patch_vllm_attention

# 1. Configure TriAttention
config = TriAttentionConfig(
    kv_budget=2048,          # Maximum tokens to keep
    divide_length=128,        # Compression interval
    pruning_mode="per_head",  # Selection strategy
    stats_path=None,          # Optional: path to precomputed stats
)

# 2. Create wrapper
wrapper = TriAttentionWrapper(config)

# 3. Load vLLM model
llm = LLM(
    model="your-model-path",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
)

# 4. Enable compression
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
patch_vllm_attention(model, wrapper)

# 5. Generate
outputs = llm.generate(
    ["Your prompt here"],
    SamplingParams(temperature=0.6, max_tokens=32768)
)
```

### Command-Line Benchmark

```bash
python benchmarks/reasoning/run_math_vllm.py \
  --model-path /path/to/model \
  --dataset-path data/aime24.jsonl \
  --output-path outputs/results.jsonl \
  --kv-budget 2048 \
  --pruning-mode per_head
```

## Configuration Options

### TriAttentionConfig Parameters

```python
TriAttentionConfig(
    # === Core Parameters ===
    kv_budget=2048,              # Max tokens to retain
    divide_length=128,            # Compression trigger interval
    pruning_mode="per_head",      # "per_head" | "per_layer"

    # === Scoring Parameters ===
    score_aggregation="mean",     # "mean" | "max"
    offset_max_length=65536,      # Max offset for position scoring

    # === Budget Management ===
    window_size=128,              # Recent tokens always protected
    protect_prefill=True,         # Don't prune prefill tokens

    # === Stats Path ===
    stats_path="/path/to/stats.pt",  # Optional precomputed stats

    # === Model Config (auto-detected if not specified) ===
    head_dim=128,                 # Head dimension
    num_kv_heads=8,               # Number of KV heads
    num_layers=32,                # Number of layers
)
```

### Benchmark Script Options

```bash
# Model and data
--model-path PATH              # Model checkpoint path
--dataset-path PATH            # JSONL dataset file
--output-path PATH             # Output results file

# TriAttention compression
--kv-budget 2048               # KV cache budget
--divide-length 128            # Compression interval
--window-size 128              # Recent token window
--pruning-mode per_head        # Selection strategy
--sparse-stats-path PATH       # Precomputed stats (optional)

# Generation parameters
--num-samples 8                # Samples per question
--temperature 0.6              # Sampling temperature
--max-tokens 32768             # Max generation length

# System
--load-dtype bfloat16          # Model dtype
--gpu-memory-utilization 0.9   # GPU memory ratio
```

## Testing

### Quick Test

```bash
python test/test_simple_patch.py
```

### Full Test Suite

```bash
python test/test_vllm_hook.py
```

Expected: `5/5 tests passed`

## Verify Compression is Active

```python
# After patching
print(f"Compression enabled: {wrapper._patched}")
print(f"Active requests: {wrapper.get_active_requests()}")

# After generation
state = wrapper.get_request_state_summary("decode_0")
if state:
    print(f"Compression state: {state}")
```

## Troubleshooting

### Import Error

```python
# If you get ImportError, check path
import sys
sys.path.insert(0, '/data/rbg/users/weian/project/rl/dc/TriAttention_vLLM')
```

### Patching Failed

```python
# Check model structure
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
print(f"Model type: {type(model)}")
print(f"Has model.model: {hasattr(model, 'model')}")
print(f"Has layers: {hasattr(model.model, 'layers') if hasattr(model, 'model') else False}")
```

### No Compression Happening

Check:
1. `wrapper._patched == True`
2. Sequence length exceeds `kv_budget + divide_length`
3. Not during prefill phase (only decode is compressed)

## Performance Tips

1. **Set appropriate budget**: `kv_budget=2048` for math, higher for long context
2. **Use compression interval**: `divide_length=128` to amortize overhead
3. **Enable stats**: Precomputed stats improve compression quality
4. **Monitor memory**: Check GPU memory usage before/after compression

## Example: Math Reasoning

```python
from vllm import LLM, SamplingParams
from triattention import TriAttentionConfig, TriAttentionWrapper, patch_vllm_attention

# Configure for math reasoning
config = TriAttentionConfig(
    kv_budget=2048,
    divide_length=128,
    pruning_mode="per_head",
    window_size=128,
)

wrapper = TriAttentionWrapper(config)

# Load DeepSeek-R1
llm = LLM(
    model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
    dtype="bfloat16",
    gpu_memory_utilization=0.9,
    max_model_len=32768,
)

# Enable compression
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
patch_vllm_attention(model, wrapper)

# Generate solutions
question = "What is the sum of all positive integers less than 100?"
outputs = llm.generate(
    [question],
    SamplingParams(temperature=0.6, top_p=0.95, max_tokens=8192, n=8)
)

# Get answers
for i, output in enumerate(outputs[0].outputs):
    print(f"Sample {i+1}: {output.text[:100]}...")
```

## Documentation

- **Implementation Guide**: `docs/vllm_hook_implementation.md`
- **Summary**: `VLLM_HOOK_SUMMARY.md`
- **Code**: `triattention/vllm_integration.py`

## Support

**Environment**: `conda activate trivllm`
**vLLM Version**: 0.7.0
**Tested Models**: DeepSeek-R1-Distill-Qwen-7B, Llama, Qwen architectures

For issues, check:
1. Test suite passes: `python test/test_vllm_hook.py`
2. vLLM version: `python -c "import vllm; print(vllm.__version__)"`
3. Documentation: `docs/vllm_hook_implementation.md`
