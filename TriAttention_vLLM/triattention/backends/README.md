# TriAttention Backend for vLLM

This directory contains the **Backend Inheritance Approach (Plan A)** for integrating TriAttention with vLLM.

## Overview

The TriAttention backend extends vLLM's `FlashAttentionBackend` to add KV cache compression capabilities. This is an alternative to the monkey-patching approach in `vllm_integration.py`.

## Architecture

```
TriAttentionBackend (inherits FlashAttentionBackend)
    ├── get_name() -> "TRIATTENTION"
    ├── get_impl_cls() -> TriAttentionImpl
    └── [other methods inherited from FlashAttentionBackend]

TriAttentionImpl (inherits FlashAttentionImpl)
    ├── __init__() - Initialize with model parameters
    ├── forward() - Wrap parent's forward() with compression
    └── [other methods inherited from FlashAttentionImpl]
```

## Files

- `__init__.py` - Public API: `setup_triattention()`, `register_triattention_backend()`
- `triattention_backend.py` - Backend class (~60 lines)
- `triattention_impl.py` - Implementation class with compression logic (~220 lines)

## Usage

### Basic Usage

```python
from triattention import TriAttentionConfig
from triattention.backends import setup_triattention, register_triattention_backend
from vllm import LLM, SamplingParams

# 1. Configure TriAttention
config = TriAttentionConfig(
    stats_path="path/to/stats.pt",
    kv_budget=2048,
    divide_length=128,
)
setup_triattention(config)

# 2. Register backend
register_triattention_backend()

# 3. Use vLLM normally
llm = LLM(model="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B", enforce_eager=True)
outputs = llm.generate(prompts, SamplingParams(max_tokens=100))
```

### Advanced Configuration

```python
config = TriAttentionConfig(
    stats_path="stats.pt",
    kv_budget=2048,
    divide_length=128,
    pruning_mode="per_head",          # per_head or per_layer
    protect_prefill=True,             # Protect prefill tokens
    window_size=32,                   # Protect recent N tokens
    sparse_normalize_scores=True,     # Normalize scores before TopK
)
```

## Plan A vs Plan B Comparison

| Aspect | Plan A: Backend Inheritance | Plan B: Monkey Patching |
|--------|----------------------------|-------------------------|
| **Code Location** | `triattention/backends/` | `triattention/vllm_integration.py` |
| **Integration** | Subclass FlashAttentionBackend | Patch `forward()` at runtime |
| **Registration** | `register_triattention_backend()` | `patch_vllm_attention(model, wrapper)` |
| **Architecture** | Clean OOP inheritance | Runtime method replacement |
| **Maintainability** | High - follows vLLM patterns | Medium - requires understanding vLLM internals |
| **Compatibility** | Tied to vLLM backend API | Tied to specific method signatures |
| **Code Duplication** | Minimal - inherits everything | Minimal - reuses compression logic |
| **Testing** | Easy - standard class testing | Harder - requires model instance |
| **vLLM Version** | Depends on backend stability | Depends on method signatures |

## When to Use

### Use Backend Approach (Plan A) when:
- ✅ You want clean, maintainable code
- ✅ You're integrating into a larger system
- ✅ You need to extend/customize backend behavior
- ✅ You want standard OOP patterns

### Use Monkey Patching (Plan B) when:
- ✅ You need quick prototyping
- ✅ vLLM backend API changes frequently
- ✅ You want to patch specific models only
- ✅ You need fine-grained control over patching

## Implementation Details

### TriAttentionBackend

Minimal subclass that:
1. Overrides `get_name()` to return `"TRIATTENTION"`
2. Overrides `get_impl_cls()` to return `TriAttentionImpl`
3. Inherits all other methods (KV cache shape, block operations, metadata)

### TriAttentionImpl

Wraps `FlashAttentionImpl.forward()`:
1. Calls `super().forward()` to populate KV cache
2. Checks if compression is needed (decode step + threshold)
3. Calls `_apply_triattention_compression()` from `vllm_integration.py`
4. Returns the output unchanged

### Compression Logic Reuse

Both Plan A and Plan B use the **same compression implementation**:
- `_apply_triattention_compression()` - Main compression entry point
- `TriAttentionWrapper` - Per-request state management
- `TriAttentionCompressor` - Scoring + TopK selection

This ensures **behavioral consistency** between the two approaches.

## Limitations

### Current Limitations:
1. **Layer Index Detection**: vLLM doesn't expose `layer_idx` in attention layer
   - Workaround: Use heuristic or set via custom attribute

2. **Block Size Detection**: Defaults to 16, should be extracted from `cache_config`
   - Workaround: Try to read from `layer.cache_config.block_size`

3. **CUDA Graphs**: Not compatible with compression (dynamic cache modification)
   - Requirement: Must use `enforce_eager=True`

### Future Improvements:
- [ ] Auto-detect layer index from model structure
- [ ] Extract block_size from vLLM config automatically
- [ ] Support CUDA graphs (requires static compression plan)
- [ ] Add backend-specific metrics/monitoring
- [ ] Implement backend-level configuration validation

## Testing

### Unit Testing
```python
# Test backend registration
from triattention.backends import TriAttentionBackend
from vllm.attention.backends.flash_attn import FlashAttentionBackend

assert issubclass(TriAttentionBackend, FlashAttentionBackend)
assert TriAttentionBackend.get_name() == "TRIATTENTION"
```

### Integration Testing
```python
# Test with vLLM
from triattention.backends import setup_triattention, register_triattention_backend
from triattention import TriAttentionConfig
from vllm import LLM

config = TriAttentionConfig(stats_path="stats.pt", kv_budget=2048)
setup_triattention(config)
register_triattention_backend()

llm = LLM(model="facebook/opt-125m", enforce_eager=True)
outputs = llm.generate(["Hello"], max_tokens=10)
assert len(outputs) == 1
```

## Migration Guide

### From Monkey Patching to Backend

**Before (Plan B):**
```python
from triattention.vllm_integration import TriAttentionWrapper, patch_vllm_attention

wrapper = TriAttentionWrapper(config)
llm = LLM(model="...")
model = llm.llm_engine.model_executor.driver_worker.model_runner.model
patch_vllm_attention(model, wrapper)
```

**After (Plan A):**
```python
from triattention.backends import setup_triattention, register_triattention_backend

setup_triattention(config)
register_triattention_backend()
llm = LLM(model="...", enforce_eager=True)
```

### Behavioral Differences
- **None** - Both use the same compression logic
- Compression triggers at same thresholds
- Same per-request state management
- Same numerical results

## Contributing

When extending the backend:
1. Keep `TriAttentionBackend` minimal - only override what's necessary
2. Put compression logic in `TriAttentionImpl.forward()`
3. Reuse functions from `vllm_integration.py` to avoid duplication
4. Add docstrings explaining vLLM integration points
5. Test with multiple vLLM versions to ensure compatibility

## References

- vLLM Attention Backends: `vllm/attention/backends/`
- FlashAttention Backend: `vllm/attention/backends/flash_attn.py`
- TriAttention Core: `triattention/compressor.py`
- Monkey Patching Approach: `triattention/vllm_integration.py`
