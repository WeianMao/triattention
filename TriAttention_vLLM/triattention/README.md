# TriAttention Core Library

Core implementation of TriAttention KV cache compression algorithm.

## Module Structure

```
triattention/
├── __init__.py          # Package exports
├── config.py            # TriAttentionConfig configuration class
├── state.py             # CompressionState state management
├── compressor.py        # TriAttentionCompressor main class
├── scoring.py           # Scoring logic (Python wrapper + PyTorch fallback)
├── utils.py             # Utility functions
└── kernels/             # Triton kernels
    ├── __init__.py
    └── scoring_kernel.py  # Triton scoring kernel (placeholder)
```

## Quick Start

```python
from triattention import TriAttentionCompressor, TriAttentionConfig

# Create configuration
config = TriAttentionConfig(
    kv_budget=2048,
    divide_length=128,
    stats_path="path/to/stats.pt",
    pruning_mode="per_head",
)

# Initialize compressor
compressor = TriAttentionCompressor(config)

# Compress KV cache
compressed_k, compressed_v, new_positions = compressor.compress(
    key_states=k_cache,      # [batch, num_kv_heads, seq_len, head_dim]
    value_states=v_cache,    # [batch, num_kv_heads, seq_len, head_dim]
    cache_positions=positions,  # [seq_len]
    layer_idx=0,
)
```

## Phase 1 Status

Current implementation status:

- [x] Configuration class with full parameter support
- [x] State management with per_head/per_layer position tracking
- [x] Utility functions (stats loading, RoPE verification)
- [x] Compressor skeleton with lazy initialization
- [x] PyTorch scoring fallback implementation
- [ ] Triton scoring kernel (placeholder - to be implemented)
- [x] PyTorch TopK + Gather (Phase 1 default)

## Design Alignment

This implementation aligns with:
- Phase 1 design specifications (docs/phases/phase1_triton_implementation/README.md)
- R-KV parameter conventions and naming
- Data structure requirements (docs/implementation/data_structures.md)
- vLLM integration constraints (docs/implementation/vllm_integration.md)

## Next Steps

1. Implement Triton scoring kernel (kernels/scoring_kernel.py)
2. Add correctness tests comparing Triton vs PyTorch scoring
3. Add performance benchmarks
4. Integrate with vLLM attention backend
