# Development Principles

This document outlines the core development principles for TriAttention_vLLM.

## 0. Core Principles

### 0.1 Minimal Modification Principle

- **Reuse First**: Leverage vLLM's official implementation whenever possible, avoid building parallel systems
- **Minimal Intrusion**: Only modify and add where necessary, avoid large-scale refactoring
- **Extend, Don't Replace**: Prefer extension through inheritance, wrapping, hooks instead of replacing existing code
- **Overflow Pages Design**:
  - Don't heavily modify vLLM engine, wrap it externally for overflow management
  - Reuse vLLM's existing block allocator and KV cache management
  - Minimize changes, only add necessary budget/overflow tracking logic

### 0.2 Phase 1: Strict R-KV Script Alignment

Phase 1 implementation must **strictly align** with the behavior of the following three reference scripts:

| Variant | Reference Script |
|---------|------------------|
| per-head | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perhead.sh` |
| per-layer-per-head | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_layer_perhead.sh` |
| per-layer | `R-KV/weian_script/aime_sampled8/speckv/aime24/run_speckv_aime24_qwen_norm_aligned_perlayer.sh` |

**Alignment Requirements (including but not limited to)**:
- Scoring formula, aggregation strategy (mean/max)
- Offsets geometric sequence generation
- Pruning trigger conditions (divide_length, budget)
- Per-head/per-layer token selection logic
- RoPE processing approach
- All default configuration parameters

### 0.3 Phase Compatibility

- Phase 1 implementation **must not block** subsequent phase development
- No need to deliberately reserve interfaces, but consider extension points
- Design decisions should consider feasibility of future features (e.g., memory-triggered compression)

## Reference Files

### Source Files (R-KV)

| File | Purpose |
|------|---------|
| `R-KV/weian_development/speckv/speckv_rkv_style.py` | Main implementation |
| `R-KV/weian_development/speckv/round_pruning_utils.py` | Scoring, RoPE |
| `R-KV/weian_development/tests/` | Test suite |
| `R-KV/HuggingFace/evaluation/` | Evaluation scripts |

### Target Files (vLLM 0.15.x)

| File | Purpose |
|------|---------|
| `vllm/attention/layer.py` | Unified Attention layer |
| `vllm/v1/attention/backends/triton_attn.py` | Triton backend |
| `vllm/v1/attention/ops/triton_*.py` | Kernel examples |

---

*Document Version: 1.0*
*Created: 2026-02-02*
