# Review Checklist

This document provides a comprehensive checklist for reviewing TriAttention_vLLM implementations.

## Code Quality

- [ ] Code follows existing vLLM conventions and patterns
- [ ] Minimal modifications to vLLM core (extension over replacement)
- [ ] No duplicated logic - reuses existing vLLM utilities
- [ ] Clear variable names and function signatures
- [ ] Appropriate error handling and edge case coverage

## Correctness

- [ ] Strict alignment with R-KV reference scripts (Phase 1)
  - [ ] Scoring formula matches reference implementation
  - [ ] Aggregation strategy (mean/max) correct
  - [ ] Offsets geometric sequence generation aligned
  - [ ] Pruning trigger conditions match (divide_length, budget)
  - [ ] Per-head/per-layer token selection logic correct
  - [ ] RoPE processing approach aligned
  - [ ] All config parameters default correctly

- [ ] Position indices handling
  - [ ] Proper indexing for PagedAttention blocks
  - [ ] Correct dtype (int32/int64 as needed)
  - [ ] Valid for both prefill and decode paths

- [ ] KV cache format compatibility
  - [ ] Matches vLLM's PagedAttention layout
  - [ ] Correct shape handling for [num_blocks, block_size, num_heads, head_size]
  - [ ] Proper handling of partial blocks

## Testing

- [ ] Unit tests for core components
  - [ ] Scoring kernels
  - [ ] Pruning logic
  - [ ] RoPE utilities
  - [ ] Configuration handling

- [ ] Integration tests
  - [ ] vLLM hook integration
  - [ ] PagedAttention compatibility
  - [ ] Per-request state isolation

- [ ] Correctness tests
  - [ ] Output matches R-KV reference (< 1% perplexity diff)
  - [ ] FP16/BF16 numerical stability
  - [ ] Equivalence tests pass

- [ ] Edge case coverage
  - [ ] Empty sequences
  - [ ] Single token sequences
  - [ ] Budget exceeded scenarios
  - [ ] Request cancellation/slot reuse

## Performance

- [ ] No unnecessary allocations in hot path
- [ ] Efficient Triton kernel implementations
- [ ] Proper use of torch.compile where applicable
- [ ] Memory usage within expected bounds
- [ ] Throughput meets targets (>= 1.5x at 2048 budget)
- [ ] Latency overhead acceptable (< 10%)

## vLLM Integration

- [ ] Compatible with PagedAttention
- [ ] Proper hook registration and invocation
- [ ] No interference with CUDA Graph mode (or proper fallback)
- [ ] Supports common inference paths (prefill, decode)
- [ ] Per-request state properly isolated
- [ ] Works with target models (Qwen, LLaMA, DeepSeek, Mistral)

## Documentation

- [ ] Implementation details documented in backend/reference/
- [ ] Design decisions recorded in DESIGN_DECISIONS.md
- [ ] Known issues tracked in interface/OPEN_ISSUES.md
- [ ] Pending decisions in interface/PENDING_DECISIONS.md
- [ ] Status updated in interface/CURRENT_STATUS.md

## Risks & Mitigations

- [ ] PagedAttention integration complexity addressed
- [ ] CUDA Graph compatibility verified (or eager fallback provided)
- [ ] Numerical precision validated (FP16/BF16)
- [ ] Performance profiled and optimized
- [ ] Stability tested (long-running, multi-request scenarios)

## Success Criteria

| Metric | Target | Status |
|--------|--------|--------|
| Correctness | < 1% perplexity difference | |
| Throughput | >= 1.5x (2048 budget) | |
| Latency overhead | < 10% | |
| Stability | 24h zero crashes | |
| Model coverage | LLaMA/Qwen/DeepSeek/Mistral | |

---

*Document Version: 1.0*
*Created: 2026-02-02*
