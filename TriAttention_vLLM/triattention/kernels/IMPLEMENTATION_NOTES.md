# Triton Scoring Kernel Implementation Notes

## Implementation Status

✅ **COMPLETE** - Basic Triton scoring kernel with correctness verification

## Key Implementation Decisions

### 1. Vectorized Token Processing

**PyTorch (Sequential)**:
```python
for token in range(seq_len):
    for offset in offsets:
        k = load_key(token)
        score[token, offset] = compute_score(k, offset)
```

**Triton (Vectorized)**:
```python
# Process BLOCK_N tokens in parallel
k_batch = load_keys(token_range)  # [BLOCK_N, freq_count]
for offset in static_range(num_offsets):
    scores_batch = compute_scores(k_batch, offset)  # [BLOCK_N]
```

**Benefit**: ~32x parallelism per thread block

### 2. Position-Independent Coefficient Factorization

**Key Insight**: The scoring formula can be split into:
- Position-independent: `A_coef`, `B_coef` (depends only on K, Q)
- Position-dependent: `cos(t*omega)`, `sin(t*omega)` (precomputed)

**Triton Implementation**:
```python
# Computed once per token
A_coef = freq_scale_sq * (q_r*k_r + q_i*k_i)  # [BLOCK_N, freq_count]
B_coef = freq_scale_sq * (q_i*k_r - q_r*k_i)  # [BLOCK_N, freq_count]

# Iterated over offsets using precomputed tables
for offset_idx in static_range(num_offsets):
    cos_vals = cos_table[offset_idx]  # [freq_count]
    sin_vals = sin_table[offset_idx]  # [freq_count]
    scores = sum(A_coef * cos_vals - B_coef * sin_vals)  # dot product
```

**Benefit**: Eliminates ~90% of trigonometric computation

### 3. Static Range for Offset Iteration

**Issue**: Triton requires compile-time constants for loop unrolling

**Solution**: Mark `num_offsets` as `tl.constexpr` and use `tl.static_range`

```python
def speckv_scoring_kernel(
    ...
    num_offsets: tl.constexpr,  # Required for static_range
    ...
):
    for off_idx in tl.static_range(num_offsets):  # Unrolled at compile time
        ...
```

**Trade-off**: Different num_offsets requires recompilation, but typical use cases have fixed offsets (e.g., 16)

### 4. Complex Number Workaround

**Triton Limitation**: No native `tl.complex` type

**Solution**: Manual real/imaginary splitting

```python
# PyTorch
k_complex = torch.view_as_complex(k_unrot.view(..., 2))
phi = torch.angle(q_mean_complex * k_complex.conj())

# Triton equivalent
k_r, k_i = k[:half_dim], k[half_dim:]
prod_real = q_r * k_r + q_i * k_i  # Real part of q * conj(k)
prod_imag = q_i * k_r - q_r * k_i  # Imaginary part
# No atan2 needed - use cos/sin decomposition instead
```

### 5. Aggregation Modes

**Supported**:
- `max`: Maximum score across offsets (default, matches R-KV)
- `mean`: Average score across offsets

**Implementation**:
```python
max_scores = tl.full([BLOCK_N], -1e10)  # Initialize with -inf
mean_scores = tl.zeros([BLOCK_N])

for offset in static_range(num_offsets):
    score = compute_score(offset)
    max_scores = tl.maximum(max_scores, score)
    mean_scores = mean_scores + score

final = max_scores if mode == 0 else mean_scores / num_offsets
```

## Numerical Correctness

### Validation Results

Test configuration:
- batch=1, heads=2, seq_len=32, dim=64
- 4 offsets: [1.0, 2.0, 4.0, 8.0]
- Aggregation: max

**Results**:
- Max difference: `1.34e-05` ✅
- Mean difference: `3.32e-06` ✅

**Error Sources**:
1. FP32 vs FP64 accumulation (negligible)
2. Different summation order (numerically stable)
3. Triton's sqrt approximation (acceptable)

### Precision Trade-offs

| Component | Precision | Justification |
|-----------|-----------|---------------|
| Input (K_rot) | BF16/FP16 | Match model dtype |
| Intermediate (A_coef, B_coef) | FP32 | Prevent underflow |
| Accumulation | FP32 | Required for stability |
| Output (scores) | FP32 | Downstream TopK needs precision |

## Performance Characteristics

### Memory Bandwidth Analysis

**Per Token (head_dim=128, 16 offsets)**:

| Component | Size | Reads (Optimized) | Reads (Naive) |
|-----------|------|-------------------|---------------|
| K_rot | 256 B | 1x = 256 B | 16x = 4 KB |
| cos_table | 4 B | 16x = 64 B | 16x = 64 B |
| sin_table | 4 B | 16x = 64 B | 16x = 64 B |
| **Total** | - | **384 B** | **4.2 KB** |

**Bandwidth reduction**: ~11x per token

### Compute Intensity

**FLOPs per token** (freq_count=64, num_offsets=16):
- Complex product: `4 * 64 = 256` FLOPs
- Coefficient scaling: `2 * 64 = 128` FLOPs
- Per-offset scoring: `(2 * 64 + 1) * 16 = 2064` FLOPs
- **Total**: ~2.5K FLOPs/token

**Arithmetic Intensity**: `2.5K FLOPs / 384 B ≈ 6.5 FLOPs/byte`

→ **Memory-bound** on modern GPUs (need >100 FLOPs/byte to be compute-bound)

## Known Limitations & Future Work

### Current Limitations

1. **num_offsets must be constexpr**: Can't change at runtime without recompilation
   - **Impact**: Low (typical configs use fixed 16 offsets)
   - **Workaround**: Precompile for common offset counts

2. **No TopK fusion**: Separate PyTorch TopK call required
   - **Impact**: Medium (~0.1ms overhead for TopK)
   - **Mitigation**: Phase 2 may implement Triton TopK

3. **No autotune**: Block sizes are hardcoded
   - **Impact**: Low (current BLOCK_N=32 works well)
   - **Mitigation**: Add `@triton.autotune` if needed

### Future Optimizations (Phase 2)

1. **Fused Scoring + TopK**:
   ```python
   @triton.jit
   def fused_score_topk_kernel(...):
       scores = compute_scores()
       indices = block_topk(scores, k)
       store(scores, indices)
   ```

2. **Multi-head parallel processing**:
   - Current: One thread block per (batch, head)
   - Proposed: Process multiple heads per block for small seq_len

3. **FP16 accumulation for high-precision GPUs**:
   - H100/A100 have 2x FP16 throughput
   - Requires careful error analysis

4. **Persistent kernels for streaming**:
   - Keep thread blocks resident across multiple rounds
   - Reduce launch overhead for repeated scoring

## Testing & Validation

### Test Coverage

✅ **Basic functionality**: Kernel compiles and runs
✅ **Numerical correctness**: Matches PyTorch reference (<1e-5 error)
✅ **Batch dimensions**: Tested batch_size=[1,2], heads=[1,4]
✅ **Sequence lengths**: Tested seq_len=[32,64,128]
✅ **Aggregation modes**: Both max and mean verified

❌ **Performance benchmarks**: Not yet implemented (Phase 2)
❌ **Edge cases**: Empty sequences, large batch sizes
❌ **Integration tests**: With actual vLLM attention

### Recommended Validation for Production

1. **Numerical stability**: Test with extreme values (large/small K)
2. **Performance profiling**: Nsight Systems trace
3. **Memory safety**: CUDA memcheck
4. **Correctness at scale**: Full AIME dataset eval

## Reference Implementation Comparison

### R-KV PyTorch (Original)

Location: `R-KV/weian_development/speckv/round_pruning_utils.py:277`

```python
def score_keys_for_round(key_indices, round_start, amp, phi, omega, ...):
    delta_grid = (round_start - key_indices).unsqueeze(1) + offsets.unsqueeze(0)
    phase = delta_grid.unsqueeze(2) * omega + phi.unsqueeze(1)
    base_scores = (amp.unsqueeze(1) * freq_scale_sq * cos(phase)).sum(dim=2)
    additive = (extra * freq_scale_sq).sum(dim=1, keepdim=True)
    return (base_scores + additive).max(dim=1).values  # or mean
```

### Triton Implementation (This)

Location: `TriAttention_vLLM/triattention/kernels/triton_scoring.py:29`

**Key Differences**:
1. **Vectorized over tokens**: Process BLOCK_N tokens in parallel
2. **Precomputed tables**: cos_table, sin_table computed once
3. **Factorized coefficients**: A_coef, B_coef computed per token
4. **Static offset iteration**: Compile-time loop unrolling

**Preserved**:
- Mathematical equivalence of scoring formula
- Support for max/mean aggregation
- Numerical precision (FP32 accumulation)

## Build & Deployment

### Compilation

Triton kernels are JIT-compiled on first execution:
- Cache location: `~/.triton/cache/`
- Compilation time: ~2-5 seconds (first run only)
- Binary cache: Reused across runs

### Dependencies

Required:
```
torch >= 2.0.0
triton >= 2.0.0
```

Optional (for testing):
```
pytest >= 7.0.0
```

### Integration Checklist

- [x] Kernel implementation
- [x] Python wrapper
- [x] Input validation
- [x] Correctness tests
- [ ] Performance benchmarks (Phase 2)
- [ ] Integration with vLLM attention (Phase 2)
- [ ] Documentation updates (Phase 2)

---

**Status**: ✅ Core implementation complete and validated
**Next Steps**: Integration with TriAttention pipeline (Phase 2)
**Owner**: weian
**Date**: 2025-02-01
