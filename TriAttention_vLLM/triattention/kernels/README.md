# Triton Scoring Kernel Implementation

## Overview

This directory contains the Triton implementation of the frequency-domain scoring kernel for KV cache compression, based on the R-KV algorithm.

## Files

- `triton_scoring.py`: Main Triton kernel implementation
  - `speckv_scoring_kernel`: Low-level Triton JIT kernel
  - `speckv_scoring`: Python wrapper function
- `scoring_kernel.py`: Placeholder (to be replaced/integrated in future)
- `__init__.py`: Package exports

## Scoring Formula

The kernel implements the optimized scoring formula:

```
score = base_scores + additive

base_scores = sum_over_freq(amp * freq_scale^2 * cos(t*omega + phi_rot))
additive = sum_over_freq((|q_abs_mean| - |q_mean_complex|) * |k_rot| * freq_scale^2)
```

Using trigonometric identity for efficiency:
```
cos(t*omega + phi_rot) = cos(t*omega)*cos(phi_rot) - sin(t*omega)*sin(phi_rot)
```

## Key Optimizations

### 1. Avoid RoPE Inversion
Instead of inverting RoPE on K_rot to get original K, we use K_rot directly with phase correction:
- Saves complex multiplication per key
- Phase correction: φ = φ_rot + p·ω is implicit in the formula

### 2. Single K Read per Token
Load K_rot once from global memory, then iterate over all offsets in registers:
- Reduces memory bandwidth by ~16x (for 16 offsets)
- Critical for performance on memory-bound workloads

### 3. Shared Trigonometric Tables
Precompute cos(t*omega) and sin(t*omega) for all offsets:
- Tables shared across all tokens in a batch
- Eliminates redundant sin/cos computation
- Very small memory footprint (~4KB for typical configs)

## Usage

```python
from triattention.kernels import speckv_scoring

# Input preparation
K_rot = ...  # [batch, num_heads, seq_len, head_dim]
q_mean_real = ...  # [num_heads, freq_count]
q_mean_imag = ...  # [num_heads, freq_count]
q_abs_mean = ...  # [num_heads, freq_count]
freq_scale_sq = ...  # [num_heads, freq_count]

# Build trigonometric tables
round_start = 1000
offsets = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, ...])
omega = ...  # Inverse frequencies [freq_count]

cos_table = torch.zeros(num_offsets, freq_count)
sin_table = torch.zeros(num_offsets, freq_count)
for i, offset in enumerate(offsets):
    t = round_start + offset
    cos_table[i] = torch.cos(t * omega)
    sin_table[i] = torch.sin(t * omega)

# Run kernel
scores = speckv_scoring(
    K_rot,
    q_mean_real,
    q_mean_imag,
    q_abs_mean,
    freq_scale_sq,
    cos_table,
    sin_table,
    aggregation="max"  # or "mean"
)
# Output: [batch, num_heads, seq_len]
```

## Performance Characteristics

### Block Configuration
- `BLOCK_N = 32`: Tokens processed per thread block
- `BLOCK_F = next_power_of_2(freq_count)`: Frequency dimension blocking

### Grid Layout
- Grid dimensions: `(batch_size * num_heads, cdiv(seq_len, BLOCK_N))`
- Each CUDA block processes BLOCK_N tokens for one (batch, head) pair

### Memory Access Pattern
1. Load Q statistics once per head (shared across all tokens)
2. Load K_rot in coalesced manner: `[BLOCK_N, freq_count]`
3. Load cos/sin tables sequentially for each offset
4. Store results in coalesced manner

### Expected Performance
For typical configs (seq_len=8K, budget=256, 16 offsets):
- Scoring overhead: ~0.3-0.4ms
- Memory bandwidth: ~2MB (vs ~32MB without optimization #2)
- Compute: Mostly memory-bound, FP32 accumulation

## Testing

Basic correctness test:
```bash
cd TriAttention_vLLM
python test/test_kernel_standalone.py
```

Full test suite (requires pytest):
```bash
pytest test/test_scoring_kernel.py -v
```

## Implementation Notes

### Complex Number Handling
Since Triton doesn't have native complex types, we split into real/imaginary components:
- K_rot uses front/back half pairing: `[real_0..real_n, imag_0..imag_n]`
- Complex product computed manually: `(a+bi)(c-di) = (ac+bd) + (bc-ad)i`

### Static Range for Offsets
`num_offsets` must be `tl.constexpr` to use `tl.static_range`, which enables:
- Compile-time loop unrolling
- Better instruction scheduling
- Reduced register pressure

### Aggregation Modes
- `max`: Take maximum score across all offsets (default for R-KV)
- `mean`: Average score across all offsets (optional)

## Future Optimizations (Phase 2)

Potential improvements if performance targets not met:
1. **Autotune**: Add `@triton.autotune` for block size selection
2. **Fused TopK**: Integrate TopK selection into scoring kernel
3. **Multi-stage pipeline**: Overlap scoring with TopK computation
4. **FP16 accumulation**: Use lower precision for intermediate results

## Compatibility

- Requires: Triton >= 2.0.0
- Tested on: CUDA 11.8+, A100/H100 GPUs
- Python: 3.9+
- PyTorch: 2.0+

## Reference

Based on R-KV algorithm from:
- `R-KV/weian_development/speckv/round_pruning_utils.py`
- Design doc: `TriAttention_vLLM/docs/design/optimization.md`
- Technical notes: `TriAttention_vLLM/docs/phases/phase1_triton_implementation/TECHNICAL_NOTES.md`
