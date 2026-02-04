# Triton-PyTorch Equivalence Implementation Guide

This guide documents the critical implementation patterns that ensure numerical equivalence between the Triton kernel and PyTorch reference implementation.

## Overview

The scoring kernel computes importance scores for KV cache tokens using frequency-domain analysis with RoPE phase correction. The implementation must match between:
- **Triton kernel**: `/triattention/kernels/triton_scoring.py` (GPU-accelerated)
- **PyTorch reference**: `/triattention/scoring.py` → `compute_pytorch_reference()` in test file (CPU/GPU)

## Critical Implementation Patterns

### 1. Position Delta Calculation

**CORRECT** ✓
```python
# Both implementations must use the same formula
delta_t = round_start + offset - positions

# Where:
# - round_start: current query position (scalar)
# - offset: scoring offset value (scalar or tensor element)
# - positions: original position of each key token (tensor)
```

**INCORRECT** ✗
```python
# Wrong sign
delta_t = positions - (round_start + offset)

# Missing offset
delta_t = round_start - positions

# Wrong reference point
delta_t = cache_index - positions  # Should use round_start, not cache index
```

### 2. Complex Number Handling

**K_rot Layout**: Interleaved format `[r0, i0, r1, i1, r2, i2, ...]`

**Triton (CORRECT)** ✓
```python
# Extract real and imaginary parts
k_r_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + (f_offs[None, :] * 2)
k_i_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + (f_offs[None, :] * 2 + 1)

k_r = tl.load(k_r_ptrs, mask=...)  # [BLOCK_N, BLOCK_F]
k_i = tl.load(k_i_ptrs, mask=...)  # [BLOCK_N, BLOCK_F]
```

**PyTorch (CORRECT)** ✓
```python
# Reshape and split
k_pairs = K_rot.view(batch, num_heads, seq_len, freq_count, 2)
k_r = k_pairs[..., 0]  # [batch, num_heads, seq_len, freq_count]
k_i = k_pairs[..., 1]
```

**INCORRECT** ✗
```python
# Wrong indexing
k_r = K_rot[..., 0::2]  # This assumes packed layout, not interleaved
k_i = K_rot[..., 1::2]
```

### 3. Coefficient Computation

**CORRECT** ✓
```python
# Step 1: Complex product Q_mean * conj(K_rot)
prod_real = q_r * k_r + q_i * k_i
prod_imag = q_i * k_r - q_r * k_i

# Step 2: Apply frequency scaling to coefficients
A_coef = freq_scale_sq * prod_real
B_coef = freq_scale_sq * prod_imag
```

**INCORRECT** ✗
```python
# Missing freq_scale_sq
A_coef = prod_real  # Wrong!
B_coef = prod_imag  # Wrong!

# Applying freq_scale_sq at wrong stage
base_scores = (prod_real * cos_vals - prod_imag * sin_vals).sum() * freq_scale_sq  # Wrong!
```

### 4. Additive Term (Position-Independent)

**CORRECT** ✓
```python
# Compute magnitude difference term
q_mean_abs = sqrt(q_r^2 + q_i^2 + 1e-8)
k_abs = sqrt(k_r^2 + k_i^2 + 1e-8)

extra_coef = (q_abs_mean - q_mean_abs) * k_abs * freq_scale_sq
extra_sum = extra_coef.sum(over_frequencies)  # Sum once per token
```

**INCORRECT** ✗
```python
# Using |Q_mean| instead of |Q_mean_complex|
extra_coef = (q_abs_mean - q_abs_mean) * k_abs  # Always zero! Wrong!

# Missing freq_scale_sq
extra_coef = (q_abs_mean - q_mean_abs) * k_abs  # Missing freq_scale_sq!

# Summing at wrong stage
extra_sum = extra_coef.sum()  # Should sum over frequencies, not all dimensions
```

### 5. Aggregation Order

**CORRECT** ✓
```python
scores_per_offset = []

for offset in offsets:
    # Compute phase-dependent term
    phase = delta_t * omega
    base_scores = (A_coef * cos(phase) - B_coef * sin(phase)).sum(over_frequencies)

    # Add position-independent term
    combined = base_scores + extra_sum  # Combine BEFORE aggregation

    scores_per_offset.append(combined)

# Aggregate over offsets
if aggregation == "max":
    final_scores = torch.stack(scores_per_offset).max(dim=0).values
else:  # mean
    final_scores = torch.stack(scores_per_offset).mean(dim=0)
```

**INCORRECT** ✗
```python
# Aggregating before adding extra_sum
base_scores_all = []
for offset in offsets:
    base_scores = (A_coef * cos(phase) - B_coef * sin(phase)).sum()
    base_scores_all.append(base_scores)

final_base = torch.stack(base_scores_all).max(dim=0).values
final_scores = final_base + extra_sum  # Wrong! Should aggregate combined scores
```

### 6. Trigonometric Expansion

**CORRECT** ✓
```python
# Compute cos and sin separately
cos_vals = torch.cos(phase)  # or tl.cos(phase)
sin_vals = torch.sin(phase)  # or tl.sin(phase)

# Apply trigonometric identity
base_scores = (A_coef * cos_vals - B_coef * sin_vals).sum(dim=-1)
```

**Why this works**:
```
Original formula:
  score = sum[amp * cos(delta * omega + phi)]

Trigonometric expansion:
  cos(delta * omega + phi) = cos(delta * omega) * cos(phi) - sin(delta * omega) * sin(phi)

Where:
  A_coef = amp * cos(phi) = freq_scale * (q_r * k_r + q_i * k_i)
  B_coef = amp * sin(phi) = freq_scale * (q_i * k_r - q_r * k_i)
```

**INCORRECT** ✗
```python
# Computing phase with precomputed phi (may have sign errors)
phi = atan2(q_i, q_r) - atan2(k_i, k_r)  # Risky! Can have sign issues
phase_total = delta * omega + phi
base_scores = amp * cos(phase_total).sum()  # Prone to numerical errors
```

## Shape Conventions

### Tensor Shapes

| Tensor | Shape | Notes |
|--------|-------|-------|
| `K_rot` | `[batch, num_heads, seq_len, head_dim]` | RoPE-rotated keys, interleaved format |
| `position_indices` | `[seq_len]` or `[batch, seq_len]` | Original positions of keys |
| `q_mean_real` | `[num_heads, freq_count]` | Real part of Q mean |
| `q_mean_imag` | `[num_heads, freq_count]` | Imaginary part of Q mean |
| `q_abs_mean` | `[num_heads, freq_count]` | `|Q|` mean (element-wise magnitude before averaging) |
| `freq_scale_sq` | `[num_heads, freq_count]` | Frequency scaling factors (squared) |
| `omega` | `[freq_count]` | Angular frequencies for RoPE |
| `offsets` | `[num_offsets]` | Scoring offset values |

### Broadcast Rules

**PyTorch**:
```python
# Expand Q statistics to match K dimensions
q_r_exp = q_mean_real.unsqueeze(0).unsqueeze(2)  # [num_heads, freq] -> [1, num_heads, 1, freq]
# Broadcasting: [1, H, 1, F] * [B, H, N, F] -> [B, H, N, F]
```

**Triton**:
```python
# Use explicit broadcasting with [None, :]
q_r[None, :]  # [BLOCK_F] -> [1, BLOCK_F]
k_r  # [BLOCK_N, BLOCK_F]
# Broadcasting: [1, F] * [N, F] -> [N, F]
```

## Numerical Precision

### Epsilon Values

**CORRECT** ✓
```python
# Use 1e-8 for magnitude calculations to avoid division by zero
q_mean_abs = sqrt(q_r^2 + q_i^2 + 1e-8)
k_abs = sqrt(k_r^2 + k_i^2 + 1e-8)
```

**INCORRECT** ✗
```python
# No epsilon - can cause NaN with zero inputs
q_mean_abs = sqrt(q_r^2 + q_i^2)  # NaN if both are 0!

# Too large epsilon - can bias results
q_mean_abs = sqrt(q_r^2 + q_i^2 + 1e-4)  # Changes magnitudes significantly
```

### Expected Precision

| Dtype | Max Error | Mean Error | Notes |
|-------|-----------|------------|-------|
| FP32 | < 1e-4 | < 1e-4 | Reference precision |
| FP16 | < 5e-2 | < 2e-2 | May fail on sm_75 |
| BF16 | < 1e-1 | < 5e-2 | Requires sm_80+ |

## Testing Strategy

### 1. Small-Scale Debug Test

Use controlled inputs to verify intermediate values:
```python
batch, num_heads, seq_len, freq_count = 1, 2, 4, 4
num_offsets = 2

# Print intermediate values:
# - k_r, k_i, k_abs
# - prod_real, prod_imag
# - A_coef, B_coef
# - extra_coef, extra_sum
# - phase, cos_vals, sin_vals
# - base_scores, combined
```

### 2. Comprehensive Configuration Tests

Test across parameter ranges:
- Batch sizes: [1, 2, 4, 8]
- Sequence lengths: [16, 32, 64, 128, 256, 1024]
- Head counts: [1, 4, 8, 16, 32]
- Offset counts: [1, 2, 4, 8, 16, 32]
- Aggregation modes: ["mean", "max"]

### 3. Edge Cases

- Single token sequences (seq_len=1)
- Very long sequences (seq_len>512)
- Zero inputs (all tensors initialized to 0)
- Large values (scaled by 100x)
- Uniform scores (all equal)
- Negative scores

## Common Pitfalls

### 1. Position Calculation Errors

**Symptom**: Large numerical errors (magnitude 10-100)

**Causes**:
- Wrong sign in `delta_t` calculation
- Using cache index instead of position_indices
- Missing offset in phase calculation

**Debug**: Print `delta_t` values and verify against expected positions

### 2. Aggregation Order Bugs

**Symptom**: Different results for mean vs max aggregation

**Causes**:
- Aggregating base_scores before adding extra_sum
- Applying extra_sum per-offset instead of once

**Debug**: Print scores before and after extra_sum addition

### 3. Broadcasting Mismatches

**Symptom**: Shape errors or unexpected broadcasting behavior

**Causes**:
- Missing unsqueeze/expand in PyTorch
- Wrong None placement in Triton

**Debug**: Print tensor shapes at each step

### 4. Complex Number Indexing

**Symptom**: Large errors in phase-dependent terms

**Causes**:
- Using packed layout instead of interleaved
- Swapping real/imaginary indices
- Incorrect stride calculation in Triton

**Debug**: Print k_r, k_i values and compare with K_rot raw values

## Verification Checklist

Before committing changes to the scoring kernel:

- [ ] Run `test_triton_pytorch_equivalence.py::test_basic_equivalence_fp32`
- [ ] Run `test_different_batch_sizes` (tests [1, 2, 4, 8])
- [ ] Run `test_different_seq_lengths` (tests [16, 32, 64, 128, 256])
- [ ] Run `test_different_num_offsets` (tests [1, 2, 4, 8, 16, 32])
- [ ] Run `test_numerical_stability_zero_inputs`
- [ ] Run `test_numerical_stability_large_values`
- [ ] Run debug script with small inputs and verify intermediate values
- [ ] Check max absolute error < 1e-4 for FP32
- [ ] Check mean absolute error < 1e-4 for FP32
- [ ] Verify both "mean" and "max" aggregation pass

## References

- **Algorithm Spec**: R-KV paper formula (Section 3.2)
- **Triton Kernel**: `TriAttention_vLLM/triattention/kernels/triton_scoring.py`
- **PyTorch Reference**: `TriAttention_vLLM/test/test_triton_pytorch_equivalence.py::compute_pytorch_reference()`
- **Test Suite**: `TriAttention_vLLM/test/test_triton_pytorch_equivalence.py`
- **Debug Script**: `TriAttention_vLLM/test/debug_scoring_equivalence.py`

## Changelog

- **2026-02-01**: Fixed position delta calculation bug (magnitude ~50 error → < 1e-6)
- **2026-02-01**: All equivalence tests passing (13/16, 3 skipped for BF16)
- **2026-02-01**: Created comprehensive test suite and debug utilities
