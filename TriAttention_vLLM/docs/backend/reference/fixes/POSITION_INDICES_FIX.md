# Position Indices Fix Summary

## Problem Identified

The original Triton kernel implementation had a critical flaw in handling out-of-order KV cache:

1. **Missing position tracking**: The kernel assumed KV cache tokens were in sequential order
2. **Incorrect phase calculation**: Used precomputed cos/sin tables based on offset alone
3. **No position correction**: Formula was `cos(t*omega + phi_rot)` where `t = round_start + offset`
4. **Fails on compressed cache**: After compression, KV cache is reordered, but phase was not adjusted

## Root Cause

The original implementation in `triton_scoring.py`:
```python
# Old approach (WRONG for out-of-order cache)
actual_offsets = round_start + offsets.float()
phase = actual_offsets[:, None] * omega[None, :]
cos_table = torch.cos(phase)
sin_table = torch.sin(phase)
```

This computed `cos((round_start + offset) * omega)`, but each key has its own original position `p`, so the correct phase should be `cos((round_start + offset - p) * omega)`.

## Solution Implemented

### 1. Added position_indices Parameter

**Kernel signature** (`triton_scoring.py`):
```python
@triton.jit
def speckv_scoring_kernel(
    K_rot_ptr,
    position_indices_ptr,  # NEW: [batch, seq_len] or [seq_len]
    q_mean_real_ptr,
    # ... other params
    omega_ptr,             # NEW: angular frequencies
    offsets_ptr,           # NEW: offset values (not precomputed tables)
    round_start,           # NEW: current query position base
    # ...
):
```

### 2. Dynamic Phase Calculation

**Inside kernel**:
```python
# Load original positions for each key
positions = tl.load(position_indices_ptr + n_offs, mask=n_mask)  # [BLOCK_N]

# Load omega and offset
omega = tl.load(omega_ptr + f_offs)  # [BLOCK_F]
offset = tl.load(offsets_ptr + off_idx)

# Compute position-corrected phase
delta_t = round_start + offset - positions  # [BLOCK_N]
phase = delta_t[:, None] * omega[None, :]   # [BLOCK_N, BLOCK_F]

# Compute trigonometry on-the-fly
cos_vals = tl.cos(phase)  # [BLOCK_N, BLOCK_F]
sin_vals = tl.sin(phase)  # [BLOCK_N, BLOCK_F]
```

### 3. Updated Python Wrapper

**`speckv_scoring()` signature**:
```python
def speckv_scoring(
    K_rot: torch.Tensor,
    position_indices: torch.Tensor,  # NEW
    q_mean_real: torch.Tensor,
    q_mean_imag: torch.Tensor,
    q_abs_mean: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    omega: torch.Tensor,             # NEW: replaced cos_table/sin_table
    offsets: torch.Tensor,           # NEW
    round_start: int,                # NEW
    aggregation: str = "max",
) -> torch.Tensor:
```

### 4. Updated Caller in scoring.py

**`compute_scores_triton()`**:
```python
# Prepare position_indices from cache_positions
position_indices = cache_positions.to(dtype=config.position_indices_dtype)

# Call kernel with new interface
scores = speckv_scoring(
    K_rot=K_rot,
    position_indices=position_indices,  # NEW
    # ... other params
    omega=omega_input,                  # NEW
    offsets=offsets_input,              # NEW
    round_start=round_start,            # NEW
    aggregation=config.score_aggregation,
)
```

## Changes Made

### Files Modified

1. **`triattention/kernels/triton_scoring.py`**:
   - Added `position_indices_ptr`, `omega_ptr`, `offsets_ptr`, `round_start` parameters
   - Removed precomputed `cos_table_ptr`, `sin_table_ptr`
   - Added position loading logic
   - Changed to on-the-fly trigonometry with position correction
   - Updated docstrings to reflect position-aware behavior

2. **`triattention/scoring.py`**:
   - Updated `compute_scores_triton()` to pass `position_indices`
   - Removed cos/sin table precomputation
   - Pass `omega`, `offsets`, `round_start` directly to kernel
   - Added docstring note about position indices

3. **`triattention/kernels/__init__.py`**:
   - Updated docstring to mention position_indices support
   - Removed backward compatibility alias `scoring_kernel`

### Files Deleted

4. **`triattention/kernels/scoring_kernel.py`**:
   - Removed placeholder implementation (was returning all ones)

### Files Added

5. **`test_position_indices.py`**:
   - Interface test to verify position_indices parameter acceptance
   - Tests both sequential and out-of-order positions

6. **`POSITION_INDICES_FIX.md`** (this file):
   - Documentation of the fix

## Correctness Verification

### Formula Alignment

Original R-KV formula (from design doc):
```
score = sum_over_freq(|Q_mean| * |K| * freq_scale^2 * cos((t-p)*omega + phi))
```

Where:
- `t = round_start + offset` (query position)
- `p = position_indices[token_idx]` (original key position)
- `phi = phase difference between Q_mean and K`

**Our implementation**:
```python
delta_t = round_start + offset - positions  # (t - p)
phase = delta_t[:, None] * omega[None, :]   # (t - p) * omega
# Then combined with phi_rot via trigonometric identity
base_scores = sum(A_coef * cos(phase) - B_coef * sin(phase))
```

Where `A_coef` and `B_coef` encode the phase difference `phi_rot`.

This correctly implements `cos((t-p)*omega + phi_rot)`.

## Performance Considerations

### Trade-offs

**Old approach**:
- Precomputed cos/sin tables: `O(num_offsets * freq_count)` storage
- Table lookup: Fast but assumes sequential positions

**New approach**:
- On-the-fly trigonometry: `O(1)` storage per offset
- Position-dependent computation: Slightly slower but correct for out-of-order cache
- Triton's `tl.cos()` and `tl.sin()` are hardware-accelerated (CUDA math library)

### Expected Impact

- **Memory**: Reduced (no cos/sin tables)
- **Compute**: Minimal increase (modern GPUs have fast transcendental functions)
- **Correctness**: Now handles out-of-order KV cache correctly

## Testing

### Quick Test

```bash
python test_position_indices.py
```

Expected output:
```
✓ Interface test passed: position_indices parameter accepted
  Output shape: (2, 4, 8)
  Score range: [min, max]
✓ Out-of-order positions test passed
  Shuffled positions: [3, 1, 7, 0, 5, 2, 6, 4]
  Score range: [min, max]

✓✓✓ All tests passed ✓✓✓
```

### Integration Test

To verify end-to-end correctness, run existing test suite:
```bash
cd TriAttention_vLLM/test
python test_scoring_correctness.py
```

This should now pass with out-of-order KV cache scenarios.

## Backward Compatibility

### Breaking Changes

The `speckv_scoring()` function signature changed:
- **Removed**: `cos_table`, `sin_table`
- **Added**: `position_indices`, `omega`, `offsets`, `round_start`

### Migration Guide

If external code was calling `speckv_scoring()` directly:

**Before**:
```python
scores = speckv_scoring(
    K_rot=K_rot,
    q_mean_real=q_mean_real,
    q_mean_imag=q_mean_imag,
    q_abs_mean=q_abs_mean,
    freq_scale_sq=freq_scale_sq,
    cos_table=cos_table,
    sin_table=sin_table,
    aggregation="max",
)
```

**After**:
```python
scores = speckv_scoring(
    K_rot=K_rot,
    position_indices=cache_positions,  # NEW
    q_mean_real=q_mean_real,
    q_mean_imag=q_mean_imag,
    q_abs_mean=q_abs_mean,
    freq_scale_sq=freq_scale_sq,
    omega=omega,                       # NEW
    offsets=offsets,                   # NEW
    round_start=round_start,           # NEW
    aggregation="max",
)
```

## Next Steps

1. **Test suite**: Run full test suite to ensure no regressions
2. **Benchmark**: Compare performance before/after fix
3. **Integration**: Test with vLLM PagedAttention integration
4. **Documentation**: Update README with position_indices requirement

## References

- Design document: Optimization 1 - RoPE phase correction
- R-KV paper: Section on frequency-domain scoring
- vLLM PagedAttention: KV cache management with position tracking
