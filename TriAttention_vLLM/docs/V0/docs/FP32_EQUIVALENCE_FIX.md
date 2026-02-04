# FP32 Triton-PyTorch Equivalence Bug Fix

**Date**: 2026-02-01
**Status**: ✅ FIXED

## Problem

The `test_basic_equivalence_fp32` tests were failing with large numerical discrepancies (~50 error) between Triton and PyTorch implementations.

## Root Cause

The Triton scoring kernel incorrectly computed the phase with an extra `phi_rot` term:

```python
# INCORRECT (old code)
phi_rot = tl.atan2(prod_imag, prod_real + 1e-8)
phase = t * omega[None, :] + phi_rot  # [BLOCK_N, BLOCK_F]
cos_vals = tl.cos(phase)
sin_vals = tl.sin(phase)
base_scores = tl.sum(A_coef * cos_vals - B_coef * sin_vals, axis=1)
```

This was mathematically incorrect because:
- `A_coef = freq_scale * prod_real` where `prod_real = Re(Q_mean * conj(K_rot))`
- `B_coef = freq_scale * prod_imag` where `prod_imag = Im(Q_mean * conj(K_rot))`

The complex product already encodes the phase information from both Q and K. If we write:
- `prod = |prod| * e^{i*phi_rot}` where `phi_rot = atan2(prod_imag, prod_real)`
- Then `prod_real = |prod| * cos(phi_rot)` and `prod_imag = |prod| * sin(phi_rot)`

The score formula:
```
score = sum_f [ A_coef * cos(t*omega) - B_coef * sin(t*omega) ]
```

is **already correct** without adding `phi_rot` to the phase!

Adding `phi_rot` to the phase would double-count the phase information, leading to incorrect results.

## The Fix

### Triton Kernel Changes

**File**: `triattention/kernels/triton_scoring.py`

1. **Removed `phi_rot` computation** (lines 262-270):
```python
# BEFORE:
phi_rot = tl.atan2(prod_imag, prod_real + 1e-8)  # REMOVED

# AFTER:
# (no phi_rot computation)
```

2. **Simplified phase calculation** (lines 286-305):
```python
# BEFORE:
phase = t * omega[None, :] + phi_rot  # [BLOCK_N, BLOCK_F]
cos_vals = tl.cos(phase)  # [BLOCK_N, BLOCK_F]
sin_vals = tl.sin(phase)  # [BLOCK_N, BLOCK_F]
base_scores = tl.sum(A_coef * cos_vals - B_coef * sin_vals, axis=1)

# AFTER:
phase = t * omega  # [BLOCK_F]
cos_vals = tl.cos(phase)  # [BLOCK_F]
sin_vals = tl.sin(phase)  # [BLOCK_F]
base_scores = tl.sum(A_coef * cos_vals[None, :] - B_coef * sin_vals[None, :], axis=1)
```

3. **Updated cached kernel** (lines 432-473):
```python
# BEFORE:
cos_phi = tl.cos(phi_rot)
sin_phi = tl.sin(phi_rot)
cos_vals = cos_t_omega[None, :] * cos_phi - sin_t_omega[None, :] * sin_phi
sin_vals = sin_t_omega[None, :] * cos_phi + cos_t_omega[None, :] * sin_phi

# AFTER:
# Use precomputed trig values directly (no phi_rot correction needed)
base_scores = tl.sum(A_coef * cos_t_omega[None, :] - B_coef * sin_t_omega[None, :], axis=1)
```

### Key Changes:
1. Removed `phi_rot = atan2(prod_imag, prod_real)` computation
2. Changed phase from `t * omega + phi_rot` to just `t * omega`
3. Changed cos/sin from `[BLOCK_N, BLOCK_F]` to `[BLOCK_F]` with broadcasting
4. Simplified cached kernel to use precomputed trig values directly

## Mathematical Explanation

The scoring formula is:
```
score = sum_f [ freq_scale[f]^2 * Re(Q_mean[f] * conj(K_rot[f])) * cos(t*omega[f])
               - freq_scale[f]^2 * Im(Q_mean[f] * conj(K_rot[f])) * sin(t*omega[f]) ]
      + extra_term
```

Where:
- `Q_mean[f] = |Q_mean[f]| * e^{i*theta_q[f]}`
- `K_rot[f] = |K_rot[f]| * e^{i*theta_k[f]}`

The complex product:
```
Q_mean * conj(K_rot) = |Q_mean| * |K_rot| * e^{i*(theta_q - theta_k)}
```

So:
```
Re(Q_mean * conj(K_rot)) = |Q_mean| * |K_rot| * cos(theta_q - theta_k)
Im(Q_mean * conj(K_rot)) = |Q_mean| * |K_rot| * sin(theta_q - theta_k)
```

The phase difference `(theta_q - theta_k)` is **already encoded** in `prod_real` and `prod_imag`. We don't need to extract it as `phi_rot` and add it again!

## Verification

All equivalence tests now pass:

```bash
$ pytest test/test_triton_pytorch_equivalence.py -xvs
13 passed, 3 skipped
```

Key test results:
- `test_basic_equivalence_fp32[mean]`: ✅ PASS (max error < 1e-4)
- `test_basic_equivalence_fp32[max]`: ✅ PASS (max error < 1e-4)
- All configuration tests: ✅ PASS
- Numerical stability tests: ✅ PASS
- Optimization equivalence tests: ✅ PASS (max error < 5e-4)

## Impact

This fix:
1. ✅ Corrects the mathematical implementation to match the reference
2. ✅ Simplifies the kernel (removed unnecessary `phi_rot` computation)
3. ✅ Slightly improves performance (fewer trig operations)
4. ✅ Maintains compatibility with all existing tests and configurations

## Related Documentation

- `docs/RKV_EQUIVALENCE_FIX.md`: Phase calculation derivation
- `test/test_triton_pytorch_equivalence.py`: Equivalence test suite
- `test/test_optimization_equivalence.py`: End-to-end validation
