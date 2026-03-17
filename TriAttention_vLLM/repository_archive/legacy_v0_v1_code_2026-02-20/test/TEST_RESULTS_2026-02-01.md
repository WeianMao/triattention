# Test Execution Report - February 1, 2026

## Test Environment

- **Environment**: trivllm conda environment
- **Python**: 3.10.19
- **Pytest**: 9.0.2
- **GPU**: Tesla T4 (compute capability 7.5)
- **CUDA Target**: sm_75

## Summary

### Overall Results
- **Total Tests Run**: 49
- **Passed**: 36 (73.5%)
- **Failed**: 13 (26.5%)

### Test Suite 1: test_scoring_kernel.py

**Status**: PASS

```
======================== 33 passed in 82.97s (0:01:22) =========================
```

**Results by Test Pattern**:
- `test_scoring_kernel_correctness[max-*]`: 16/16 PASSED
- `test_scoring_kernel_correctness[mean-*]`: 16/16 PASSED
- `test_basic_functionality`: 1/1 PASSED

**Conclusion**: Scoring kernel implementation is CORRECT and STABLE across all parameter combinations.

---

### Test Suite 2: test_triton_pytorch_equivalence.py

**Status**: PARTIAL FAILURE

```
======================== 13 failed, 3 passed in 10.16s =========================
```

#### Passed Tests (3)
1. `test_numerical_stability_zero_inputs` - PASSED
2. `test_reproducibility` - PASSED
3. `test_aggregation_mean_vs_max` - PASSED

#### Failed Tests (13)

##### Category 1: Equivalence Mismatches (FP32) - 2 failures
1. **test_basic_equivalence_fp32[mean]** - FAILED
   - Max abs error: 5.17e+01
   - Mean abs error: 1.21e+01
   - **Issue**: Large numerical discrepancies between Triton and PyTorch implementations

2. **test_basic_equivalence_fp32[max]** - FAILED
   - Max abs error: 5.07e+01
   - Mean abs error: 1.22e+01
   - **Issue**: Large numerical discrepancies between Triton and PyTorch implementations

##### Category 2: Hardware Incompatibility (BF16) - Multiple failures
Tests requiring BF16 dtype operations fail due to GPU hardware limitations:

3. **test_basic_equivalence_bf16[mean]** - FAILED
4. **test_basic_equivalence_bf16[max]** - FAILED
5. **test_different_batch_sizes** - FAILED (tries BF16)
6. **test_different_seq_lengths** - FAILED (tries BF16)
7. **test_different_num_heads** - FAILED (tries BF16)
8. **test_different_num_offsets** - FAILED (tries BF16)
9. **test_edge_case_single_token** - FAILED (tries BF16)
10. **test_edge_case_very_long_sequence** - FAILED (tries BF16)
11. **test_numerical_stability_large_values** - FAILED (tries BF16)
12. **test_comprehensive_configurations** - FAILED (tries BF16)
13. **test_dtype_promotion_consistency** - FAILED (tries BF16)

**Hardware Error Details**:
```
RuntimeError: Internal Triton PTX codegen error:
ptxas /tmp/tmpvXXXX.ptx, line XXX; error: Feature '.bf16' requires .target sm_80 or higher
```

The GPU is **Tesla T4 (compute capability 7.5)** which does NOT support BF16 operations.
BF16 requires **Ampere architecture or newer (sm_80+)** (e.g., A100, H100, RTX 3090).

---

## Error Analysis

### Issue 1: Triton-PyTorch Equivalence Mismatch (FP32)

**Status**: CRITICAL - Needs Investigation

**Affected Tests**:
- `test_basic_equivalence_fp32[mean]`
- `test_basic_equivalence_fp32[max]`

**Error Magnitude**:
- Maximum absolute error: ~50
- Mean absolute error: ~12
- Tolerance expected: rtol=0.0001, atol=0.0001

**Root Cause Possibilities**:
1. **Aggregation Logic Bug**: The mean/max computation differs between Triton and PyTorch
2. **Broadcasting Issue**: Shape handling or tensor operations in Triton kernel
3. **Offset Handling**: The `offsets` parameter may have indexing inconsistencies
4. **Numerical Precision**: Accumulation order differences between implementations

**Required Actions**:
- Debug `test_triton_pytorch_equivalence.py::test_basic_equivalence_fp32` with detailed value inspection
- Compare Triton kernel output with PyTorch reference implementation step-by-step
- Verify offset indexing logic in Triton kernel matches PyTorch

### Issue 2: Hardware Incompatibility (BF16)

**Status**: EXPECTED - Hardware Limitation

**Affected Tests**: 11 tests (all that use BF16 dtype)

**Root Cause**:
- GPU is Tesla T4 (compute capability sm_75)
- BF16 operations require sm_80 or higher
- Test suite attempts BF16 testing on incompatible hardware

**Solution Options**:
1. **Skip BF16 tests on sm_75 hardware** (recommended)
   - Modify test fixtures to detect GPU capability
   - Skip BF16 tests automatically on sm_<80
2. **Run tests on compatible GPU** (A100, H100, or RTX 3090 equivalent)
3. **Remove BF16 tests** (if not critical for deployment)

**Recommended Fix**:
```python
# In test_triton_pytorch_equivalence.py

import torch

def skip_if_unsupported_dtype(dtype):
    """Skip test if dtype not supported by current GPU."""
    if dtype == torch.bfloat16:
        # Check GPU compute capability
        cap = torch.cuda.get_device_capability(0)
        major, minor = cap
        compute_cap = major * 10 + minor
        if compute_cap < 80:  # BF16 needs sm_80+
            pytest.skip(f"BF16 not supported on sm_{compute_cap}")
```

---

## Key Findings

### SUCCESS
✓ **Scoring kernel implementation is numerically correct**
  - All 33 test_scoring_kernel.py tests pass
  - Validates correctness of core computation logic
  - Stable across different parameter combinations (batch size, seq length, heads, aggregation modes)

### FAILURE
✗ **Triton-PyTorch equivalence broken for FP32**
  - 2 critical failures in basic equivalence tests
  - Large numerical discrepancies (error ~50)
  - Likely bug in Triton kernel implementation vs PyTorch reference

✗ **BF16 tests fail due to hardware limitations**
  - 11 tests fail on Tesla T4 (sm_75)
  - BF16 requires sm_80+ hardware
  - Not a code bug, but a test environment issue

---

## Recommendations

### Priority 1: Fix FP32 Equivalence (CRITICAL)
1. **Investigate numerical differences** in Triton kernel
2. **Debug offset indexing** - verify correct tensor access patterns
3. **Compare accumulation logic** - check mean/max computation order
4. **Add detailed print statements** to both implementations for side-by-side comparison
5. **Check test input generation** - ensure test data is valid

**Estimated effort**: 2-4 hours debugging

### Priority 2: Handle Hardware Incompatibility (IMPORTANT)
1. Add GPU capability detection to test suite
2. Conditionally skip BF16 tests on sm_<80 hardware
3. Document minimum hardware requirements in test README

**Estimated effort**: 30 minutes

### Priority 3: Extend Test Coverage (OPTIONAL)
1. Add detailed equivalence tests with smaller tensors for easier debugging
2. Add dimension-specific tests (test one parameter at a time)
3. Add tolerance analysis to find acceptable error margins

**Estimated effort**: 1-2 hours

---

## Test Execution Details

### Test Run 1: test_scoring_kernel.py
```bash
Command: python -m pytest test_scoring_kernel.py -v --tb=short
Time: 82.97 seconds
Result: 33 passed
```

### Test Run 2: test_triton_pytorch_equivalence.py
```bash
Command: python -m pytest test_triton_pytorch_equivalence.py -v --tb=short
Time: 10.16 seconds
Result: 13 failed, 3 passed
```

---

## Next Steps

1. **Immediate**: Fix FP32 equivalence issues in Triton kernel
2. **Short-term**: Add hardware capability detection to tests
3. **Medium-term**: Run full test suite on compatible GPU (sm_80+) to verify BF16 tests pass
4. **Documentation**: Update INSTALLATION.md with GPU hardware requirements
