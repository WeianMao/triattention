# Quick Reference: Bug Fixes (2026-02-01)

## Critical Bug (MUST FIX)

**TrigTableCache Index Error**
- File: `triattention/kernels/triton_scoring.py`
- Line: 106
- Symptom: IndexError when `round_start < compress_interval`
- Fix: Added validation before index calculation
- Test: `pytest test/test_bug_fixes.py::TestTrigTableCacheFix -v`

## Medium Priority Bugs (SHOULD FIX)

**Invalid position_indices dtype**
- File: `triattention/config.py`
- Line: 157
- Symptom: Precision loss for long sequences with bfloat16
- Fix: Removed bfloat16 from valid dtypes
- Test: `pytest test/test_bug_fixes.py::TestConfigDtypeFix -v`

**Division by Zero in normalize_scores**
- File: `triattention/utils.py`
- Line: 193-206
- Symptom: NaN/Inf when all scores are constant
- Fix: Use `torch.where` to avoid division by zero
- Test: `pytest test/test_bug_fixes.py::TestNormalizeScoresFix -v`

**disable_mlr Not Implemented**
- Files: `triattention/scoring.py`, `triattention/kernels/triton_scoring.py`
- Symptom: Warning instead of error, MLR still used
- Fix: Automatic fallback to PyTorch when `disable_mlr=True`
- Test: Covered by existing tests

## Run All Tests

```bash
conda run -n trivllm python -m pytest test/test_bug_fixes.py -v
```

Expected: 11 passed

## Affected Functions

1. `TrigTableCache.get_trig_values()` - CRITICAL FIX
2. `TriAttentionConfig.__post_init__()` - dtype validation
3. `normalize_scores()` - division by zero
4. `compute_scores()` - disable_mlr fallback
5. `speckv_scoring()` - disable_mlr error handling
