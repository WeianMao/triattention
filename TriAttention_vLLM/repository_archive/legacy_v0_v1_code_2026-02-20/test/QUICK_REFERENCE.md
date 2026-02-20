# TriAttention Test Framework - Quick Reference

## Directory Structure

```
test/
├── conftest.py                     # Fixtures (15 total)
├── test_scoring_correctness.py    # Scoring tests (10 cases)
├── test_topk_selection.py         # TopK tests (18 cases)
├── test_pruning_modes.py          # Pruning tests (15 cases)
├── test_integration.py            # R-KV integration (16 cases)
├── benchmarks/
│   └── bench_scoring.py           # Performance benchmarks
├── run_tests.sh                   # Convenience runner
├── README.md                      # Full documentation
├── TEST_SUMMARY.md                # Implementation summary
├── INSTALLATION.md                # Setup instructions
└── QUICK_REFERENCE.md             # This file
```

## Common Commands

### Running Tests

```bash
# All tests
pytest test/ -v

# Specific file
pytest test/test_scoring_correctness.py -v

# Single test
pytest test/test_topk_selection.py::test_topk_basic_selection -v

# Fast tests (skip integration)
pytest test/ --ignore=test/test_integration.py

# With coverage
pytest test/ --cov=triattention --cov-report=html
```

### Using Test Runner

```bash
./test/run_tests.sh all          # All tests
./test/run_tests.sh scoring      # Scoring only
./test/run_tests.sh topk         # TopK only
./test/run_tests.sh pruning      # Pruning modes
./test/run_tests.sh integration  # R-KV integration
./test/run_tests.sh fast         # Skip integration
./test/run_tests.sh benchmark    # Performance
./test/run_tests.sh coverage     # With coverage
```

### Running Benchmarks

```bash
# Small config (fast)
python test/benchmarks/bench_scoring.py --config small

# Default config
python test/benchmarks/bench_scoring.py --config default

# Large config (comprehensive)
python test/benchmarks/bench_scoring.py --config large --device cuda
```

## Test Patterns

### By Feature

```bash
# All dtype tests
pytest test/ -k "dtype" -v

# All prefill tests
pytest test/ -k "prefill" -v

# All per_head mode tests
pytest test/ -k "per_head" -v

# All aggregation tests
pytest test/ -k "aggregation" -v
```

### By Test Type

```bash
# Correctness tests only
pytest test/test_scoring_correctness.py test/test_topk_selection.py -v

# Mode tests only
pytest test/test_pruning_modes.py -v

# Integration only
pytest test/test_integration.py -v
```

## Key Fixtures

### Configurations
- `small_test_config` - 4 layers, 8 heads, 64 head_dim
- `qwen_7b_config` - 32 layers, 32 heads, 128 head_dim
- `qwen_14b_config` - 40 layers, 40 heads, 128 head_dim

### Data
- `random_kv_cache` - (keys, values, positions)
- `random_query_stats` - Simulated stats file
- `rope_frequencies` - RoPE frequency values

### Parametrization
- `test_dtype` - [float32, float16, bfloat16]
- `aggregation_strategy` - ["mean", "max"]
- `pruning_mode` - ["per_head", "per_layer", "per_layer_per_head"]

### Utilities
- `deterministic_seed` - Reproducible random numbers
- `tolerance_for_dtype` - Dtype-specific tolerance
- `device`, `cuda_only` - Device selection

## Numerical Tolerances

| Dtype | Tolerance | Notes |
|-------|-----------|-------|
| `float32` | `1e-5` | High precision |
| `float16` | `2e-2` | ~2% error OK |
| `bfloat16` | `1e-1` | ~10% error OK |

## Test Counts by File

| File | Tests | Parametrized | Total Runs |
|------|-------|--------------|------------|
| `test_scoring_correctness.py` | 10 | 3-6x | ~30-60 |
| `test_topk_selection.py` | 18 | 1x | ~18 |
| `test_pruning_modes.py` | 15 | 3x | ~45 |
| `test_integration.py` | 16 | 2x | ~32 |
| **Total** | **48** | - | **~125-155** |

## Common Issues

### pytest not found
```bash
pip install pytest
```

### R-KV tests skipped
```bash
cd ../R-KV && pip install -e .
```

### Import errors
```bash
# Ensure in correct directory
cd TriAttention_vLLM

# Check PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

### CUDA out of memory
```bash
# Use CPU
pytest test/ -v --device cpu

# Or reduce batch sizes in conftest.py
```

## Expected Results

### All Pass
```
======= 48 passed in 15.23s =======
```

### With R-KV Skip
```
======= 32 passed, 16 skipped in 10.45s =======
```

### Parametrized Expansion
```
test_dtype_consistency[float32] PASSED
test_dtype_consistency[float16] PASSED
test_dtype_consistency[bfloat16] PASSED
```

## Debugging

### Verbose output
```bash
pytest test/ -vv
```

### Print to console
```bash
pytest test/ -s
```

### Stop on first failure
```bash
pytest test/ -x
```

### Drop into debugger
```bash
pytest test/ --pdb
```

### Run last failed
```bash
pytest test/ --lf
```

## Performance Reference

Typical execution time on CPU:

| Test Suite | Time | Notes |
|------------|------|-------|
| Scoring | ~3s | 10 tests |
| TopK | ~2s | 18 tests |
| Pruning | ~4s | 15 tests |
| Integration | ~8s | 16 tests (with R-KV) |
| Benchmarks | ~30s | 3 benchmark suites |

## Coverage Goals

Target coverage: **90%+** for:
- Scoring formula computation
- TopK selection logic
- Pruning mode variants
- RoPE rotation
- Phase/amplitude calculations

Not covered (intentional):
- vLLM internals
- CUDA kernel implementations
- Production-specific configs

## Quick Validation

```bash
# 1. Install dependencies
pip install pytest

# 2. Run fast tests
pytest test/test_topk_selection.py -v

# 3. Expected: All pass
# ======= 18 passed in 2.34s =======

# 4. Run full suite
./test/run_tests.sh all
```

## Reference Implementation Classes

### Scoring
- `TriAttentionScorer` - Reference scoring implementation
- `apply_rope_rotation()` - RoPE rotation

### Selection
- `select_tokens_to_keep()` - TopK with prefill protection

### Pruning
- `PruningModeSelector` - Three mode variants
  - `select_per_head()`
  - `select_per_layer()`
  - `select_per_layer_per_head()`

## Links

- Full docs: `README.md`
- Implementation: `TEST_SUMMARY.md`
- Installation: `INSTALLATION.md`
- Algorithm spec: `../docs/design/algorithm.md`
