# TriAttention Test Suite

Comprehensive testing framework for TriAttention vLLM implementation.

## Directory Structure

```
test/
├── __init__.py                      # Test package
├── conftest.py                      # Pytest fixtures
├── test_scoring_correctness.py     # Scoring formula validation
├── test_topk_selection.py          # TopK selection logic tests
├── test_pruning_modes.py           # Pruning mode variants tests
├── test_integration.py             # R-KV integration tests
├── benchmarks/
│   ├── __init__.py
│   └── bench_scoring.py            # Performance benchmarks
└── README.md                        # This file
```

## Test Coverage

### 1. Scoring Correctness (`test_scoring_correctness.py`)

Tests the mathematical correctness of TriAttention scoring formula:

- **Single position scoring**: Validates scoring at one future position
- **Multi-position scoring**: Tests scoring across multiple positions with mean/max aggregation
- **Dtype consistency**: Validates FP32, FP16, BF16 precision
- **Phase computation**: Tests phase difference calculation (phi = arg(Q * K_conj))
- **RoPE rotation**: Validates rotation correctness and magnitude preservation
- **Position-independent term**: Tests extra coefficient handling
- **Numerical stability**: Edge cases and symmetry properties

**Key Test**:
```python
def test_single_position_scoring_fp32():
    # Validates scoring formula:
    # score = sum_f [A_f * s_f^2 * cos((t-p)*omega + phi)] + sum_f [E_f * s_f^2]
```

### 2. TopK Selection (`test_topk_selection.py`)

Tests token selection logic:

- **Basic selection**: Top-k without prefill protection
- **Prefill protection**: Preserving prefill tokens from pruning
- **Budget enforcement**: Correct token count selection
- **Score ranking**: Highest scores selected
- **Edge cases**: Empty decode, budget exceeds total, tied scores
- **Boundary conditions**: prefill_len = 0, prefill > budget

**Key Test**:
```python
def test_topk_with_prefill_protection():
    # Validates:
    # 1. All prefill tokens kept
    # 2. Top-(budget-prefill_len) decode tokens kept
    # 3. Exactly budget tokens total
```

### 3. Pruning Modes (`test_pruning_modes.py`)

Tests three pruning granularity modes:

- **per_head**: Each KV head selects tokens globally across all layers
- **per_layer**: All heads in same layer share token selection
- **per_layer_per_head**: Each (layer, head) pair independently selects

**Key Test**:
```python
def test_per_layer_per_head_independent():
    # Validates independent selection per (layer, head)
    # Different heads can select different tokens
```

### 4. Integration Tests (`test_integration.py`)

Validates integration with R-KV reference implementation:

- **Attention score computation**: Matches R-KV `compute_attention_scores`
- **Similarity computation**: Matches R-KV `cal_similarity`
- **Scoring formula**: Validates R-KV's final scoring formula
- **TopK selection**: Compares index selection
- **Full pipeline**: End-to-end R-KV compression
- **Dtype consistency**: FP32 and BF16 comparison

**Key Test**:
```python
def test_rkv_full_pipeline_fp32():
    # Runs full R-KV compression pipeline
    # Validates output shapes and finite values
```

### 5. Benchmarks (`benchmarks/bench_scoring.py`)

Performance benchmarks for scoring operations:

- **Single position scoring**: Latency and throughput
- **Multi-position scoring**: Scaling with position count
- **RoPE rotation**: Rotation overhead
- **Dtype comparison**: FP32 vs BF16 performance

## Running Tests

### Run All Tests
```bash
cd TriAttention_vLLM
pytest test/ -v
```

### Run Specific Test File
```bash
pytest test/test_scoring_correctness.py -v
pytest test/test_topk_selection.py -v
pytest test/test_pruning_modes.py -v
pytest test/test_integration.py -v
```

### Run Tests by Pattern
```bash
# Run all dtype tests
pytest test/ -k "dtype" -v

# Run all prefill tests
pytest test/ -k "prefill" -v

# Run all per_head mode tests
pytest test/ -k "per_head" -v
```

### Run with Coverage
```bash
pytest test/ --cov=triattention --cov-report=html
```

### Run Benchmarks
```bash
# Default config
python test/benchmarks/bench_scoring.py --config default

# Small config (fast)
python test/benchmarks/bench_scoring.py --config small

# Large config (comprehensive)
python test/benchmarks/bench_scoring.py --config large --device cuda
```

## Fixtures (from `conftest.py`)

### Model Configurations
- `qwen_7b_config`: Qwen-7B parameters
- `qwen_14b_config`: Qwen-14B parameters
- `small_test_config`: Reduced config for fast testing

### Data Fixtures
- `rope_frequencies`: RoPE frequency values
- `random_kv_cache`: Random KV cache tensors
- `random_query_stats`: Simulated stats file data

### Parametrized Fixtures
- `test_dtype`: Parametrizes over [float32, float16, bfloat16]
- `aggregation_strategy`: Parametrizes over ["mean", "max"]
- `pruning_mode`: Parametrizes over ["per_head", "per_layer", "per_layer_per_head"]

### Utilities
- `deterministic_seed`: Sets random seed for reproducibility
- `tolerance_for_dtype`: Returns appropriate tolerance for dtype
- `device`, `cuda_only`: Device selection and CUDA availability

## Example: Adding New Tests

### 1. Add Test to Existing File

```python
# In test_scoring_correctness.py
def test_new_scoring_feature(random_query_stats, rope_frequencies, deterministic_seed):
    # Setup
    scorer = TriAttentionScorer(...)

    # Execute
    result = scorer.new_feature(...)

    # Validate
    assert result.shape == expected_shape
    assert torch.all(torch.isfinite(result))
```

### 2. Create New Test File

```python
# test_new_feature.py
import pytest
import torch

def test_feature_basic(small_test_config, deterministic_seed):
    # Test basic functionality
    pass

def test_feature_edge_cases(test_dtype, tolerance_for_dtype):
    # Test edge cases across dtypes
    pass
```

### 3. Add Benchmark

```python
# In benchmarks/bench_new_feature.py
class NewFeatureBenchmark:
    def benchmark_operation(self, ...):
        # Warmup
        for _ in range(10):
            operation()

        # Time
        timings = []
        for _ in range(num_trials):
            start = time.perf_counter()
            operation()
            torch.cuda.synchronize()
            end = time.perf_counter()
            timings.append(end - start)

        return {"mean_ms": mean(timings) * 1000}
```

## Tolerance Guidelines

### Numerical Tolerances by Dtype

| Dtype | Absolute Tolerance | Use Case |
|-------|-------------------|----------|
| `float32` | `1e-5` | High precision validation |
| `float16` | `2e-2` | ~2% error acceptable |
| `bfloat16` | `1e-1` | ~10% error acceptable (7 mantissa bits) |

### When to Use Which Tolerance

- **Mathematical equivalence**: Use strictest (`1e-5` for FP32)
- **Cross-dtype comparison**: Use dtype-specific tolerance
- **Integration tests**: Relaxed (`5e-4` for accumulated errors)

## R-KV Integration

### Running with R-KV Available

Tests marked with `@pytest.mark.skipif(not RKV_AVAILABLE, ...)` require R-KV installed:

```bash
# Install R-KV
cd R-KV
pip install -e .

# Run integration tests
pytest test/test_integration.py -v
```

### Without R-KV

Integration tests will be skipped:
```
test_integration.py::TestRKVIntegration::test_rkv_full_pipeline SKIPPED (R-KV not available)
```

## Test Philosophy

### What We Test

1. **Mathematical correctness**: Scoring formulas match specification
2. **Shape consistency**: Tensors have expected dimensions
3. **Numerical stability**: No NaN/Inf in outputs
4. **Edge cases**: Empty inputs, budget extremes, tied scores
5. **Dtype compatibility**: Works across FP32/FP16/BF16
6. **Integration**: Matches R-KV reference where applicable

### What We Don't Test

- **vLLM internals**: Covered by vLLM's own tests
- **CUDA kernel correctness**: Validated separately in kernel tests
- **Production config**: Use separate end-to-end tests

## CI/CD Integration

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - name: Install dependencies
        run: |
          pip install -r requirements.txt
          pip install pytest pytest-cov
      - name: Run tests
        run: pytest test/ -v --cov=triattention
```

## Debugging Failed Tests

### 1. Run with Verbose Output
```bash
pytest test/test_scoring_correctness.py::test_single_position_scoring_fp32 -vv
```

### 2. Drop into Debugger on Failure
```bash
pytest test/ --pdb
```

### 3. Print Intermediate Values
```python
def test_debug():
    result = compute()
    print(f"Result: {result}")  # Will show on failure
    assert condition
```

### 4. Check Tensor Contents
```python
# In test
print(f"Shape: {tensor.shape}, Dtype: {tensor.dtype}")
print(f"Min: {tensor.min()}, Max: {tensor.max()}, Mean: {tensor.mean()}")
print(f"Has NaN: {torch.isnan(tensor).any()}")
```

## Performance Notes

### Test Execution Time

- **Fast tests** (~0.1s each): scoring, topk, pruning modes
- **Moderate tests** (~1s each): dtype parametrized tests
- **Slow tests** (~5s each): integration tests with R-KV
- **Benchmarks** (~30s): Full suite with warmup

### Speeding Up Tests

```bash
# Run in parallel (requires pytest-xdist)
pytest test/ -n auto

# Run only fast tests
pytest test/ -m "not slow"

# Skip integration tests
pytest test/ --ignore=test/test_integration.py
```

## Contributing

When adding new features:

1. Add tests to appropriate file (or create new file)
2. Use existing fixtures from `conftest.py`
3. Follow naming convention: `test_<feature>_<aspect>`
4. Add docstrings explaining what is tested
5. Validate both correctness and edge cases
6. Consider adding benchmark if performance-critical

## References

- **Algorithm design**: `../docs/design/algorithm.md`
- **Optimization details**: `../docs/design/optimization.md`
- **R-KV reference**: `../../R-KV/rkv/compression/r1_kv.py`
- **vLLM docs**: https://docs.vllm.ai/
