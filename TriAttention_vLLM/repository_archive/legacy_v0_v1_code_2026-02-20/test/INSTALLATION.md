# Test Framework Installation

## Prerequisites

Python 3.9+ with PyTorch installed.

## Install Testing Dependencies

```bash
cd TriAttention_vLLM

# Install test dependencies
pip install pytest pytest-cov

# Optional: Install R-KV for integration tests
cd ../R-KV
pip install -e .
cd ../TriAttention_vLLM
```

## Verify Installation

```bash
# Check pytest installed
pytest --version

# Check test discovery
pytest test/ --collect-only

# Expected output: "collected 48 items" (or more with parametrized tests)
```

## Quick Validation

Run a simple test to verify setup:

```bash
# Run one test file
pytest test/test_topk_selection.py -v

# Expected: All tests pass
```

## Troubleshooting

### Import Errors

If you see `ModuleNotFoundError: No module named 'pytest'`:
```bash
pip install pytest
```

### R-KV Tests Skipped

If integration tests show `SKIPPED (R-KV not available)`:
```bash
# Install R-KV
cd ../R-KV
pip install -e .
```

### CUDA Tests Fail

If CUDA tests fail:
- Tests will automatically fall back to CPU
- Use `--ignore=test/test_integration.py` to skip GPU-specific tests

## Expected Test Output

### Successful Run
```
test/test_scoring_correctness.py::test_single_position_scoring_fp32 PASSED
test/test_scoring_correctness.py::test_multi_position_aggregation[mean] PASSED
test/test_topk_selection.py::test_topk_basic_selection PASSED
...

======= 48 passed in 12.34s =======
```

### With Skipped Tests (R-KV not installed)
```
test/test_integration.py::TestRKVIntegration::test_rkv_full_pipeline SKIPPED (R-KV not available)
...

======= 32 passed, 16 skipped in 8.45s =======
```

## Running in CI/CD

### GitHub Actions Example

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2

      - name: Set up Python
        uses: actions/setup-python@v2
        with:
          python-version: '3.10'

      - name: Install dependencies
        run: |
          pip install torch pytest pytest-cov
          pip install -e .

      - name: Run tests
        run: |
          pytest test/ -v --cov=triattention --cov-report=xml

      - name: Upload coverage
        uses: codecov/codecov-action@v2
        with:
          file: ./coverage.xml
```

## Development Workflow

1. **Make changes** to TriAttention code
2. **Run fast tests**: `./test/run_tests.sh fast`
3. **Fix failures** if any
4. **Run full suite**: `./test/run_tests.sh all`
5. **Check coverage**: `./test/run_tests.sh coverage`
6. **Commit** if all tests pass

## Performance Baseline

On typical hardware:
- **Fast tests**: ~5-10 seconds
- **Full suite**: ~15-30 seconds
- **With R-KV**: ~20-40 seconds
- **Benchmarks**: ~30-60 seconds

Faster with:
- CUDA GPU available
- Parallel execution: `pytest test/ -n auto` (requires pytest-xdist)
