#!/bin/bash

# TriAttention Test Runner
# Convenience script for running test suites

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR/.."

echo "========================================"
echo "TriAttention Test Suite Runner"
echo "========================================"
echo ""

function run_tests() {
    local test_name=$1
    local test_path=$2
    local extra_args=${3:-}

    echo "Running: $test_name"
    echo "----------------------------------------"
    if pytest "$test_path" -v $extra_args; then
        echo "✓ PASSED: $test_name"
    else
        echo "✗ FAILED: $test_name"
        return 1
    fi
    echo ""
}

case "${1:-all}" in
    all)
        echo "Running all tests..."
        pytest test/ -v --tb=short
        ;;

    scoring)
        run_tests "Scoring Correctness" "test/test_scoring_correctness.py"
        ;;

    topk)
        run_tests "TopK Selection" "test/test_topk_selection.py"
        ;;

    pruning)
        run_tests "Pruning Modes" "test/test_pruning_modes.py"
        ;;

    integration)
        run_tests "R-KV Integration" "test/test_integration.py"
        ;;

    fast)
        echo "Running fast tests (excluding integration)..."
        pytest test/ -v --ignore=test/test_integration.py --tb=short
        ;;

    benchmark)
        echo "Running benchmarks..."
        python test/benchmarks/bench_scoring.py --config small
        ;;

    coverage)
        echo "Running tests with coverage..."
        pytest test/ --cov=triattention --cov-report=term --cov-report=html
        echo ""
        echo "Coverage report generated in htmlcov/index.html"
        ;;

    help)
        echo "Usage: $0 [test_suite]"
        echo ""
        echo "Test suites:"
        echo "  all          - Run all tests (default)"
        echo "  scoring      - Run scoring correctness tests"
        echo "  topk         - Run TopK selection tests"
        echo "  pruning      - Run pruning mode tests"
        echo "  integration  - Run R-KV integration tests"
        echo "  fast         - Run all except integration tests"
        echo "  benchmark    - Run performance benchmarks"
        echo "  coverage     - Run tests with coverage report"
        echo "  help         - Show this message"
        ;;

    *)
        echo "Unknown test suite: $1"
        echo "Run '$0 help' for usage"
        exit 1
        ;;
esac
