"""
Quick validation test for run_pruning_experiment.py

Tests that all core functions can be imported and basic functionality works.
"""

import sys
from pathlib import Path

# Add experiment directory to path
EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))

def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    try:
        from run_pruning_experiment import (
            setup_logging,
            train_model,
            auto_threshold_search,
            run_pruning_experiment
        )
        from model import Module1KeyPruningNetwork
        from train import load_trace_data
        import torch
        print("  All imports successful")
        return True
    except ImportError as e:
        print(f"  Import failed: {e}")
        return False


def test_output_directory():
    """Test that output directory can be created."""
    print("Testing output directory creation...")
    try:
        output_dir = EXP_DIR / 'output' / 'pruning_experiments'
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"  Output directory created: {output_dir}")
        return True
    except Exception as e:
        print(f"  Directory creation failed: {e}")
        return False


def test_model_creation():
    """Test that model can be created with different configs."""
    print("Testing model creation...")
    try:
        from model import Module1KeyPruningNetwork
        import torch

        # Test with MLP
        model1 = Module1KeyPruningNetwork(
            num_bins=64,
            num_freqs=64,
            num_kernels=3,
            mlp_hidden=64,
            anchor_positions=[1000, 10000, 100000]
        )
        params1 = model1.get_param_count()
        print(f"  Model with MLP created: {params1['total']:,} params")

        # Test with minimal MLP (avg pooling mode)
        model2 = Module1KeyPruningNetwork(
            num_bins=64,
            num_freqs=64,
            num_kernels=3,
            mlp_hidden=1,
            anchor_positions=[1000, 10000, 100000]
        )
        params2 = model2.get_param_count()
        print(f"  Model with avg pooling created: {params2['total']:,} params")

        return True
    except Exception as e:
        print(f"  Model creation failed: {e}")
        return False


def test_config_loading():
    """Test that config file can be loaded."""
    print("Testing config loading...")
    try:
        import yaml
        config_path = EXP_DIR / 'config.yaml'
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        print(f"  Config loaded: {config['experiment']['name']}")
        return True
    except Exception as e:
        print(f"  Config loading failed: {e}")
        return False


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("VALIDATION TESTS FOR run_pruning_experiment.py")
    print("=" * 70)

    tests = [
        test_imports,
        test_output_directory,
        test_model_creation,
        test_config_loading
    ]

    results = []
    for test_func in tests:
        print()
        result = test_func()
        results.append(result)

    print()
    print("=" * 70)
    print("TEST RESULTS")
    print("=" * 70)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("All validation tests passed!")
        return 0
    else:
        print("Some tests failed. Please check the output above.")
        return 1


if __name__ == '__main__':
    sys.exit(main())
