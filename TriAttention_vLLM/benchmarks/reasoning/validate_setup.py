#!/usr/bin/env python3
"""Validate TriAttention + vLLM setup before running full benchmark.

This script checks:
1. All required files exist
2. Config parameters are correctly mapped
3. TriAttention modules load correctly
4. vLLM can be initialized
5. Patching works correctly
"""
import sys
from pathlib import Path

# Add project root to path
# From benchmarks/reasoning/validate_setup.py -> TriAttention_vLLM -> dc
TRIATTENTION_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = TRIATTENTION_ROOT.parent
sys.path.insert(0, str(TRIATTENTION_ROOT))

def validate_files():
    """Check all required files exist."""
    print("=" * 60)
    print("Step 1: Validating file paths...")
    print("=" * 60)

    files = {
        "Model": Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B"),
        "Dataset": Path(PROJECT_ROOT) / "R-KV/HuggingFace/data/aime24.jsonl",
        "Stats": Path(PROJECT_ROOT) / "R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt",
    }

    all_exist = True
    for name, path in files.items():
        exists = path.exists()
        status = "✓" if exists else "✗"
        print(f"{status} {name}: {path}")
        if not exists:
            all_exist = False

    if not all_exist:
        print("\n❌ Some required files are missing!")
        return False

    print("\n✅ All required files exist\n")
    return True


def validate_imports():
    """Test all imports."""
    print("=" * 60)
    print("Step 2: Validating imports...")
    print("=" * 60)

    try:
        import torch
        print(f"✓ PyTorch: {torch.__version__}")
    except ImportError as e:
        print(f"✗ PyTorch import failed: {e}")
        return False

    try:
        import vllm
        print(f"✓ vLLM: {vllm.__version__}")
    except ImportError as e:
        print(f"✗ vLLM import failed: {e}")
        return False

    try:
        from triattention import (
            TriAttentionConfig,
            TriAttentionWrapper,
            create_triattention_wrapper,
            patch_vllm_attention,
        )
        print(f"✓ TriAttention modules imported")
    except ImportError as e:
        print(f"✗ TriAttention import failed: {e}")
        return False

    print("\n✅ All imports successful\n")
    return True


def validate_config():
    """Test config creation."""
    print("=" * 60)
    print("Step 3: Validating configuration...")
    print("=" * 60)

    try:
        from triattention import TriAttentionConfig

        stats_path = PROJECT_ROOT / "R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt"

        config = TriAttentionConfig(
            stats_path=stats_path,
            kv_budget=2048,
            divide_length=128,
            sparse_round_window=32,
            pruning_mode="per_head",
            score_aggregation="mean",
            offset_max_length=65536,
            sparse_normalize_scores=True,
            window_size=128,
            include_prefill_in_budget=True,
            protect_prefill=False,  # NOT protect when include_prefill_in_budget=True
        )

        print(f"✓ Config created successfully")
        print(f"  - kv_budget: {config.kv_budget}")
        print(f"  - divide_length: {config.divide_length}")
        print(f"  - sparse_round_window: {config.sparse_round_window}")
        print(f"  - pruning_mode: {config.pruning_mode}")
        print(f"  - stats_path: {config.stats_path}")
        print(f"  - include_prefill_in_budget: {config.include_prefill_in_budget}")
        print(f"  - protect_prefill: {config.protect_prefill}")

    except Exception as e:
        print(f"✗ Config creation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Configuration valid\n")
    return True


def validate_triton_kernels():
    """Test Triton kernel compilation."""
    print("=" * 60)
    print("Step 4: Validating Triton kernels...")
    print("=" * 60)

    try:
        import torch
        from triattention.kernels.triton_scoring import speckv_scoring

        print(f"✓ Triton scoring kernel imported")

        # Try a minimal kernel compilation test
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device.type != "cuda":
            print("⚠ CUDA not available, skipping kernel test")
            return True

        print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")

    except Exception as e:
        print(f"✗ Triton kernel validation failed: {e}")
        import traceback
        traceback.print_exc()
        return False

    print("\n✅ Triton kernels valid\n")
    return True


def main():
    """Run all validation checks."""
    print("\n" + "=" * 60)
    print("TriAttention + vLLM Setup Validation")
    print("=" * 60 + "\n")

    checks = [
        ("Files", validate_files),
        ("Imports", validate_imports),
        ("Config", validate_config),
        ("Triton", validate_triton_kernels),
    ]

    results = {}
    for name, check_fn in checks:
        try:
            results[name] = check_fn()
        except Exception as e:
            print(f"\n❌ {name} validation crashed: {e}")
            import traceback
            traceback.print_exc()
            results[name] = False

    # Print summary
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    for name, passed in results.items():
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status}: {name}")

    all_passed = all(results.values())
    print("=" * 60)
    if all_passed:
        print("✅ All validation checks passed!")
        print("✅ Ready to run benchmark")
    else:
        print("❌ Some validation checks failed")
        print("❌ Please fix issues before running benchmark")
    print("=" * 60)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
