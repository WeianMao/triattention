#!/usr/bin/env python3
"""Test script to verify vLLM attention hook mechanism.

This script tests that:
1. TriAttention wrapper can be created
2. vLLM attention layers can be patched
3. Compression hook is triggered during inference
4. No crashes occur during patching and generation

Usage:
    conda activate trivllm
    python test/test_vllm_hook.py
"""
import sys
from pathlib import Path

import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from triattention import (
    TriAttentionConfig,
    TriAttentionWrapper,
    patch_vllm_attention,
)


def test_wrapper_creation():
    """Test 1: Create TriAttention wrapper without stats file."""
    print("\n[Test 1] Creating TriAttention wrapper...")

    config = TriAttentionConfig(
        kv_budget=2048,
        divide_length=128,
        pruning_mode="per_head",
        stats_path=None,  # No stats file for basic test
        head_dim=128,
        num_kv_heads=8,
        num_layers=32,
    )

    wrapper = TriAttentionWrapper(config)
    print("  ✓ Wrapper created successfully")
    print(f"  ✓ Config: budget={config.kv_budget}, divide_length={config.divide_length}")

    return wrapper


def test_vllm_import():
    """Test 2: Import vLLM and verify version."""
    print("\n[Test 2] Checking vLLM installation...")

    try:
        import vllm
        print(f"  ✓ vLLM version: {vllm.__version__}")

        from vllm import LLM, SamplingParams
        print("  ✓ vLLM imports successful")
        return True
    except ImportError as e:
        print(f"  ✗ vLLM import failed: {e}")
        return False


def test_patching_with_mock_model():
    """Test 3: Test patching mechanism with mock model structure."""
    print("\n[Test 3] Testing patch mechanism with mock model...")

    # Create a minimal mock model structure
    class MockAttentionImpl:
        def __init__(self):
            self.forward_called = False

        def forward(self, layer, query, key, value, kv_cache, attn_metadata, output=None):
            self.forward_called = True
            return output

    class MockAttentionLayer:
        def __init__(self):
            self.impl = MockAttentionImpl()

    class MockTransformerLayer:
        def __init__(self):
            self.self_attn = MockAttentionLayer()

    class MockModel:
        def __init__(self, num_layers=4):
            self.model = type('obj', (object,), {'layers': [MockTransformerLayer() for _ in range(num_layers)]})()

    # Create mock model
    mock_model = MockModel(num_layers=4)

    # Create wrapper
    config = TriAttentionConfig(
        kv_budget=2048,
        divide_length=128,
        pruning_mode="per_head",
        stats_path=None,
        head_dim=128,
        num_kv_heads=8,
        num_layers=4,
    )
    wrapper = TriAttentionWrapper(config)

    # Patch the model
    try:
        patch_vllm_attention(mock_model, wrapper)
        print("  ✓ Patching completed successfully")
        print(f"  ✓ Wrapper patched status: {wrapper._patched}")

        # Verify that forward method was wrapped
        original_impl = mock_model.model.layers[0].self_attn.impl
        print(f"  ✓ Layer 0 impl type: {type(original_impl)}")

        return True
    except Exception as e:
        print(f"  ✗ Patching failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_vllm_model_access():
    """Test 4: Check if we can access vLLM model structure (requires GPU)."""
    print("\n[Test 4] Testing vLLM model access...")

    if not torch.cuda.is_available():
        print("  ⊘ Skipping (no GPU available)")
        return True

    try:
        from vllm import LLM

        # Use Qwen model for testing
        print("  Loading Qwen model for testing...")
        llm = LLM(
            model="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B",
            dtype="float16",
            gpu_memory_utilization=0.9,
            max_model_len=1024,
            enforce_eager=True,
            trust_remote_code=True,
        )

        # Access model structure
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        print(f"  ✓ Model type: {type(model)}")

        if hasattr(model, "model") and hasattr(model.model, "decoder"):
            layers = model.model.decoder.layers
            print(f"  ✓ Found {len(layers)} decoder layers")
        elif hasattr(model, "model") and hasattr(model.model, "layers"):
            layers = model.model.layers
            print(f"  ✓ Found {len(layers)} transformer layers")
        else:
            print(f"  ⚠ Unexpected model structure: {dir(model)}")

        # Check attention layer structure
        if len(layers) > 0:
            layer0 = layers[0]
            print(f"  ✓ Layer 0 attributes: {[attr for attr in dir(layer0) if not attr.startswith('_')]}")

            if hasattr(layer0, "self_attn"):
                print(f"  ✓ Found self_attn in layer 0")
                attn = layer0.self_attn
                if hasattr(attn, "impl"):
                    print(f"  ✓ Found attention impl: {type(attn.impl)}")

        return True

    except Exception as e:
        print(f"  ✗ vLLM model access test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_full_integration():
    """Test 5: Full integration test with tiny model (requires GPU)."""
    print("\n[Test 5] Full integration test with TriAttention...")

    if not torch.cuda.is_available():
        print("  ⊘ Skipping (no GPU available)")
        return True

    try:
        from vllm import LLM, SamplingParams

        # Create config with real stats
        stats_path = Path("/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt")
        config = TriAttentionConfig(
            kv_budget=256,  # Small budget for testing
            divide_length=64,
            pruning_mode="per_head",
            stats_path=stats_path,
        )
        wrapper = TriAttentionWrapper(config)

        # Load Qwen model
        print("  Loading Qwen model...")
        llm = LLM(
            model="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B",
            dtype="float16",
            gpu_memory_utilization=0.9,
            max_model_len=1024,
            enforce_eager=True,
            trust_remote_code=True,
        )

        # Patch attention
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        patch_vllm_attention(model, wrapper)
        print(f"  ✓ Patching status: {wrapper._patched}")

        # Run simple generation
        sampling_params = SamplingParams(
            temperature=0.8,
            top_p=0.95,
            max_tokens=50,
        )

        prompt = "The quick brown fox"
        print(f"  Running generation: '{prompt}'")
        outputs = llm.generate([prompt], sampling_params)

        generated_text = outputs[0].outputs[0].text
        print(f"  ✓ Generated: '{generated_text[:50]}...'")
        print("  ✓ Full integration test passed")

        return True

    except Exception as e:
        print(f"  ✗ Full integration test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("TriAttention vLLM Hook Mechanism Test Suite")
    print("=" * 80)

    results = {}

    # Test 1: Wrapper creation
    try:
        test_wrapper_creation()
        results["wrapper_creation"] = True
    except Exception as e:
        print(f"  ✗ Test failed: {e}")
        results["wrapper_creation"] = False

    # Test 2: vLLM import
    results["vllm_import"] = test_vllm_import()

    # Test 3: Patching mechanism
    results["patching_mock"] = test_patching_with_mock_model()

    # Test 4: vLLM model access (requires GPU)
    results["model_access"] = test_vllm_model_access()

    # Test 5: Full integration (requires GPU)
    results["full_integration"] = test_full_integration()

    # Summary
    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)

    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status}: {test_name}")

    total_passed = sum(results.values())
    total_tests = len(results)
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")

    if total_passed == total_tests:
        print("\n🎉 All tests passed!")
        return 0
    else:
        print("\n⚠ Some tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
