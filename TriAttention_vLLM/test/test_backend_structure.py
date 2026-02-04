#!/usr/bin/env python3
"""Unit tests for TriAttention Backend structure.

This script verifies that the backend implementation correctly inherits
from vLLM's FlashAttention backend and provides the expected interface.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_backend_imports():
    """Test that backend modules can be imported."""
    print("Testing backend imports...")

    try:
        from triattention.backends import (
            TriAttentionBackend,
            TriAttentionImpl,
            setup_triattention,
            register_triattention_backend,
        )
        print("✓ All backend modules imported successfully")
        return True
    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_backend_inheritance():
    """Test that TriAttentionBackend inherits from FlashAttentionBackend."""
    print("\nTesting backend inheritance...")

    try:
        from triattention.backends import TriAttentionBackend
        from vllm.attention.backends.flash_attn import FlashAttentionBackend

        assert issubclass(TriAttentionBackend, FlashAttentionBackend), \
            "TriAttentionBackend must inherit from FlashAttentionBackend"

        print("✓ TriAttentionBackend correctly inherits from FlashAttentionBackend")
        return True
    except ImportError as e:
        print(f"✗ Import failed (vLLM not available?): {e}")
        return False
    except AssertionError as e:
        print(f"✗ Inheritance check failed: {e}")
        return False


def test_impl_inheritance():
    """Test that TriAttentionImpl inherits from FlashAttentionImpl."""
    print("\nTesting implementation inheritance...")

    try:
        from triattention.backends import TriAttentionImpl
        from vllm.attention.backends.flash_attn import FlashAttentionImpl

        assert issubclass(TriAttentionImpl, FlashAttentionImpl), \
            "TriAttentionImpl must inherit from FlashAttentionImpl"

        print("✓ TriAttentionImpl correctly inherits from FlashAttentionImpl")
        return True
    except ImportError as e:
        print(f"✗ Import failed (vLLM not available?): {e}")
        return False
    except AssertionError as e:
        print(f"✗ Inheritance check failed: {e}")
        return False


def test_backend_interface():
    """Test that TriAttentionBackend provides the expected interface."""
    print("\nTesting backend interface...")

    try:
        from triattention.backends import TriAttentionBackend

        # Check get_name()
        name = TriAttentionBackend.get_name()
        assert name == "TRIATTENTION", f"Expected 'TRIATTENTION', got '{name}'"
        print(f"✓ Backend name: {name}")

        # Check get_impl_cls()
        from triattention.backends import TriAttentionImpl
        impl_cls = TriAttentionBackend.get_impl_cls()
        assert impl_cls is TriAttentionImpl, \
            f"get_impl_cls() should return TriAttentionImpl, got {impl_cls}"
        print("✓ Backend returns correct implementation class")

        # Check inherited methods exist
        assert hasattr(TriAttentionBackend, 'get_supported_head_sizes'), \
            "Missing get_supported_head_sizes()"
        assert hasattr(TriAttentionBackend, 'get_metadata_cls'), \
            "Missing get_metadata_cls()"
        assert hasattr(TriAttentionBackend, 'get_kv_cache_shape'), \
            "Missing get_kv_cache_shape()"
        print("✓ Inherited methods are available")

        return True
    except Exception as e:
        print(f"✗ Interface check failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_config_setup():
    """Test configuration setup and retrieval."""
    print("\nTesting configuration setup...")

    try:
        from triattention import TriAttentionConfig
        from triattention.backends import setup_triattention, get_triattention_config

        # Create config
        config = TriAttentionConfig(
            kv_budget=2048,
            divide_length=128,
        )

        # Setup config
        setup_triattention(config)

        # Retrieve config
        retrieved = get_triattention_config()
        assert retrieved is config, "Retrieved config should be the same object"
        assert retrieved.kv_budget == 2048, "Config values should be preserved"

        print("✓ Configuration setup and retrieval works correctly")
        return True
    except Exception as e:
        print(f"✗ Config setup failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_registration_without_config():
    """Test that registration fails without config setup."""
    print("\nTesting registration without config...")

    try:
        # Clear config first
        import triattention.backends as backends_module
        backends_module._TRIATTENTION_CONFIG = None

        from triattention.backends import register_triattention_backend

        try:
            register_triattention_backend()
            print("✗ Registration should fail without config")
            return False
        except RuntimeError as e:
            if "not set" in str(e).lower():
                print("✓ Registration correctly fails without config")
                return True
            else:
                print(f"✗ Unexpected error: {e}")
                return False
    except Exception as e:
        print(f"✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run all tests."""
    print("=" * 80)
    print("TriAttention Backend Structure Tests")
    print("=" * 80)

    tests = [
        test_backend_imports,
        test_backend_inheritance,
        test_impl_inheritance,
        test_backend_interface,
        test_config_setup,
        test_registration_without_config,
    ]

    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"✗ Test {test.__name__} crashed: {e}")
            import traceback
            traceback.print_exc()
            results.append(False)

    print("\n" + "=" * 80)
    print("Test Summary")
    print("=" * 80)
    passed = sum(results)
    total = len(results)
    print(f"Passed: {passed}/{total}")

    if passed == total:
        print("\n✓ All tests passed!")
        return 0
    else:
        print(f"\n✗ {total - passed} test(s) failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())
