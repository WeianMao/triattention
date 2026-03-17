"""Standalone verification script for per-request state isolation fix.

This script demonstrates that the P1 issue has been fixed:
- Multiple requests can now maintain separate compression states
- State changes in one request don't affect others
- Proper cleanup occurs when requests complete
"""

import sys
import os

# Add triattention to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from triattention.config import TriAttentionConfig
from triattention.vllm_integration import TriAttentionWrapper, PagedKVCacheCompressor


def test_basic_isolation():
    """Test basic per-request state isolation."""
    print("Test 1: Basic per-request isolation")

    config = TriAttentionConfig(
        stats_path=None,
        kv_budget=64,
        divide_length=32,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
    )

    wrapper = TriAttentionWrapper(config)

    # Register two requests
    req1_id = "request_1"
    req2_id = "request_2"

    wrapper.register_request(req1_id)
    wrapper.register_request(req2_id)

    # Get compressors for same layer but different requests
    layer_idx = 0
    comp1 = wrapper.get_compressor(layer_idx, req1_id)
    comp2 = wrapper.get_compressor(layer_idx, req2_id)

    # Verify they are different instances
    assert comp1 is not comp2, "Compressors should be different instances"

    print("  ✓ Different requests have separate compressor instances")

    # Modify state in request 1
    comp1.state.absolute_position = 100
    comp1.state.compression_count = 5

    # Verify request 2's state is unaffected
    assert comp2.state.absolute_position == 0, "Request 2 state should be unaffected"
    assert comp2.state.compression_count == 0, "Request 2 state should be unaffected"

    print("  ✓ State changes in one request don't affect others")
    print("  PASSED\n")


def test_state_independence():
    """Test state independence across multiple requests and layers."""
    print("Test 2: State independence across requests and layers")

    config = TriAttentionConfig(
        stats_path=None,
        kv_budget=64,
        divide_length=32,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
    )

    wrapper = TriAttentionWrapper(config)

    # Create multiple requests
    requests = ["req_A", "req_B", "req_C"]
    layers = [0, 1, 2]

    # Initialize states with different values
    for req_id in requests:
        wrapper.register_request(req_id)
        for layer_idx in layers:
            comp = wrapper.get_compressor(layer_idx, req_id)
            # Set unique value based on request and layer
            comp.state.absolute_position = (requests.index(req_id) + 1) * 100 + layer_idx

    # Verify each request-layer pair has independent state
    for req_id in requests:
        for layer_idx in layers:
            comp = wrapper.get_compressor(layer_idx, req_id)
            expected_pos = (requests.index(req_id) + 1) * 100 + layer_idx
            assert comp.state.absolute_position == expected_pos, \
                f"Request {req_id} layer {layer_idx} state incorrect"

    print(f"  ✓ Verified independence for {len(requests)} requests × {len(layers)} layers")
    print("  PASSED\n")


def test_cleanup():
    """Test proper cleanup on request unregistration."""
    print("Test 3: Cleanup on request unregistration")

    config = TriAttentionConfig(
        stats_path=None,
        kv_budget=64,
        divide_length=32,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
    )

    wrapper = TriAttentionWrapper(config)

    # Create requests
    num_requests = 10
    for i in range(num_requests):
        req_id = f"req_{i}"
        wrapper.register_request(req_id)
        # Create compressors for multiple layers
        for layer in range(4):
            wrapper.get_compressor(layer, req_id)

    assert len(wrapper.request_compressors) == num_requests, \
        f"Should have {num_requests} registered requests"

    print(f"  ✓ Created {num_requests} requests with 4 layers each")

    # Unregister half the requests
    for i in range(num_requests // 2):
        req_id = f"req_{i}"
        wrapper.unregister_request(req_id)

    remaining = num_requests - num_requests // 2
    assert len(wrapper.request_compressors) == remaining, \
        f"Should have {remaining} requests after cleanup"

    print(f"  ✓ Properly cleaned up {num_requests // 2} requests")

    # Verify remaining requests are intact
    for i in range(num_requests // 2, num_requests):
        req_id = f"req_{i}"
        assert req_id in wrapper.request_compressors, f"Request {req_id} should still exist"
        assert len(wrapper.request_compressors[req_id]) == 4, \
            f"Request {req_id} should still have 4 layers"

    print(f"  ✓ Remaining {remaining} requests are intact")
    print("  PASSED\n")


def test_layer_validation():
    """Test layer index validation."""
    print("Test 4: Layer index validation")

    config = TriAttentionConfig(
        stats_path=None,
        kv_budget=64,
        divide_length=32,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
    )

    wrapper = TriAttentionWrapper(config)

    # Valid layer indices should work
    try:
        wrapper.get_compressor(0, "req1")
        wrapper.get_compressor(3, "req1")
        print("  ✓ Valid layer indices accepted")
    except ValueError:
        print("  ✗ Valid layer indices rejected")
        return False

    # Invalid layer indices should raise ValueError
    try:
        wrapper.get_compressor(-1, "req1")
        print("  ✗ Negative layer index should be rejected")
        return False
    except ValueError:
        print("  ✓ Negative layer index rejected")

    try:
        wrapper.get_compressor(4, "req1")
        print("  ✗ Out-of-bounds layer index should be rejected")
        return False
    except ValueError:
        print("  ✓ Out-of-bounds layer index rejected")

    print("  PASSED\n")
    return True


def test_backward_compatibility():
    """Test backward compatibility when request_id is not provided."""
    print("Test 5: Backward compatibility (no request_id)")

    config = TriAttentionConfig(
        stats_path=None,
        kv_budget=64,
        divide_length=32,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
    )

    wrapper = TriAttentionWrapper(config)

    # Get compressor without request_id (should use default)
    comp1 = wrapper.get_compressor(0)
    comp2 = wrapper.get_compressor(0)

    # Should return same instance (default request)
    assert comp1 is comp2, "Should return same instance for default request"

    print("  ✓ Default request returns same compressor instance")

    # Verify default request was created
    assert wrapper._default_request_id in wrapper.request_compressors, \
        "Default request should be auto-registered"

    print("  ✓ Default request auto-registered")
    print("  PASSED\n")


def test_paged_compressor_isolation():
    """Test PagedKVCacheCompressor per-request isolation."""
    print("Test 6: PagedKVCacheCompressor isolation")

    config = TriAttentionConfig(
        stats_path=None,
        kv_budget=64,
        divide_length=32,
        num_layers=4,
        num_kv_heads=4,
        head_dim=64,
    )

    paged_comp = PagedKVCacheCompressor(config, block_size=16)

    # Register requests
    paged_comp.register_request("req1")
    paged_comp.register_request("req2")

    # Get compressors
    comp1 = paged_comp._get_compressor("req1")
    comp2 = paged_comp._get_compressor("req2")

    # Verify different instances
    assert comp1 is not comp2, "Should be different compressor instances"

    print("  ✓ Different requests have separate compressor instances")

    # Modify state in one
    comp1.state.absolute_position = 50

    # Verify independence
    assert comp1.state.absolute_position == 50
    assert comp2.state.absolute_position == 0

    print("  ✓ State changes are isolated")

    # Unregister
    paged_comp.unregister_request("req1")
    assert "req1" not in paged_comp.request_compressors

    print("  ✓ Cleanup works correctly")
    print("  PASSED\n")


def main():
    """Run all verification tests."""
    print("=" * 60)
    print("Per-Request State Isolation Fix Verification")
    print("=" * 60)
    print()

    tests = [
        test_basic_isolation,
        test_state_independence,
        test_cleanup,
        test_layer_validation,
        test_backward_compatibility,
        test_paged_compressor_isolation,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            result = test()
            if result is not False:
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"  ✗ FAILED: {e}")
            failed += 1
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed}/{len(tests)} tests passed")
    if failed > 0:
        print(f"         {failed}/{len(tests)} tests failed")
    print("=" * 60)

    return failed == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
