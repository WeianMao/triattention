"""Test per-request state isolation in TriAttention vLLM integration.

This test verifies that multiple requests can be processed concurrently
without their compression states interfering with each other.
"""

import pytest
import torch

from triattention.config import TriAttentionConfig
from triattention.vllm_integration import TriAttentionWrapper, PagedKVCacheCompressor


class TestPerRequestIsolation:
    """Test suite for per-request state isolation."""

    def test_wrapper_multiple_requests_isolation(self, tmp_path):
        """Test that multiple requests maintain separate compressor states."""
        # Create minimal config
        config = TriAttentionConfig(
            stats_path=None,  # Will use lazy init
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
        comp1_layer0 = wrapper.get_compressor(layer_idx, req1_id)
        comp2_layer0 = wrapper.get_compressor(layer_idx, req2_id)

        # Verify they are different instances
        assert comp1_layer0 is not comp2_layer0

        # Verify requests are registered
        assert req1_id in wrapper.request_compressors
        assert req2_id in wrapper.request_compressors

        # Unregister request 1
        wrapper.unregister_request(req1_id)

        # Verify request 1 is gone but request 2 remains
        assert req1_id not in wrapper.request_compressors
        assert req2_id in wrapper.request_compressors

    def test_wrapper_state_independence(self, tmp_path):
        """Test that state changes in one request don't affect another."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
        )

        wrapper = TriAttentionWrapper(config)

        req1_id = "request_A"
        req2_id = "request_B"

        # Get compressors for both requests
        comp1 = wrapper.get_compressor(0, req1_id)
        comp2 = wrapper.get_compressor(0, req2_id)

        # Modify state in request 1
        comp1.state.absolute_position = 100
        comp1.state.compression_count = 5
        comp1.state.prefill_length = 50

        # Verify request 2's state is unaffected
        assert comp2.state.absolute_position == 0
        assert comp2.state.compression_count == 0
        assert comp2.state.prefill_length == 0

        # Get state summaries
        summary1 = wrapper.get_request_state_summary(req1_id)
        summary2 = wrapper.get_request_state_summary(req2_id)

        assert summary1[0]["absolute_position"] == 100
        assert summary2[0]["absolute_position"] == 0

    def test_wrapper_reset_specific_request(self, tmp_path):
        """Test that resetting one request doesn't affect others."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
        )

        wrapper = TriAttentionWrapper(config)

        req1_id = "request_X"
        req2_id = "request_Y"

        # Get compressors and modify states
        comp1 = wrapper.get_compressor(0, req1_id)
        comp2 = wrapper.get_compressor(0, req2_id)

        comp1.state.absolute_position = 100
        comp2.state.absolute_position = 200

        # Reset only request 1
        wrapper.reset_all(req1_id)

        # Verify request 1 is reset but request 2 is not
        assert comp1.state.absolute_position == 0
        assert comp2.state.absolute_position == 200

    def test_wrapper_layer_idx_validation(self, tmp_path):
        """Test layer index validation."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=4,
            num_kv_heads=2,
            head_dim=64,
        )

        wrapper = TriAttentionWrapper(config)

        # Valid layer indices should work
        wrapper.get_compressor(0, "req1")
        wrapper.get_compressor(3, "req1")

        # Invalid layer indices should raise ValueError
        with pytest.raises(ValueError, match="Invalid layer_idx"):
            wrapper.get_compressor(-1, "req1")

        with pytest.raises(ValueError, match="Invalid layer_idx"):
            wrapper.get_compressor(4, "req1")

    def test_wrapper_backward_compatibility(self, tmp_path):
        """Test backward compatibility when request_id is not provided."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
        )

        wrapper = TriAttentionWrapper(config)

        # Get compressor without request_id (should use default)
        comp1 = wrapper.get_compressor(0)
        comp2 = wrapper.get_compressor(0)

        # Should return same instance (default request)
        assert comp1 is comp2

        # Verify default request was created
        assert wrapper._default_request_id in wrapper.request_compressors

    def test_wrapper_active_requests_tracking(self, tmp_path):
        """Test tracking of active requests."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
        )

        wrapper = TriAttentionWrapper(config)

        # Initially no active requests
        assert len(wrapper.get_active_requests()) == 0

        # Register requests
        wrapper.register_request("req1")
        wrapper.register_request("req2")
        wrapper.register_request("req3")

        # Should track all requests except default
        active = wrapper.get_active_requests()
        assert len(active) == 3
        assert "req1" in active
        assert "req2" in active
        assert "req3" in active

        # Unregister one request
        wrapper.unregister_request("req2")
        active = wrapper.get_active_requests()
        assert len(active) == 2
        assert "req2" not in active

    def test_paged_compressor_per_request_isolation(self, tmp_path):
        """Test PagedKVCacheCompressor per-request isolation."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=2,
            num_kv_heads=2,
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
        assert comp1 is not comp2

        # Modify state in one
        comp1.state.absolute_position = 50

        # Verify independence
        assert comp1.state.absolute_position == 50
        assert comp2.state.absolute_position == 0

        # Unregister
        paged_comp.unregister_request("req1")
        assert "req1" not in paged_comp.request_compressors

    def test_paged_compressor_auto_registration(self, tmp_path):
        """Test auto-registration in PagedKVCacheCompressor."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
        )

        paged_comp = PagedKVCacheCompressor(config, block_size=16)

        # Get compressor without registration (should auto-register)
        comp = paged_comp._get_compressor("auto_req")

        # Verify auto-registration occurred
        assert "auto_req" in paged_comp.request_compressors
        assert comp is not None

    def test_wrapper_re_registration_resets_state(self, tmp_path):
        """Test that re-registering a request resets its state."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=2,
            num_kv_heads=2,
            head_dim=64,
        )

        wrapper = TriAttentionWrapper(config)

        req_id = "test_req"

        # Register and modify state
        comp = wrapper.get_compressor(0, req_id)
        comp.state.absolute_position = 100
        comp.state.compression_count = 5

        # Re-register (should reset)
        wrapper.register_request(req_id)

        # Get compressor again
        comp_new = wrapper.get_compressor(0, req_id)

        # State should be reset
        assert comp_new.state.absolute_position == 0
        assert comp_new.state.compression_count == 0


class TestMemoryManagement:
    """Test memory management for per-request states."""

    def test_memory_cleanup_on_unregister(self, tmp_path):
        """Test that unregistering requests properly cleans up memory."""
        config = TriAttentionConfig(
            stats_path=None,
            kv_budget=64,
            divide_length=32,
            num_layers=4,
            num_kv_heads=4,
            head_dim=64,
        )

        wrapper = TriAttentionWrapper(config)

        # Create multiple requests with compressors across layers
        num_requests = 10
        num_layers = 4

        for i in range(num_requests):
            req_id = f"req_{i}"
            wrapper.register_request(req_id)
            # Create compressors for all layers
            for layer in range(num_layers):
                wrapper.get_compressor(layer, req_id)

        # Verify all requests are registered
        assert len(wrapper.request_compressors) == num_requests

        # Unregister half the requests
        for i in range(num_requests // 2):
            req_id = f"req_{i}"
            wrapper.unregister_request(req_id)

        # Verify cleanup
        assert len(wrapper.request_compressors) == num_requests // 2

        # Verify remaining requests are intact
        for i in range(num_requests // 2, num_requests):
            req_id = f"req_{i}"
            assert req_id in wrapper.request_compressors
            assert len(wrapper.request_compressors[req_id]) == num_layers
