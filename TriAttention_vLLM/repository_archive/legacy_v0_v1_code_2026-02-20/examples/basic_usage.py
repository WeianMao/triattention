"""Basic usage example for TriAttention compression.

This example demonstrates:
1. Creating a TriAttention wrapper with configuration
2. Per-request state management
3. Compression during generation
4. Proper cleanup

Prerequisites:
- Frequency statistics file (generate using tools/generate_stats.py)
- PyTorch with CUDA support
"""
import torch
from pathlib import Path
from triattention.config import TriAttentionConfig
from triattention.vllm_integration import TriAttentionWrapper


def example_basic_compression():
    """Example 1: Basic compression with single request."""
    # Step 1: Configure TriAttention
    config = TriAttentionConfig(
        stats_path=Path("path/to/frequency_stats.pt"),
        kv_budget=2048,
        divide_length=128,
        pruning_mode="per_head",
        num_layers=32,
        num_kv_heads=8,
        head_dim=128,
        device="cuda",
    )

    # Step 2: Create wrapper
    wrapper = TriAttentionWrapper(config)

    # Step 3: Register request
    request_id = "req_001"
    wrapper.register_request(request_id)

    # Simulate prefill
    batch_size, num_kv_heads, prefill_len, head_dim = 1, 8, 512, 128
    key_cache = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim, device="cuda")
    value_cache = torch.randn(batch_size, num_kv_heads, prefill_len, head_dim, device="cuda")
    cache_positions = torch.arange(prefill_len, device="cuda")

    # Check if compression needed (not yet, cache < budget)
    layer_idx = 0
    if wrapper.should_compress(layer_idx, prefill_len, request_id):
        key_cache, value_cache, positions = wrapper.compress_kv_cache(
            key_cache, value_cache, cache_positions, layer_idx, request_id
        )
        print(f"Compressed cache to {key_cache.shape[2]} tokens")
    else:
        print(f"No compression needed, cache size: {prefill_len}")

    # Simulate decode steps until compression triggers
    current_len = prefill_len
    for step in range(1, 2000):
        # Add new token
        current_len += 1
        new_key = torch.randn(batch_size, num_kv_heads, 1, head_dim, device="cuda")
        new_value = torch.randn(batch_size, num_kv_heads, 1, head_dim, device="cuda")
        key_cache = torch.cat([key_cache, new_key], dim=2)
        value_cache = torch.cat([value_cache, new_value], dim=2)

        # Check compression trigger
        if wrapper.should_compress(layer_idx, current_len, request_id):
            cache_positions = torch.arange(current_len, device="cuda")
            key_cache, value_cache, keep_indices = wrapper.compress_kv_cache(
                key_cache, value_cache, cache_positions, layer_idx, request_id
            )
            current_len = key_cache.shape[2]
            print(f"Step {step}: Compressed cache from {current_len + 128} to {current_len} tokens")

    # Step 4: Cleanup when request completes
    wrapper.unregister_request(request_id)
    print(f"Request {request_id} completed and cleaned up")


def example_multi_request():
    """Example 2: Managing multiple concurrent requests."""
    config = TriAttentionConfig(
        stats_path=Path("path/to/frequency_stats.pt"),
        kv_budget=1024,
        divide_length=64,
        pruning_mode="per_layer",
    )

    wrapper = TriAttentionWrapper(config)

    # Register multiple requests
    request_ids = ["req_001", "req_002", "req_003"]
    for req_id in request_ids:
        wrapper.register_request(req_id)

    # Each request maintains independent state
    for req_id in request_ids:
        state = wrapper.get_request_state_summary(req_id)
        print(f"{req_id} state: {state}")

    # Process requests independently...
    # (compression logic same as example 1)

    # Cleanup completed requests
    for req_id in request_ids:
        wrapper.unregister_request(req_id)

    print(f"Active requests: {wrapper.get_active_requests()}")


def example_error_handling():
    """Example 3: Common error scenarios and handling."""
    # Error 1: Missing stats file
    try:
        config = TriAttentionConfig(
            stats_path=None,  # Missing!
            kv_budget=2048,
        )
        wrapper = TriAttentionWrapper(config)
        # This will fail on first compress() call
        wrapper.compress_kv_cache(
            torch.randn(1, 8, 100, 128),
            torch.randn(1, 8, 100, 128),
            torch.arange(100),
            layer_idx=0,
        )
    except ValueError as e:
        print(f"Error 1 (Missing stats_path): {e}")

    # Error 2: Stats file not found
    try:
        config = TriAttentionConfig(
            stats_path=Path("nonexistent_stats.pt"),
            kv_budget=2048,
        )
        wrapper = TriAttentionWrapper(config)
        # This will fail on first compress() call when loading stats
        wrapper.compress_kv_cache(
            torch.randn(1, 8, 100, 128),
            torch.randn(1, 8, 100, 128),
            torch.arange(100),
            layer_idx=0,
        )
    except FileNotFoundError as e:
        print(f"Error 2 (Stats file not found): {e}")

    # Error 3: Invalid budget
    try:
        config = TriAttentionConfig(
            stats_path=Path("stats.pt"),
            kv_budget=-100,  # Invalid!
        )
    except ValueError as e:
        print(f"Error 3 (Invalid budget): {e}")

    # Error 4: Invalid divide_length
    try:
        config = TriAttentionConfig(
            stats_path=Path("stats.pt"),
            kv_budget=2048,
            divide_length=0,  # Invalid!
        )
    except ValueError as e:
        print(f"Error 4 (Invalid divide_length): {e}")


if __name__ == "__main__":
    print("TriAttention Basic Usage Examples")
    print("=" * 50)

    print("\n1. Basic Compression Example")
    print("-" * 50)
    # Uncomment when you have stats file:
    # example_basic_compression()
    print("(Skipped - requires frequency statistics file)")

    print("\n2. Multi-Request Example")
    print("-" * 50)
    # example_multi_request()
    print("(Skipped - requires frequency statistics file)")

    print("\n3. Error Handling Example")
    print("-" * 50)
    example_error_handling()

    print("\nTo run full examples, generate frequency statistics first:")
    print("  python -m triattention.tools.generate_stats <model_path> <output_path>")
