"""Verify the state tracking issue in vLLM integration.

This demonstrates the bug where:
1. Wrapper maintains compressor state
2. But _apply_triattention_compression creates a NEW PagedKVCacheCompressor
3. So the state is never consulted properly
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from triattention import TriAttentionConfig, TriAttentionWrapper
from triattention.vllm_integration import PagedKVCacheCompressor


def test_state_tracking_bug():
    """Demonstrate the state tracking bug."""
    print("\n" + "=" * 80)
    print("Demonstrating State Tracking Bug")
    print("=" * 80)

    # Create wrapper
    config = TriAttentionConfig(
        kv_budget=256,
        divide_length=64,
        protect_prefill=False,
        pruning_mode="per_head",
        stats_path=None,
        head_dim=128,
        num_kv_heads=4,
        num_layers=28,
    )
    wrapper = TriAttentionWrapper(config)

    # Simulate decode steps
    request_id = "decode_0"
    layer_idx = 0

    print(f"\nConfiguration:")
    print(f"  kv_budget: {config.kv_budget}")
    print(f"  divide_length: {config.divide_length}")
    print(f"  trigger_threshold: {config.kv_budget + config.divide_length}")
    print()

    # Simulate sequence growth
    for seq_len in [100, 200, 300, 320, 350, 400, 512]:
        # This is what the hook does - it calls wrapper.should_compress
        # which uses the wrapper's compressor state
        should_compress_wrapper = wrapper.should_compress(layer_idx, seq_len, request_id)

        # Get the wrapper's compressor to check its state
        wrapper_compressor = wrapper.get_compressor(layer_idx, request_id)

        print(f"seq_len={seq_len}:")
        print(f"  Wrapper compressor state:")
        print(f"    current_cache_len: {wrapper_compressor.state.current_cache_len}")
        print(f"    prefill_length: {wrapper_compressor.state.prefill_length}")
        print(f"    compression_count: {wrapper_compressor.state.compression_count}")
        print(f"  wrapper.should_compress() = {should_compress_wrapper}")

        # But then the hook creates a NEW PagedKVCacheCompressor!
        # (Line 810-816 in vllm_integration.py:_apply_triattention_compression)
        new_paged_compressor = PagedKVCacheCompressor(
            config=config,
            block_size=16,
        )
        new_paged_compressor.register_request(request_id)

        # This new paged compressor has a fresh TriAttentionCompressor!
        new_comp = new_paged_compressor._get_compressor(request_id)
        should_compress_new = new_comp.state.should_compress(seq_len)

        print(f"  NEW PagedKVCacheCompressor's state:")
        print(f"    current_cache_len: {new_comp.state.current_cache_len}")
        print(f"    prefill_length: {new_comp.state.prefill_length}")
        print(f"    compression_count: {new_comp.state.compression_count}")
        print(f"  should_compress() = {should_compress_new}")
        print()

    print("=" * 80)
    print("\nBUG IDENTIFIED:")
    print("  - Wrapper maintains state in its compressor instances")
    print("  - But _apply_triattention_compression creates a NEW PagedKVCacheCompressor")
    print("  - The new compressor has fresh state (current_cache_len=0)")
    print("  - So should_compress() always sees seq_len from scratch")
    print("  - State is never properly tracked!")
    print("=" * 80)


if __name__ == "__main__":
    test_state_tracking_bug()
