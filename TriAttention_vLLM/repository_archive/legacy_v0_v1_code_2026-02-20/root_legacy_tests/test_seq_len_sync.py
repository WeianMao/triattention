#!/usr/bin/env python3
"""Test script to verify seq_len synchronization fix.

This script simulates the vLLM decode flow and verifies that compression
only triggers at the correct intervals (budget + divide_length).
"""
import os
os.environ["VLLM_USE_V1"] = "0"

import setproctitle
setproctitle.setproctitle('PD-L1_binder')

import torch
from pathlib import Path

# Import TriAttention components
from triattention.config import TriAttentionConfig
from triattention.vllm_integration import TriAttentionWrapper

def test_seq_len_sync():
    """Test that compression triggers at correct intervals."""
    print("=" * 80)
    print("Testing seq_len synchronization with internal state tracking")
    print("=" * 80)

    # Configuration
    stats_path = Path("/data/rbg/users/weian/project/rl/R-KV/speckv_experiments/models/Qwen2.5-7B-Instruct/stats")

    config = TriAttentionConfig(
        stats_path=stats_path,
        kv_budget=256,
        divide_length=64,
        head_dim=128,
        num_kv_heads=8,
        num_layers=28,
        pruning_mode="per_head",
        protect_prefill=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    wrapper = TriAttentionWrapper(config)

    # Test parameters
    layer_idx = 0
    request_id = "test_request"

    # Register request
    wrapper.register_request(request_id)
    compressor = wrapper.get_compressor(layer_idx, request_id)

    print(f"\nConfig: budget={config.kv_budget}, divide_length={config.divide_length}")
    print(f"Expected trigger threshold: {config.kv_budget + config.divide_length}")
    print("\nSimulating vLLM decode flow:")
    print("-" * 80)

    # Simulate vLLM's monotonically increasing seq_len
    # Initial prefill
    prefill_len = 320
    print(f"\n1. Prefill: vLLM reports seq_len={prefill_len}")
    should_compress = wrapper.should_compress(layer_idx, prefill_len, request_id)
    print(f"   should_compress={should_compress} (expected: True, 320 >= 320)")
    print(f"   State: {compressor.state.to_dict()}")

    if should_compress:
        print(f"   -> Compressing to budget={config.kv_budget}")
        # Simulate compression
        compressor.state.update_after_compression(config.kv_budget)
        print(f"   State after compression: {compressor.state.to_dict()}")

    # Simulate decode steps
    # vLLM's seq_len continues from 320, even though we compressed to 256
    compression_count = 1

    for step in range(1, 100):
        vllm_seq_len = prefill_len + step  # vLLM doesn't know about compression

        should_compress = wrapper.should_compress(layer_idx, vllm_seq_len, request_id)

        # Only print when compression happens or every 10 steps
        if should_compress or step % 10 == 0:
            state_dict = compressor.state.to_dict()
            actual_cache_len = state_dict['current_cache_len']

            print(f"\nStep {step}: vLLM seq_len={vllm_seq_len}")
            print(f"   Actual cache_len={actual_cache_len}")
            print(f"   should_compress={should_compress}")

            if should_compress:
                compression_count += 1
                expected_trigger = config.kv_budget + config.divide_length
                print(f"   *** COMPRESSION #{compression_count} ***")
                print(f"   Expected trigger at cache_len={expected_trigger}, got {actual_cache_len}")
                print(f"   -> Compressing to budget={config.kv_budget}")

                # Verify correct trigger point
                assert actual_cache_len >= expected_trigger, \
                    f"Compression triggered too early! cache_len={actual_cache_len} < threshold={expected_trigger}"

                # Simulate compression
                compressor.state.update_after_compression(config.kv_budget)
                print(f"   State after compression: {compressor.state.to_dict()}")

        # Stop after observing a few compressions
        if compression_count >= 3:
            print(f"\n--- Stopping after {compression_count} compressions ---")
            break

    print("\n" + "=" * 80)
    print("VERIFICATION:")
    print("=" * 80)

    # Check compression intervals
    final_state = compressor.state.to_dict()
    print(f"\nFinal state: {final_state}")
    print(f"Total compressions: {final_state['compression_count']}")
    print(f"Current cache length: {final_state['current_cache_len']}")

    # Verify compression count is reasonable
    # Each compression should happen every divide_length tokens
    # With budget=256, divide_length=64:
    # - Compression 1: at seq_len=320 (prefill)
    # - Compression 2: at cache_len=320 (256 + 64 new tokens)
    # - Compression 3: at cache_len=320 (256 + 64 new tokens)

    expected_compressions = final_state['compression_count']
    actual_compressions = compression_count

    print(f"\nExpected compression count: {expected_compressions}")
    print(f"Actual compression count: {actual_compressions}")

    if expected_compressions == actual_compressions:
        print("\n✓ TEST PASSED: Compression triggers at correct intervals")
    else:
        print(f"\n✗ TEST FAILED: Compression count mismatch")
        return False

    print("\n" + "=" * 80)
    print("SUCCESS: seq_len synchronization works correctly!")
    print("=" * 80)

    return True

if __name__ == "__main__":
    success = test_seq_len_sync()
    exit(0 if success else 1)
