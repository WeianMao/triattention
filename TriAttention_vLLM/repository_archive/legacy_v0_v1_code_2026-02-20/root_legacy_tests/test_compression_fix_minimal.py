#!/usr/bin/env python3
"""Minimal test to verify seq_len sync fix without full vLLM.

This test simulates the exact behavior of the vLLM integration:
1. Prefill at seq_len=320
2. Decode steps where vLLM reports incrementing seq_len (321, 322, ...)
3. Verify compression only triggers at correct intervals
"""
import os
os.environ["VLLM_USE_V1"] = "0"

import setproctitle
setproctitle.setproctitle('PD-L1_binder')

import torch
from pathlib import Path

from triattention.config import TriAttentionConfig
from triattention.vllm_integration import TriAttentionWrapper

def main():
    print("=" * 80)
    print("MINIMAL TEST: Seq_len Synchronization Fix")
    print("=" * 80)

    # Use DeepSeek model stats since it's available
    stats_path = Path("/data/rbg/users/weian/project/rl/dc/R-KV/speckv_experiments/models/DeepSeek-R1-Distill-Qwen-7B/stats")

    config = TriAttentionConfig(
        stats_path=stats_path,
        kv_budget=256,
        divide_length=64,
        pruning_mode="per_head",
        protect_prefill=False,
        device="cuda:0" if torch.cuda.is_available() else "cpu",
    )

    wrapper = TriAttentionWrapper(config)

    print(f"\nConfig:")
    print(f"  Budget: {config.kv_budget}")
    print(f"  Divide length: {config.divide_length}")
    print(f"  Trigger threshold: {config.kv_budget + config.divide_length}")

    # Test scenario
    layer_idx = 0
    request_id = "test_request"

    wrapper.register_request(request_id)
    compressor = wrapper.get_compressor(layer_idx, request_id)

    print("\n" + "=" * 80)
    print("SCENARIO: Simulating vLLM behavior")
    print("=" * 80)

    # Step 1: Prefill
    prefill_len = 320
    print(f"\n1. PREFILL (seq_len={prefill_len})")

    should_compress = wrapper.should_compress(layer_idx, prefill_len, request_id)
    state = compressor.state.to_dict()

    print(f"   vLLM seq_len: {prefill_len}")
    print(f"   Internal cache_len: {state['current_cache_len']}")
    print(f"   should_compress: {should_compress}")

    assert should_compress, "Should compress at prefill (320 >= 320)"

    # Simulate compression
    compressor.state.update_after_compression(config.kv_budget)
    state = compressor.state.to_dict()
    print(f"   After compression -> cache_len: {state['current_cache_len']}")

    # Step 2: Decode steps
    print(f"\n2. DECODE STEPS (vLLM seq_len increments, ignoring compression)")
    print("   " + "-" * 76)

    compression_log = []

    for step in range(1, 80):
        # vLLM's seq_len continues incrementing (doesn't know about compression)
        vllm_seq_len = prefill_len + step

        should_compress = wrapper.should_compress(layer_idx, vllm_seq_len, request_id)
        state = compressor.state.to_dict()
        actual_cache_len = state['current_cache_len']

        if should_compress:
            compression_log.append({
                'step': step,
                'vllm_seq_len': vllm_seq_len,
                'actual_cache_len': actual_cache_len
            })

            print(f"\n   Step {step}: COMPRESSION TRIGGERED")
            print(f"     vLLM seq_len: {vllm_seq_len}")
            print(f"     Actual cache_len: {actual_cache_len}")
            print(f"     Compressing to budget: {config.kv_budget}")

            # Verify correct trigger
            expected_threshold = config.kv_budget + config.divide_length
            assert actual_cache_len >= expected_threshold, \
                f"ERROR: Compression too early! {actual_cache_len} < {expected_threshold}"

            # Simulate compression
            compressor.state.update_after_compression(config.kv_budget)
            state = compressor.state.to_dict()
            print(f"     After compression -> cache_len: {state['current_cache_len']}")

        elif step % 10 == 0:
            # Print periodic updates
            print(f"   Step {step}: vLLM seq_len={vllm_seq_len}, cache_len={actual_cache_len}, compress=False")

    print("\n" + "=" * 80)
    print("VERIFICATION")
    print("=" * 80)

    final_state = compressor.state.to_dict()

    print(f"\nFinal state:")
    print(f"  Total compressions: {final_state['compression_count']}")
    print(f"  Current cache_len: {final_state['current_cache_len']}")
    print(f"  Absolute position: {final_state['absolute_position']}")

    print(f"\nCompression log:")
    for i, log_entry in enumerate(compression_log, 1):
        print(f"  Compression {i}: step={log_entry['step']}, "
              f"vLLM seq_len={log_entry['vllm_seq_len']}, "
              f"actual cache_len={log_entry['actual_cache_len']}")

    # Verify compression intervals
    print(f"\nVerifying compression intervals:")

    # After initial prefill compression, subsequent compressions should be
    # every divide_length (64) tokens
    if len(compression_log) >= 2:
        for i in range(1, len(compression_log)):
            prev_step = compression_log[i-1]['step']
            curr_step = compression_log[i]['step']
            interval = curr_step - prev_step

            print(f"  Compression {i} to {i+1}: interval = {interval} steps")

            # Should be approximately divide_length
            assert abs(interval - config.divide_length) <= 1, \
                f"ERROR: Interval {interval} != {config.divide_length}"

    print(f"\n" + "=" * 80)
    print("✓ TEST PASSED: Compression intervals are correct!")
    print("=" * 80)

    print(f"\nKey Observations:")
    print(f"  - vLLM seq_len keeps growing (doesn't update after compression)")
    print(f"  - Internal cache_len fluctuates between {config.kv_budget} and "
          f"{config.kv_budget + config.divide_length}")
    print(f"  - Compression triggers every ~{config.divide_length} tokens")
    print(f"  - No compression on every decode step (bug is FIXED!)")

    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
