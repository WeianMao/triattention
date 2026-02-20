#!/usr/bin/env python3
"""Test vLLM integration to verify compression intervals are correct."""
import os
os.environ["VLLM_USE_V1"] = "0"

import setproctitle
setproctitle.setproctitle('PD-L1_binder')

import torch
from pathlib import Path
from vllm import LLM, SamplingParams

# Import TriAttention
from triattention.config import TriAttentionConfig
from triattention.vllm_integration import create_triattention_wrapper, patch_vllm_attention

def test_vllm_compression_intervals():
    """Test compression intervals with actual vLLM integration."""
    print("=" * 80)
    print("Testing vLLM Integration - Compression Intervals")
    print("=" * 80)

    # Configuration
    model_name = "/data/rbg/users/weian/hf_models/Qwen2.5-7B-Instruct"
    stats_path = "/data/rbg/users/weian/project/rl/R-KV/speckv_experiments/models/Qwen2.5-7B-Instruct/stats"

    kv_budget = 256
    divide_length = 64

    print(f"\nConfiguration:")
    print(f"  Model: {model_name}")
    print(f"  Stats: {stats_path}")
    print(f"  KV Budget: {kv_budget}")
    print(f"  Divide Length: {divide_length}")
    print(f"  Expected trigger threshold: {kv_budget + divide_length}")

    # Create LLM
    print("\nInitializing vLLM...")
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=0.8,
        max_model_len=2048,
        tensor_parallel_size=1,
    )

    # Create and patch TriAttention
    print("\nPatching TriAttention...")
    tri_wrapper = create_triattention_wrapper(
        stats_path=stats_path,
        kv_budget=kv_budget,
        divide_length=divide_length,
        pruning_mode="per_head",
        protect_prefill=False,
    )

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    patch_vllm_attention(model, tri_wrapper)

    # Test prompt - generate long output to trigger multiple compressions
    prompt = "Write a detailed explanation of quantum computing, covering the following topics: superposition, entanglement, quantum gates, quantum algorithms like Shor's algorithm and Grover's algorithm, error correction, and current applications. Make your response comprehensive and detailed."

    print(f"\nPrompt length: {len(prompt.split())} words")
    print(f"\nGenerating with max_tokens=200 to trigger compressions...")

    # Sampling parameters for long generation
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=200,
        seed=42,
    )

    # Monitor compression events
    print("\n" + "=" * 80)
    print("Starting generation (watch for compression logs):")
    print("=" * 80 + "\n")

    # Generate
    outputs = llm.generate([prompt], sampling_params)

    print("\n" + "=" * 80)
    print("Generation complete!")
    print("=" * 80)

    # Print output
    generated_text = outputs[0].outputs[0].text
    print(f"\nGenerated text ({len(generated_text)} chars):")
    print("-" * 80)
    print(generated_text[:500] + "..." if len(generated_text) > 500 else generated_text)
    print("-" * 80)

    # Verify compression happened
    request_id = f"decode_0"  # Default request ID used in vLLM integration
    state_summary = tri_wrapper.get_request_state_summary(request_id)

    if state_summary:
        print("\n" + "=" * 80)
        print("Compression State Summary (Layer 0):")
        print("=" * 80)
        if 0 in state_summary:
            layer_0_state = state_summary[0]
            print(f"  Compression count: {layer_0_state['compression_count']}")
            print(f"  Current cache len: {layer_0_state['current_cache_len']}")
            print(f"  Absolute position: {layer_0_state['absolute_position']}")
            print(f"  Prefill length: {layer_0_state['prefill_length']}")

            if layer_0_state['compression_count'] >= 2:
                print("\n✓ SUCCESS: Multiple compressions observed")
                print(f"  Compressions triggered at correct intervals!")
            else:
                print(f"\n⚠ WARNING: Only {layer_0_state['compression_count']} compression(s)")
                print("  May need longer generation for multiple compressions")
    else:
        print("\n⚠ WARNING: No state found for request")
        print("  Compression may not have been triggered")

    print("\n" + "=" * 80)
    print("TEST COMPLETE")
    print("=" * 80)

    print("\nExpected behavior:")
    print(f"  - First compression at seq_len={kv_budget + divide_length} (prefill)")
    print(f"  - Subsequent compressions every {divide_length} tokens")
    print(f"  - Cache fluctuates between {kv_budget} and {kv_budget + divide_length}")
    print("\nCheck the logs above for '[TriAttention] Compressing' messages")
    print("They should show seq_len at correct intervals, not every step!")

if __name__ == "__main__":
    test_vllm_compression_intervals()
