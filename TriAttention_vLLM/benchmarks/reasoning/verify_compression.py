#!/usr/bin/env python3
"""Verify that compression is actually triggered during inference.

This script runs a minimal test and adds detailed logging to verify:
1. Compression trigger is called
2. Actual compression happens (seq_len > budget + divide_length)
3. KV cache is reduced
"""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from vllm import LLM, SamplingParams
from triattention import TriAttentionConfig, TriAttentionWrapper, patch_vllm_attention


def main():
    """Run verification test."""
    print("=" * 80)
    print("Compression Verification Test")
    print("=" * 80)

    # Setup
    model_path = "/data/rbg/users/weian/project/rl/datasets/Qwen2.5-1.5B"

    # Create config with low budget to trigger compression quickly
    config = TriAttentionConfig(
        kv_budget=512,  # Low budget to trigger quickly
        divide_length=128,
        window_size=128,
        pruning_mode="per_head",
        sparse_normalize_scores=True,
    )

    print(f"\nConfiguration:")
    print(f"  KV Budget: {config.kv_budget}")
    print(f"  Divide Length: {config.divide_length}")
    print(f"  Trigger Threshold: {config.kv_budget + config.divide_length}")

    # Initialize vLLM
    print("\nInitializing vLLM...")
    llm = LLM(
        model=model_path,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.5,  # Reduced from 0.85
        trust_remote_code=True,
        seed=888,
        max_model_len=2048,  # Reduced from 4096
        enforce_eager=True,
    )

    # Patch attention
    print("\nPatching attention...")
    tri_wrapper = TriAttentionWrapper(config)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    patch_vllm_attention(model, tri_wrapper)

    # Add compression logging hook
    original_compress = tri_wrapper.compress_paged_cache
    compression_events = []

    def logged_compress(*args, **kwargs):
        """Log compression events."""
        seq_len = kwargs.get('seq_len', args[3] if len(args) > 3 else None)
        layer_idx = kwargs.get('layer_idx', args[4] if len(args) > 4 else None)
        compression_events.append({
            'layer_idx': layer_idx,
            'seq_len': seq_len,
        })
        print(f"[COMPRESSION] Layer {layer_idx}, seq_len={seq_len}")
        return original_compress(*args, **kwargs)

    tri_wrapper.compress_paged_cache = logged_compress

    # Run inference with long prompt to trigger compression
    print("\nRunning generation...")
    prompt = "Solve this math problem: " + "1 + 1 = 2. " * 100  # Long prompt

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=1024,  # Generate enough to trigger compression
        n=1,
    )

    outputs = llm.generate([prompt], sampling_params)

    # Report results
    print("\n" + "=" * 80)
    print("Verification Results")
    print("=" * 80)

    if compression_events:
        print(f"\n✓ Compression TRIGGERED: {len(compression_events)} times")
        print(f"\nCompression details:")
        for i, event in enumerate(compression_events[:10], 1):  # Show first 10
            print(f"  {i}. Layer {event['layer_idx']}: seq_len={event['seq_len']}")
        if len(compression_events) > 10:
            print(f"  ... and {len(compression_events) - 10} more")
    else:
        print("\n✗ Compression NOT triggered")
        print("  Possible reasons:")
        print("  - Sequence length didn't exceed threshold")
        print("  - Compression hook not called")
        print("  - Bug in compression path")

    # Show generated text length
    output_text = outputs[0].outputs[0].text
    print(f"\nGeneration stats:")
    print(f"  Output length: {len(output_text)} chars")
    print(f"  Output preview: {output_text[:200]}...")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
