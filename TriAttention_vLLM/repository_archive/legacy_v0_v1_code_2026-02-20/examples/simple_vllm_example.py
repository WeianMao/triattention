#!/usr/bin/env python3
"""Simple working example of vLLM integration with TriAttention.

This example demonstrates the basic integration that currently works.
Note: Requires enforce_eager=True until CUDA graph issue is resolved.

Usage:
    conda activate trivllm
    python examples/simple_vllm_example.py
"""
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vllm import LLM, SamplingParams
from triattention import (
    TriAttentionConfig,
    TriAttentionWrapper,
    patch_vllm_attention,
)


def main():
    print("=" * 80)
    print("TriAttention vLLM Integration Example")
    print("=" * 80)

    # Step 1: Create TriAttention configuration with real stats
    stats_path = Path("/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt")
    config = TriAttentionConfig(
        kv_budget=256,  # Keep 256 tokens
        divide_length=64,  # Compress every 64 tokens
        pruning_mode="per_head",
        stats_path=stats_path,
    )

    print("\nConfiguration:")
    print(f"  KV Budget: {config.kv_budget}")
    print(f"  Divide Length: {config.divide_length}")
    print(f"  Compression Trigger: {config.kv_budget + config.divide_length}")
    print(f"  Pruning Mode: {config.pruning_mode}")

    # Step 2: Initialize vLLM engine with Qwen model
    print("\nInitializing vLLM engine with Qwen model...")
    print("  NOTE: Using enforce_eager=True (required for now)")
    llm = LLM(
        model="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B",
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Step 3: Patch vLLM attention layers
    print("\nPatching vLLM attention layers...")
    model_runner = llm.llm_engine.model_executor.driver_worker.model_runner
    model = model_runner.model
    model_config = model_runner.model_config
    cache_config = llm.llm_engine.cache_config

    wrapper = TriAttentionWrapper(config)
    patch_vllm_attention(
        model,
        wrapper,
        model_config=model_config,
        cache_config=cache_config,
    )

    print(f"  Patched: {wrapper._patched}")
    print(f"  Active requests: {len(wrapper.get_active_requests())}")

    # Step 4: Run inference
    print("\nRunning inference...")
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=100,
    )

    prompts = [
        "The quick brown fox",
        "Once upon a time",
    ]

    print(f"  Prompts: {len(prompts)}")
    print(f"  Max tokens: {sampling_params.max_tokens}")

    outputs = llm.generate(prompts, sampling_params)

    # Step 5: Display results
    print("\n" + "=" * 80)
    print("Results")
    print("=" * 80)

    for idx, output in enumerate(outputs):
        prompt = output.prompt
        generated = output.outputs[0].text
        print(f"\nPrompt {idx + 1}: {prompt}")
        print(f"Generated: {generated[:100]}...")
        print(f"  Length: {len(generated)} characters")

    # Step 6: Check compression status
    print("\n" + "=" * 80)
    print("Compression Status")
    print("=" * 80)

    total_compressors = sum(
        len(layer_compressors)
        for layer_compressors in wrapper.request_compressors.values()
    )
    print(f"Total compressor instances created: {total_compressors}")

    if total_compressors > 0:
        print("\nNote: Compressors created but compression may be skipped")
        print("      due to KV cache format mismatch (work in progress)")
    else:
        print("\nNote: No compressors created (sequence too short or")
        print("      compression not triggered)")

    print("\n" + "=" * 80)
    print("Example Complete")
    print("=" * 80)


if __name__ == "__main__":
    main()
