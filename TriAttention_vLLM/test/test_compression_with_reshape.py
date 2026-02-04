#!/usr/bin/env python3
"""Test that compression actually triggers with the fixed reshape logic."""
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vllm import LLM, SamplingParams
from triattention import (
    TriAttentionConfig,
    TriAttentionWrapper,
    patch_vllm_attention,
)


def main():
    print("Testing compression with reshape fix...")

    # Create config with real stats
    stats_path = Path("/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt")
    config = TriAttentionConfig(
        kv_budget=64,  # Keep only 64 tokens
        divide_length=16,  # Compress every 16 tokens
        pruning_mode="per_head",
        stats_path=stats_path,
    )

    print(f"Config: budget={config.kv_budget}, divide_length={config.divide_length}")
    print(f"Compression will trigger when seq_len >= {config.kv_budget + config.divide_length}")

    # Initialize vLLM with Qwen model
    print("\nInitializing vLLM with Qwen model...")
    llm = LLM(
        model="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B",
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Patch
    print("\nPatching attention layers...")
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

    # Generate with sufficient length to trigger compression
    print("\nGenerating text (target >80 tokens to trigger compression)...")
    sampling_params = SamplingParams(
        temperature=0.8,
        top_p=0.95,
        max_tokens=150,  # Generate enough to exceed threshold
    )

    prompts = [
        "The quick brown fox jumps over the lazy dog. This is a test prompt that should be long enough.",
    ]

    print(f"Prompt length: {len(prompts[0].split())} words")

    outputs = llm.generate(prompts, sampling_params)

    # Check results
    for output in outputs:
        generated = output.outputs[0].text
        print(f"\nGenerated {len(generated.split())} words")
        print(f"Total tokens (estimated): {len(output.prompt.split()) + len(generated.split())}")

    # Check if compression happened
    total_compressors = sum(
        len(layer_compressors)
        for layer_compressors in wrapper.request_compressors.values()
    )
    print(f"\n{'='*80}")
    print(f"Compressor instances created: {total_compressors}")

    if total_compressors > 0:
        print("\n✅ SUCCESS: Compression was triggered!")
        print("Check logs above for 'Applying compression' and 'Compression complete' messages")
    else:
        print("\n⚠️  WARNING: No compressors created (sequence may be too short)")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
