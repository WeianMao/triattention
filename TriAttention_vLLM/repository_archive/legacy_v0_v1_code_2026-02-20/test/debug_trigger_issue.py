"""Debug script to observe TriAttention compression trigger behavior in vLLM.

Tests the scenario:
- kv_budget=256, divide_length=64
- Expected trigger: 256 + 64 = 320 tokens
- Generate 512 tokens to observe compression
"""
import sys
from pathlib import Path
import torch

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from vllm import LLM, SamplingParams
from triattention import (
    TriAttentionConfig,
    TriAttentionWrapper,
    patch_vllm_attention,
)


def test_compression_trigger():
    """Test that compression triggers at expected threshold."""
    print("\n" + "=" * 80)
    print("Testing TriAttention Compression Trigger")
    print("=" * 80)
    print("\nConfiguration:")
    print("  kv_budget: 256")
    print("  divide_length: 64")
    print("  Expected trigger: 320 tokens")
    print("  Generate: 512 tokens")
    print("\nExpected behavior:")
    print("  - No compression until 320 tokens")
    print("  - First compression at ~320 tokens")
    print("  - Additional compressions every 64 tokens after")
    print("=" * 80 + "\n")

    # Create LLM without TriAttention first
    llm = LLM(
        model="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B",
        dtype="float16",
        gpu_memory_utilization=0.5,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=True,
    )

    # Create TriAttention config
    # Qwen2.5-7B architecture
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

    # Create wrapper and patch
    wrapper = TriAttentionWrapper(config)
    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    patch_vllm_attention(model, wrapper)

    print(f"TriAttention patched: {wrapper._patched}")
    print()

    # Simple prompt
    prompt = "Write a story about a brave knight:"

    # Sampling params to generate 512 tokens
    sampling_params = SamplingParams(
        temperature=0.0,  # Deterministic
        max_tokens=512,
        min_tokens=512,
    )

    print(f"\nPrompt: {prompt}")
    print(f"\nGenerating {sampling_params.max_tokens} tokens...\n")

    # Generate
    outputs = llm.generate([prompt], sampling_params)

    # Print results
    for output in outputs:
        generated_text = output.outputs[0].text
        num_tokens = len(output.outputs[0].token_ids)
        print(f"\n{'=' * 80}")
        print(f"Generation complete:")
        print(f"  Generated tokens: {num_tokens}")
        print(f"  Text length: {len(generated_text)} chars")
        print(f"{'=' * 80}")
        print(f"\nGenerated text (first 500 chars):")
        print(f"{generated_text[:500]}...")


if __name__ == "__main__":
    test_compression_trigger()
