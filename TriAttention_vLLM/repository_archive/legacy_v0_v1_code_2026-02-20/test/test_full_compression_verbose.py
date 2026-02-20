#!/usr/bin/env python3
"""Test full compression with Qwen model and real stats - verbose version."""
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
    print("Testing full compression with Qwen model and real stats (VERBOSE)...")

    # Use real stats file for Qwen model
    stats_path = Path("/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt")

    if not stats_path.exists():
        print(f"ERROR: Stats file not found: {stats_path}")
        print("Please ensure the stats file exists before running this test.")
        return

    print(f"Using stats: {stats_path}")

    # Create config with real stats and DEBUG LOGGING
    config = TriAttentionConfig(
        kv_budget=256,  # Keep 256 tokens
        divide_length=64,  # Compress every 64 tokens
        pruning_mode="per_head",
        stats_path=stats_path,
        enable_debug_logging=True,  # Enable verbose logging
    )

    print(f"Config: budget={config.kv_budget}, divide_length={config.divide_length}")
    print(f"Compression triggers when seq_len >= {config.kv_budget + config.divide_length}")
    print("DEBUG LOGGING: ENABLED")

    # Initialize vLLM with Qwen model
    model_path = "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B"
    print(f"\nInitializing vLLM with model: {model_path}")

    llm = LLM(
        model=model_path,
        dtype="float16",  # T4 GPU doesn't support bfloat16
        gpu_memory_utilization=0.95,
        max_model_len=512,  # Reduced to fit in GPU memory
        enforce_eager=True,  # Required for patching
        trust_remote_code=True,
    )

    # Patch attention layers
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

    # Generate text with a longer prompt to reach compression threshold
    print("\nGenerating text...")

    # Create a longer prompt by repeating context
    long_prompt = """Solve this math problem step by step with detailed reasoning.
Question: What is 2 + 2?

Please provide:
1. Initial problem statement
2. Identify the numbers
3. Apply the addition operation
4. Verify the result
5. State the final answer

Think carefully through each step."""

    print(f"Prompt length: {len(long_prompt)} characters")

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=450,  # Generate enough to reach 320+ tokens total
    )

    prompts = [long_prompt]

    outputs = llm.generate(prompts, sampling_params)

    # Check results
    for output in outputs:
        generated = output.outputs[0].text
        prompt_tokens = len(output.prompt_token_ids)
        output_tokens = len(output.outputs[0].token_ids)
        total_tokens = prompt_tokens + output_tokens

        print(f"\n{'='*80}")
        print(f"Token counts:")
        print(f"  Prompt tokens: {prompt_tokens}")
        print(f"  Generated tokens: {output_tokens}")
        print(f"  Total tokens: {total_tokens}")
        print(f"  Compression threshold: {config.kv_budget + config.divide_length}")
        print(f"  Should compress: {total_tokens >= config.kv_budget + config.divide_length}")
        print(f"\n{'='*80}")
        print(f"Generated text ({len(generated)} chars):")
        print(generated[:500] + "..." if len(generated) > 500 else generated)

    # Check compression stats
    print(f"\n{'='*80}")
    print("Compression Stats:")

    total_requests = len(wrapper.request_compressors)
    total_compressors = sum(
        len(layer_compressors)
        for layer_compressors in wrapper.request_compressors.values()
    )

    print(f"  Active requests: {total_requests}")
    print(f"  Compressor instances: {total_compressors}")

    # Check compression counts
    total_compressions = 0
    for request_id, layer_compressors in wrapper.request_compressors.items():
        for layer_idx, compressor in layer_compressors.items():
            total_compressions += compressor.state.compression_count

    print(f"  Total compressions performed: {total_compressions}")

    if wrapper._patched:
        print("\n✅ SUCCESS: TriAttention hook is active!")
    else:
        print("\n❌ FAILED: TriAttention hook not active")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
