#!/usr/bin/env python3
"""Example: Using TriAttention Backend with vLLM.

This script demonstrates how to use the TriAttention backend (Plan A)
instead of monkey-patching (Plan B).

Backend Approach Benefits:
- Clean separation of concerns
- Easier to maintain and extend
- Better integration with vLLM's architecture
- Simpler registration mechanism

Usage:
    python examples/example_backend_usage.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --stats-path /path/to/stats.pt \
        --kv-budget 2048
"""

import argparse
from pathlib import Path

import torch


def main():
    parser = argparse.ArgumentParser(description="TriAttention Backend Example")
    parser.add_argument(
        "--model",
        type=str,
        default="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        help="Model name or path",
    )
    parser.add_argument(
        "--stats-path",
        type=Path,
        required=True,
        help="Path to precomputed frequency statistics",
    )
    parser.add_argument(
        "--kv-budget",
        type=int,
        default=2048,
        help="KV cache budget (max tokens to keep)",
    )
    parser.add_argument(
        "--divide-length",
        type=int,
        default=128,
        help="Compression interval (trigger every N tokens)",
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="What is the capital of France?",
        help="Input prompt for generation",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("TriAttention Backend Example")
    print("=" * 80)

    # Step 1: Configure TriAttention
    print("\n[1/4] Configuring TriAttention...")
    from triattention import TriAttentionConfig
    from triattention.backends import setup_triattention, register_triattention_backend

    config = TriAttentionConfig(
        stats_path=args.stats_path,
        kv_budget=args.kv_budget,
        divide_length=args.divide_length,
        pruning_mode="per_head",
        protect_prefill=True,
        sparse_normalize_scores=True,
    )

    # Setup configuration globally
    setup_triattention(config)
    print(f"   Config: kv_budget={config.kv_budget}, divide_length={config.divide_length}")
    print(f"   Stats path: {config.stats_path}")

    # Step 2: Register TriAttention backend
    print("\n[2/4] Registering TriAttention backend with vLLM...")
    register_triattention_backend()
    print("   Backend registered successfully")

    # Step 3: Create vLLM instance
    print("\n[3/4] Initializing vLLM...")
    from vllm import LLM, SamplingParams

    # Note: Must use enforce_eager=True for TriAttention
    # CUDA graphs don't support dynamic KV cache modifications
    llm = LLM(
        model=args.model,
        enforce_eager=True,
        trust_remote_code=True,
        dtype="bfloat16",
        gpu_memory_utilization=0.9,
    )
    print(f"   Model loaded: {args.model}")

    # Step 4: Generate with compression
    print("\n[4/4] Generating with TriAttention compression...")
    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    outputs = llm.generate(
        prompts=[args.prompt],
        sampling_params=sampling_params,
    )

    # Display results
    print("\n" + "=" * 80)
    print("Generation Results")
    print("=" * 80)
    for output in outputs:
        prompt = output.prompt
        generated_text = output.outputs[0].text
        print(f"\nPrompt: {prompt}")
        print(f"\nGenerated:\n{generated_text}")
        print(f"\nTokens generated: {len(output.outputs[0].token_ids)}")

    print("\n" + "=" * 80)
    print("Notes:")
    print("- TriAttention compression is applied automatically during generation")
    print("- Compression happens transparently in the attention backend")
    print("- Check logs above for compression trigger messages")
    print("=" * 80)


if __name__ == "__main__":
    main()
