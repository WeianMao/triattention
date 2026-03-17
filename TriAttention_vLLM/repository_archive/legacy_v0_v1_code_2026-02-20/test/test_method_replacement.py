#!/usr/bin/env python3
"""Test if method replacement actually works."""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from triattention import (
    TriAttentionConfig,
    TriAttentionWrapper,
    patch_vllm_attention,
)


def test_method_replacement():
    if not torch.cuda.is_available():
        print("SKIP: No GPU")
        return

    from vllm import LLM

    print("Loading Qwen model...")
    llm = LLM(
        model="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B",
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=True,
    )

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    layers = model.model.layers

    # Check original forward
    orig_impl = layers[0].self_attn.attn.impl
    orig_forward = orig_impl.forward
    print(f"\nOriginal forward: {orig_forward}")
    print(f"  Type: {type(orig_forward)}")
    print(f"  Name: {orig_forward.__name__ if hasattr(orig_forward, '__name__') else 'N/A'}")

    # Now patch
    stats_path = Path("/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt")
    config = TriAttentionConfig(
        kv_budget=64,
        divide_length=16,
        pruning_mode="per_head",
        stats_path=stats_path,
    )
    wrapper = TriAttentionWrapper(config)
    patch_vllm_attention(model, wrapper)

    # Check after patching
    patched_impl = layers[0].self_attn.attn.impl
    patched_forward = patched_impl.forward
    print(f"\nPatched forward: {patched_forward}")
    print(f"  Type: {type(patched_forward)}")
    print(f"  Name: {patched_forward.__name__ if hasattr(patched_forward, '__name__') else 'N/A'}")

    print(f"\nAre they the same? {orig_forward is patched_forward}")
    print(f"Same impl object? {orig_impl is patched_impl}")


if __name__ == "__main__":
    test_method_replacement()
