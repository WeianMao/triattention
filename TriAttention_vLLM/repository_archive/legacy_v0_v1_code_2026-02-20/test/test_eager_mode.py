#!/usr/bin/env python3
"""Test compression with eager mode (no CUDA graphs)."""
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


def test_eager_mode():
    if not torch.cuda.is_available():
        print("SKIP: No GPU")
        return

    from vllm import LLM, SamplingParams

    hook_calls = []
    import triattention.vllm_integration as vllm_int
    original_apply = vllm_int._apply_triattention_compression

    def debug_apply(kv_cache, attn_metadata, tri_wrapper, layer_idx):
        hook_calls.append(layer_idx)
        print(f"[HOOK] Layer {layer_idx} called!")
        return original_apply(kv_cache, attn_metadata, tri_wrapper, layer_idx)

    vllm_int._apply_triattention_compression = debug_apply

    stats_path = Path("/data/rbg/users/weian/project/rl/dc/R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt")
    config = TriAttentionConfig(
        kv_budget=64,
        divide_length=16,
        pruning_mode="per_head",
        stats_path=stats_path,
    )

    wrapper = TriAttentionWrapper(config)

    print("\nLoading Qwen model with EAGER MODE (no CUDA graphs)...")
    llm = LLM(
        model="/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B",
        dtype="float16",
        gpu_memory_utilization=0.9,
        max_model_len=1024,
        enforce_eager=True,
        trust_remote_code=True,
    )

    model = llm.llm_engine.model_executor.driver_worker.model_runner.model
    patch_vllm_attention(model, wrapper)

    print("\nRunning generation...")
    sampling_params = SamplingParams(temperature=0.0, max_tokens=50)
    prompt = " ".join([f"word{i}" for i in range(50)])

    outputs = llm.generate([prompt], sampling_params)

    print(f"\n\nGeneration complete!")
    print(f"Total hook calls: {len(hook_calls)}")
    if hook_calls:
        print(f"Unique layers called: {set(hook_calls)}")
    else:
        print("WARNING: Hook was NEVER called!")


if __name__ == "__main__":
    test_eager_mode()
