#!/usr/bin/env python3
"""Test vLLM + TriAttention initialization without running inference.

This test validates that:
1. vLLM engine can be created
2. Model can be loaded
3. TriAttention patching works
4. No runtime errors during setup
"""
import sys
from pathlib import Path

# Add project root to path
TRIATTENTION_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(TRIATTENTION_ROOT))

from vllm import LLM
from triattention import TriAttentionConfig, TriAttentionWrapper, patch_vllm_attention


def main():
    print("=" * 60)
    print("TriAttention + vLLM Initialization Test")
    print("=" * 60)

    # Paths
    model_path = "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B"
    stats_path = TRIATTENTION_ROOT.parent / "R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt"

    print(f"\n[1/4] Creating TriAttention configuration...")
    config = TriAttentionConfig(
        stats_path=stats_path,
        kv_budget=2048,
        divide_length=128,
        sparse_round_window=32,
        pruning_mode="per_head",
        score_aggregation="mean",
        offset_max_length=65536,
        sparse_normalize_scores=True,
        window_size=128,
        include_prefill_in_budget=True,
        protect_prefill=False,
    )
    print("✓ Configuration created")

    print(f"\n[2/4] Initializing vLLM engine...")
    print(f"  Model: {model_path}")
    llm = LLM(
        model=model_path,
        dtype="bfloat16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.9,
        trust_remote_code=True,
        seed=888,
        max_model_len=4096,  # Use smaller length for initialization test
    )
    print("✓ vLLM engine initialized")

    print(f"\n[3/4] Creating TriAttention wrapper...")
    tri_wrapper = TriAttentionWrapper(config)
    print("✓ Wrapper created")

    print(f"\n[4/4] Patching vLLM attention layers...")
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        patch_vllm_attention(model, tri_wrapper)

        if tri_wrapper._patched:
            print("✓ Patching successful")
            print(f"  Compression: ENABLED")
        else:
            print("✗ Patching failed")
            print(f"  Compression: DISABLED")
            return 1
    except Exception as e:
        print(f"✗ Patching failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

    print("\n" + "=" * 60)
    print("✅ All initialization checks passed!")
    print("✅ System ready for inference")
    print("=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
