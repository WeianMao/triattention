#!/usr/bin/env python3
"""Debug script to examine forward signature."""
import sys
import inspect
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def inspect_forward():
    if not torch.cuda.is_available():
        print("SKIP: No GPU available")
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
    attn_impl = layers[0].self_attn.attn.impl

    print(f"impl type: {type(attn_impl)}")
    print(f"\nForward method signature:")
    print(inspect.signature(attn_impl.forward))
    
    print(f"\nForward method code:")
    print(f"  co_varnames: {attn_impl.forward.__code__.co_varnames}")
    print(f"  co_argcount: {attn_impl.forward.__code__.co_argcount}")


if __name__ == "__main__":
    inspect_forward()
