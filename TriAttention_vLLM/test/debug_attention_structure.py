#!/usr/bin/env python3
"""Debug script to examine vLLM attention structure."""
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def inspect_attention():
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
    print(f"\nModel type: {type(model)}")
    print(f"Model attributes: {[attr for attr in dir(model) if not attr.startswith('_')]}")

    # Navigate to layers
    if hasattr(model, "model"):
        if hasattr(model.model, "layers"):
            layers = model.model.layers
            print(f"\nFound {len(layers)} decoder layers")

            # Examine first layer
            layer0 = layers[0]
            print(f"\nLayer 0 type: {type(layer0)}")
            print(f"Layer 0 attributes: {[attr for attr in dir(layer0) if not attr.startswith('_')]}")

            if hasattr(layer0, "self_attn"):
                attn = layer0.self_attn
                print(f"\nself_attn type: {type(attn)}")
                print(f"self_attn attributes: {[attr for attr in dir(attn) if not attr.startswith('_')]}")

                # Check for impl
                if hasattr(attn, "impl"):
                    impl = attn.impl
                    print(f"\nimpl type: {type(impl)}")
                    print(f"impl attributes: {[attr for attr in dir(impl) if not attr.startswith('_')]}")
                    print(f"\nimpl.forward signature: {impl.forward.__code__.co_varnames[:impl.forward.__code__.co_argcount]}")
                else:
                    print("\nNO 'impl' attribute found in self_attn!")

                # Check for attn.attn pattern
                if hasattr(attn, "attn"):
                    print(f"\nattn.attn type: {type(attn.attn)}")
                    print(f"attn.attn attributes: {[attr for attr in dir(attn.attn) if not attr.startswith('_')]}")
                    if hasattr(attn.attn, "impl"):
                        print(f"attn.attn.impl type: {type(attn.attn.impl)}")

                # Check what methods exist
                print(f"\nMethods on self_attn:")
                for attr in dir(attn):
                    if callable(getattr(attn, attr)) and not attr.startswith('_'):
                        print(f"  - {attr}")


if __name__ == "__main__":
    inspect_attention()
