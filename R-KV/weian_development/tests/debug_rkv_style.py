"""
Debug script to trace RKV-style SpeckV behavior.

This script helps identify where the issue occurs by adding logging
at key points in the forward pass.
"""
from __future__ import annotations
import sys
from pathlib import Path

# Add paths
RKV_ROOT = Path(__file__).resolve().parents[2]
HF_RKV_ROOT = RKV_ROOT / "HuggingFace"
for path in (HF_RKV_ROOT, RKV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import DynamicCache


def debug_forward_hook(comp, name=""):
    """Create a debug hook to trace state changes."""
    def hook(model, args, kwargs, output):
        print(f"\n[{name}] Forward called")
        print(f"  absolute_position: {comp.absolute_position}")
        print(f"  cache_positions len: {len(comp.cache_positions)}")
        print(f"  prefix_length: {comp.prefix_length}")

        # Check input past_key_values
        pkv = kwargs.get('past_key_values')
        if pkv is None:
            print(f"  input past_key_values: None")
        elif isinstance(pkv, DynamicCache):
            print(f"  input past_key_values: DynamicCache, seq_len={pkv.get_seq_length()}")
        else:
            print(f"  input past_key_values: {type(pkv)}")

        # Check output
        if hasattr(output, 'past_key_values'):
            out_pkv = output.past_key_values
            if isinstance(out_pkv, DynamicCache):
                print(f"  output past_key_values: DynamicCache, seq_len={out_pkv.get_seq_length()}")
            elif out_pkv is not None:
                print(f"  output past_key_values: tuple, seq_len={out_pkv[0][0].shape[2]}")

        return output
    return hook


def test_basic_generation():
    """Test basic generation to see if the issue is reproducible."""
    print("="*70)
    print("Testing RKV-style SpeckV generation")
    print("="*70)

    # Use a small model for testing
    model_path = "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B"
    stats_path = RKV_ROOT / "outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt"

    if not Path(model_path).exists():
        print(f"Model not found: {model_path}")
        return

    if not stats_path.exists():
        print(f"Stats not found: {stats_path}")
        return

    print(f"Loading model from {model_path}...")

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        use_cache=True,
        attn_implementation="flash_attention_2",
    )
    model.eval()

    # Apply RKV-style patch
    from weian_development.speckv.speckv_rkv_style import apply_speckv_rkv_style_patch

    metadata_expectations = {
        "prompt_template": "plain",
        "use_chat_template": False,
        "system_prompt": "",
        "attn_implementation": "flash_attention_2",
        "dtype": "bfloat16",
        "kv_budget": 2048,
    }

    apply_speckv_rkv_style_patch(
        model,
        stats_path=stats_path,
        model_path=Path(model_path),
        kv_budget=2048,
        offset_max_length=65536,
        score_aggregation="mean",
        sparse_seed=0,
        head_limit=None,
        metadata_expectations=metadata_expectations,
        normalize_scores=True,
        use_rank_aggregation=False,
        include_prefill_in_budget=True,
        divide_length=128,
    )

    # Get the compressor
    comp = model._speckv_rkv_compressor

    # Simple prompt
    prompt = "What is 2 + 2? Think step by step."
    print(f"\nPrompt: {prompt}")

    inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
    print(f"Input token count: {inputs['input_ids'].shape[1]}")

    # Manual generation to trace each step
    print("\n--- Starting generation ---")

    past_key_values = None
    input_ids = inputs['input_ids']
    generated_tokens = []

    for step in range(20):  # Generate 20 tokens
        print(f"\n=== Step {step} ===")
        print(f"  input_ids shape: {input_ids.shape}")
        print(f"  comp.absolute_position (before forward): {comp.absolute_position}")
        print(f"  comp.cache_positions len (before forward): {len(comp.cache_positions)}")

        if past_key_values is not None:
            if isinstance(past_key_values, DynamicCache):
                print(f"  past_key_values type: DynamicCache, seq_len={past_key_values.get_seq_length()}")
            else:
                print(f"  past_key_values type: {type(past_key_values)}")
        else:
            print(f"  past_key_values: None")

        with torch.no_grad():
            outputs = model(
                input_ids=input_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )

        print(f"  comp.absolute_position (after forward): {comp.absolute_position}")
        print(f"  comp.cache_positions len (after forward): {len(comp.cache_positions)}")

        # Get next token
        next_token_logits = outputs.logits[:, -1, :]
        next_token = torch.argmax(next_token_logits, dim=-1)

        decoded = tokenizer.decode(next_token[0])
        print(f"  Generated token: '{decoded}' (id={next_token[0].item()})")

        generated_tokens.append(next_token[0].item())

        # Check for repetition
        if len(generated_tokens) >= 3:
            if generated_tokens[-1] == generated_tokens[-2] == generated_tokens[-3]:
                print(f"\n!!! WARNING: Token repetition detected !!!")

        # Update for next step
        past_key_values = outputs.past_key_values
        input_ids = next_token.unsqueeze(0)

        # Check output cache
        if isinstance(past_key_values, DynamicCache):
            print(f"  output past_key_values: DynamicCache, seq_len={past_key_values.get_seq_length()}")
        elif past_key_values is not None:
            print(f"  output past_key_values: tuple, seq_len={past_key_values[0][0].shape[2]}")

        # Check if cache_positions matches cache length
        if isinstance(past_key_values, DynamicCache):
            cache_len = past_key_values.get_seq_length()
        else:
            cache_len = past_key_values[0][0].shape[2]

        if len(comp.cache_positions) != cache_len:
            print(f"\n!!! ERROR: cache_positions ({len(comp.cache_positions)}) != cache_len ({cache_len}) !!!")
            break

    print("\n--- Generation complete ---")
    full_output = tokenizer.decode(generated_tokens)
    print(f"Generated: {full_output}")


if __name__ == "__main__":
    test_basic_generation()
