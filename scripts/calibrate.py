#!/usr/bin/env python3
"""Calibrate frequency-domain statistics for TriAttention.

Runs a single forward pass on plain text input, hooks into every attention
layer to capture pre-RoPE query states, and computes per-head frequency
statistics.  The resulting .pt file can be loaded directly by
``triattention.pruning_utils.load_head_frequency_stats``.

Usage
-----
    python scripts/calibrate.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
        --input calibration_text.txt \
        --output calibration/qwen7b_stats.pt \
        --max-length 32768 \
        --device cuda
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


# ---------------------------------------------------------------------------
# Helpers imported from triattention (kept local to make script self-contained)
# ---------------------------------------------------------------------------

def _determine_rope_style(config: AutoConfig) -> str:
    model_type = getattr(config, "model_type", "")
    if "llama" in model_type:
        return "half"
    return "half"


def _to_complex_pairs(tensor: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        real = tensor_real[..., ::2].contiguous()
        imag = tensor_real[..., 1::2].contiguous()
        return torch.complex(real, imag)
    freq_count = tensor.shape[-1] // 2
    real = tensor_real[..., :freq_count].contiguous()
    imag = tensor_real[..., freq_count:].contiguous()
    return torch.complex(real, imag)


# ---------------------------------------------------------------------------
# Main calibration logic
# ---------------------------------------------------------------------------

def _find_attention_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Return the list of attention sub-modules in layer order."""
    layers = []
    # Common HF naming: model.model.layers[i].self_attn
    backbone = getattr(model, "model", model)
    layer_list = getattr(backbone, "layers", None)
    if layer_list is None:
        raise RuntimeError(
            "Cannot locate transformer layers. Expected model.model.layers."
        )
    for layer_module in layer_list:
        attn = getattr(layer_module, "self_attn", None)
        if attn is None:
            raise RuntimeError("Layer missing self_attn attribute.")
        layers.append(attn)
    return layers


def calibrate(
    model_name_or_path: str,
    input_path: str,
    output_path: str,
    max_length: int = 32768,
    device: str = "cuda",
    attn_implementation: str = "flash_attention_2",
) -> None:
    device_obj = torch.device(device)
    dtype = torch.bfloat16

    # --- Load config, tokenizer, model ---
    print(f"Loading model: {model_name_or_path}", file=sys.stderr)
    config = AutoConfig.from_pretrained(model_name_or_path, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name_or_path,
        torch_dtype=dtype,
        device_map=device,
        attn_implementation=attn_implementation,
        trust_remote_code=True,
    )
    model.eval()

    num_layers = config.num_hidden_layers
    num_heads = config.num_attention_heads
    head_dim = getattr(config, "head_dim", config.hidden_size // num_heads)
    num_kv_heads = getattr(config, "num_key_value_heads", num_heads)
    rope_style = _determine_rope_style(config)

    # --- Locate attention layers ---
    attn_layers = _find_attention_layers(model)

    # --- Read and tokenize input ---
    print(f"Reading input: {input_path}", file=sys.stderr)
    text = Path(input_path).read_text(encoding="utf-8")
    input_ids = tokenizer.encode(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = input_ids.to(device_obj)
    seq_len = input_ids.shape[1]
    print(f"Tokenized length: {seq_len}", file=sys.stderr)

    # --- Register hooks to capture pre-RoPE Q ---
    captured_q: Dict[int, torch.Tensor] = {}

    def _make_pre_hook(layer_idx: int):
        def hook_fn(module, args, kwargs):
            hidden_states = args[0] if args else kwargs.get("hidden_states")
            if hidden_states is None:
                return
            # Compute pre-RoPE Q directly (including optional QK-norm, e.g. Qwen3)
            attn = module
            bsz, q_len, _ = hidden_states.shape
            q = attn.q_proj(hidden_states).view(bsz, q_len, num_heads, head_dim)
            q_norm = getattr(attn, "q_norm", None)
            if q_norm is not None:
                q = q_norm(q)
            q = q.transpose(1, 2)
            captured_q[layer_idx] = q.detach().cpu()
        return hook_fn

    handles = []
    for layer_idx, attn in enumerate(attn_layers):
        h = attn.register_forward_pre_hook(_make_pre_hook(layer_idx), with_kwargs=True)
        handles.append(h)

    # --- Forward pass ---
    print("Running forward pass...", file=sys.stderr)
    with torch.no_grad():
        model(input_ids)
    print("Forward pass complete.", file=sys.stderr)

    # Remove hooks
    for h in handles:
        h.remove()

    # --- Compute per-head frequency statistics ---
    print("Computing frequency statistics...", file=sys.stderr)
    sampled_heads: List[Tuple[int, int]] = []
    stats_dict: Dict[str, Dict[str, torch.Tensor]] = {}

    for layer_idx in range(num_layers):
        q_base = captured_q.get(layer_idx)
        if q_base is None:
            print(f"  [warn] No Q captured for layer {layer_idx}, skipping.", file=sys.stderr)
            continue

        # q_base: [1, num_heads, seq_len, head_dim] (pre-RoPE)
        for head_idx in range(num_heads):
            q_head = q_base[0, head_idx]  # [seq_len, head_dim]
            q_complex = _to_complex_pairs(q_head, style=rope_style)  # [seq_len, freq_count]

            q_mean_complex = q_complex.mean(dim=0)  # [freq_count]
            q_abs_mean = q_complex.abs().mean(dim=0)  # [freq_count]

            key = f"layer{layer_idx:02d}_head{head_idx:02d}"
            stats_dict[key] = {
                "q_mean_real": q_mean_complex.real.cpu(),
                "q_mean_imag": q_mean_complex.imag.cpu(),
                "q_abs_mean": q_abs_mean.cpu(),
            }
            sampled_heads.append((layer_idx, head_idx))

        # Free memory
        del captured_q[layer_idx]

    # --- Determine rope_type ---
    rope_scaling = getattr(config, "rope_scaling", {}) or {}
    rope_type = (
        rope_scaling.get("rope_type")
        or rope_scaling.get("type")
        or getattr(config, "rope_type", "default")
        or "default"
    )

    # --- Build metadata ---
    metadata = {
        "num_traces": 1,
        "head_dim": head_dim,
        "dtype": str(dtype).replace("torch.", ""),
        "use_chat_template": False,
        "system_prompt": "",
        "attn_implementation": attn_implementation,
        "rope_style": rope_style,
        "rope_type": rope_type,
        "sampled_heads": [[int(l), int(h)] for l, h in sampled_heads],
    }

    payload = {
        "metadata": metadata,
        "stats": stats_dict,
    }

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out)
    print(f"Saved stats to {out} ({len(sampled_heads)} heads)", file=sys.stderr)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate TriAttention frequency statistics from plain text."
    )
    parser.add_argument(
        "--model", required=True,
        help="HuggingFace model name or local path.",
    )
    parser.add_argument(
        "--input", required=True,
        help="Plain text file for calibration.",
    )
    parser.add_argument(
        "--output", required=True,
        help="Output .pt file path for stats.",
    )
    parser.add_argument(
        "--max-length", type=int, default=32768,
        help="Maximum token length (default: 32768).",
    )
    parser.add_argument(
        "--device", default="cuda",
        help="Device to run on (default: cuda).",
    )
    parser.add_argument(
        "--attn-implementation", default="flash_attention_2",
        help="Attention implementation (default: flash_attention_2).",
    )
    args = parser.parse_args()
    calibrate(
        model_name_or_path=args.model,
        input_path=args.input,
        output_path=args.output,
        max_length=args.max_length,
        device=args.device,
        attn_implementation=args.attn_implementation,
    )


if __name__ == "__main__":
    main()
