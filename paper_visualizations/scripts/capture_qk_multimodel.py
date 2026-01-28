"""Capture Q/K tensors from multiple model architectures (Qwen2, Qwen3, Llama).

Adapted from weian_development/attention_qk_analysis/capture_qk_distributed.py
for paper visualization purposes. Supports single-trace capture with direct
prompt+output input.

Usage:
    conda activate rkv
    python paper_visualizations/scripts/capture_qk_multimodel.py \
        --model-path /path/to/model \
        --trace-jsonl /path/to/trace.jsonl \
        --trace-index 0 \
        --output-dir paper_visualizations/outputs/qk_traces/model_name \
        --gpu 1 \
        --precision bfloat16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def get_model_family(config) -> str:
    """Detect model family from config."""
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])

    if "qwen3" in model_type or any("Qwen3" in a for a in architectures):
        return "qwen3"
    elif "qwen2" in model_type or any("Qwen2" in a for a in architectures):
        return "qwen2"
    elif "llama" in model_type or any("Llama" in a for a in architectures):
        return "llama"
    else:
        raise ValueError(f"Unsupported model type: {model_type}, architectures: {architectures}")


def import_model_utils(family: str):
    """Import apply_rotary_pos_emb and repeat_kv for the model family."""
    if family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import apply_rotary_pos_emb, repeat_kv
    elif family == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb, repeat_kv
    elif family == "llama":
        from transformers.models.llama.modeling_llama import apply_rotary_pos_emb, repeat_kv
    else:
        raise ValueError(f"Unknown model family: {family}")
    return apply_rotary_pos_emb, repeat_kv


class LayerCaptureBuffer:
    """Pre-allocated CPU tensors for layer-wise Q/K storage."""

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        seq_len: int,
        head_dim: int,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.seq_len = seq_len
        self.head_dim = head_dim
        self.dtype = dtype
        self.q = torch.empty((num_layers, num_heads, seq_len, head_dim), dtype=dtype)
        self.k = torch.empty((num_layers, num_heads, seq_len, head_dim), dtype=dtype)

    def store(self, layer_idx: int, q_tensor: torch.Tensor, k_tensor: torch.Tensor) -> None:
        q_cpu = q_tensor.squeeze(0).to(device="cpu", dtype=self.dtype)
        k_cpu = k_tensor.squeeze(0).to(device="cpu", dtype=self.dtype)
        self.q[layer_idx].copy_(q_cpu)
        self.k[layer_idx].copy_(k_cpu)
        del q_cpu, k_cpu


class QKCollector:
    """Forward pre-hook to capture post-RoPE Q/K tensors."""

    def __init__(self, model, buffer: LayerCaptureBuffer, model_family: str) -> None:
        self.model = model
        self.buffer = buffer
        self.model_family = model_family
        self.handles: List[torch.utils.hooks.RemovableHandle] = []

        self.apply_rotary_pos_emb, self.repeat_kv = import_model_utils(model_family)

        text_config = model.config.get_text_config()
        self.num_heads = text_config.num_attention_heads
        self.kv_heads = text_config.num_key_value_heads
        self.head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)
        self.num_key_value_groups = self.num_heads // self.kv_heads

        for layer_idx, layer in enumerate(model.model.layers):
            handle = layer.self_attn.register_forward_pre_hook(
                self._make_hook(layer_idx),
                with_kwargs=True,
            )
            self.handles.append(handle)

    def _make_hook(self, layer_idx: int):
        def hook(module, args, kwargs):
            hidden_states = kwargs.get("hidden_states")
            if hidden_states is None and args:
                hidden_states = args[0]
            position_embeddings = kwargs.get("position_embeddings")
            if position_embeddings is None and len(args) > 1:
                position_embeddings = args[1]
            if hidden_states is None or position_embeddings is None:
                raise RuntimeError("Cannot get hidden_states or position_embeddings in pre-hook")

            cos, sin = position_embeddings
            batch, seq_len, _ = hidden_states.shape
            if batch != 1:
                raise RuntimeError(f"Expected batch=1, got {batch}")

            # Project Q/K
            q_proj = module.q_proj(hidden_states)
            q_proj = q_proj.view(batch, seq_len, self.num_heads, self.head_dim)
            # Qwen3 has q_norm, Qwen2/Llama do not
            if hasattr(module, "q_norm") and module.q_norm is not None:
                q_proj = module.q_norm(q_proj)
            q_states = q_proj.permute(0, 2, 1, 3).contiguous()

            k_proj = module.k_proj(hidden_states)
            k_proj = k_proj.view(batch, seq_len, self.kv_heads, self.head_dim)
            if hasattr(module, "k_norm") and module.k_norm is not None:
                k_proj = module.k_norm(k_proj)
            k_states = k_proj.permute(0, 2, 1, 3).contiguous()

            # Apply RoPE
            q_rot, k_rot = self.apply_rotary_pos_emb(q_states, k_states, cos, sin)
            # Expand KV heads to match Q heads (GQA)
            k_rot = self.repeat_kv(k_rot, self.num_key_value_groups)

            self.buffer.store(layer_idx, q_rot.detach(), k_rot.detach())

            del q_proj, q_states, k_proj, k_states, q_rot, k_rot

        return hook

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


def load_trace_from_jsonl(jsonl_path: Path, trace_index: int) -> dict:
    """Load a specific trace from JSONL file by line index."""
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i == trace_index:
                return json.loads(line)
    raise ValueError(f"Trace index {trace_index} not found in {jsonl_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture Q/K tensors from multiple model architectures")
    parser.add_argument("--model-path", type=Path, required=True, help="Model directory path")
    parser.add_argument("--trace-jsonl", type=Path, required=True, help="JSONL file containing traces")
    parser.add_argument("--trace-index", type=int, default=0, help="Line index in JSONL file")
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory for qk.pt")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--precision", choices=["float16", "bfloat16", "float32"], default="bfloat16")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_process_command(f"PD-L1_binder_qk{args.gpu}")

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)

    dtype_map = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    load_dtype = dtype_map[args.precision]

    # Load config and detect model family
    print(f"Loading config from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_family = get_model_family(config)
    print(f"Detected model family: {model_family}")

    # Load tokenizer and model
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)

    print(f"Loading model to GPU {args.gpu}...")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=load_dtype,
        trust_remote_code=True,
        device_map=None,
    )
    model.to(device)
    model.eval()

    # Load trace
    print(f"Loading trace {args.trace_index} from {args.trace_jsonl}...")
    trace_data = load_trace_from_jsonl(args.trace_jsonl, args.trace_index)

    # Build full text: prompt + output
    prompt = trace_data.get("prompt", "")
    output = trace_data.get("output", "")
    full_text = prompt + output

    if args.verbose:
        print(f"  Question ID: {trace_data.get('id', 'N/A')}")
        print(f"  Prompt tokens: {trace_data.get('prefill_tokens', 'N/A')}")
        print(f"  Total tokens: {trace_data.get('total_tokens', 'N/A')}")

    # Tokenize
    tokenized = tokenizer(full_text, return_tensors="pt", add_special_tokens=False)
    input_ids = tokenized["input_ids"].to(device)
    attention_mask = tokenized["attention_mask"].to(device)
    seq_len = input_ids.shape[1]

    print(f"Sequence length: {seq_len} tokens")

    # Get model dimensions
    text_config = config.get_text_config()
    num_layers = text_config.num_hidden_layers
    num_heads = text_config.num_attention_heads
    head_dim = getattr(text_config, "head_dim", text_config.hidden_size // text_config.num_attention_heads)

    print(f"Model: {num_layers} layers, {num_heads} heads, head_dim={head_dim}")

    # Create buffer and collector
    buffer = LayerCaptureBuffer(num_layers, num_heads, seq_len, head_dim, dtype=load_dtype)
    collector = QKCollector(model, buffer, model_family)

    # Forward pass
    print("Running forward pass to capture Q/K...")
    try:
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
    finally:
        collector.remove()

    # Save outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    tensor_path = args.output_dir / "qk.pt"
    meta_path = args.output_dir / "metadata.json"

    print(f"Saving Q/K tensors to {tensor_path}...")
    torch.save({"q": buffer.q, "k": buffer.k}, tensor_path, _use_new_zipfile_serialization=False)

    meta_payload = {
        "model_path": str(args.model_path),
        "model_family": model_family,
        "trace_jsonl": str(args.trace_jsonl),
        "trace_index": args.trace_index,
        "question_id": trace_data.get("id"),
        "sample_idx": trace_data.get("sample_idx"),
        "question": trace_data.get("question", "")[:500],  # Truncate for readability
        "token_count": seq_len,
        "prefill_tokens": trace_data.get("prefill_tokens"),
        "sequence_length": seq_len,
        "num_layers": num_layers,
        "num_heads": num_heads,
        "num_kv_heads": text_config.num_key_value_heads,
        "head_dim": head_dim,
        "precision": str(buffer.q.dtype),
    }
    meta_path.write_text(json.dumps(meta_payload, ensure_ascii=False, indent=2), encoding="utf-8")

    # Cleanup
    del buffer.q, buffer.k, buffer
    torch.cuda.empty_cache()

    print(f"Done! Output saved to {args.output_dir}")
    print(f"  qk.pt: {tensor_path.stat().st_size / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
