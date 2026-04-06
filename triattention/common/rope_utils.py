"""Rotary position embedding utilities for TriAttention."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from transformers import AutoConfig

try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
except ImportError:
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as Qwen3RotaryEmbedding
    except ImportError:
        Qwen3RotaryEmbedding = None  # type: ignore[assignment]
try:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
except ImportError:
    LlamaRotaryEmbedding = None  # type: ignore[assignment]


def determine_rope_style(config: AutoConfig) -> str:
    model_type = getattr(config, "model_type", "")
    if "llama" in model_type:
        return "half"
    return "half"  # front/back pairing (Qwen)


def build_rotary(
    cache_device: torch.device,
    model_path: Path,
    dtype: torch.dtype,
    config: Optional[AutoConfig] = None,
) -> object:
    if config is None:
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)

    rope_style = determine_rope_style(config)
    model_type = getattr(config, "model_type", "")
    if "llama" in model_type:
        if LlamaRotaryEmbedding is None:
            raise ImportError("Llama rotary embedding is unavailable in the current transformers build.")

        rope_scaling = dict(config.rope_scaling or {})
        if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
            rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
        rope_scaling.pop("attn_factor", None)
        if "rope_type" not in rope_scaling:
            rope_scaling["rope_type"] = rope_scaling.get("type", "default")
        rope_scaling.pop("type", None)
        config.rope_scaling = rope_scaling

        rotary = LlamaRotaryEmbedding(config=config, device=cache_device)
        rotary.to(dtype=dtype, device=cache_device)
        rotary._rope_style = rope_style  # type: ignore[attr-defined]
        return rotary

    if Qwen3RotaryEmbedding is None:
        raise ImportError(
            "Neither Qwen3 nor Qwen2 rotary embeddings are available in the installed transformers package."
        )

    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    if "rope_type" not in rope_scaling:
        rope_scaling["rope_type"] = rope_scaling.get("type", "default")
    rope_scaling.pop("type", None)
    config.rope_scaling = rope_scaling
    rotary = Qwen3RotaryEmbedding(config=config, device=cache_device)
    rotary.to(dtype=dtype)
    rotary._rope_style = rope_style  # type: ignore[attr-defined]
    return rotary


def compute_frequency_scaling(
    rotary: Qwen3RotaryEmbedding,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    position_ids = torch.zeros(1, 1, device=device, dtype=torch.long)
    probe = torch.zeros(1, 1, head_dim, device=device, dtype=dtype)
    cos, sin = rotary(probe, position_ids)
    cos0 = cos[0, 0]
    sin0 = sin[0, 0]
    scale = torch.sqrt(cos0[0::2].pow(2) + sin0[0::2].pow(2))
    return scale.to(device=device, dtype=torch.float32)
