"""Shared helpers for round-based sparse KV pruning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from transformers import AutoConfig
try:
    from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
except ImportError:  # Transformers build without Qwen3 modules
    try:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as Qwen3RotaryEmbedding
    except ImportError:
        Qwen3RotaryEmbedding = None  # type: ignore[assignment]
try:
    from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
except ImportError:
    LlamaRotaryEmbedding = None  # type: ignore[assignment]

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def determine_rope_style(config: AutoConfig) -> str:
    model_type = getattr(config, "model_type", "")
    # HF Llama (incl. llama3 / YaRN scalings) applies RoPE by pairing the front/back
    # halves of the head dimension, not even/odd interleaving.
    if "llama" in model_type:
        return "half"
    return "half"  # front/back pairing (Qwen)


def rotate_half(x: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    if style == "interleaved":
        # Llama-style even/odd pairing
        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        return torch.stack((-x_odd, x_even), dim=-1).flatten(-2)
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def invert_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
    *,
    style: str = "half",
) -> torch.Tensor:
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    if style == "interleaved":
        # Allow even/odd to carry different cos/sin (e.g., YaRN/llama3).
        even = base[..., ::2]
        odd = base[..., 1::2]
        cos_even = cos_unit[..., ::2]
        cos_odd = cos_unit[..., 1::2]
        sin_even = sin_unit[..., ::2]
        sin_odd = sin_unit[..., 1::2]
        det = cos_even * cos_odd + sin_even * sin_odd
        det = det.clamp_min(1e-12)
        # Forward: y_even = x_even * cos_even - x_odd * sin_even
        #          y_odd  = x_odd  * cos_odd  + x_even * sin_odd
        orig_even = (even * cos_odd + odd * sin_even) / det
        orig_odd = (odd * cos_even - even * sin_odd) / det
        restored = torch.empty_like(base)
        restored[..., ::2] = orig_even
        restored[..., 1::2] = orig_odd
        return restored
    return base * cos_unit - rotate_half(base, style=style) * sin_unit


def to_complex_pairs(tensor: torch.Tensor, *, style: str = "half") -> torch.Tensor:
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    if style == "interleaved":
        real = tensor_real[..., ::2].contiguous()
        imag = tensor_real[..., 1::2].contiguous()
        return torch.complex(real, imag)
    freq_count = tensor.shape[1] // 2
    real = tensor_real[:, :freq_count].contiguous()
    imag = tensor_real[:, freq_count:].contiguous()
    return torch.complex(real, imag)


@dataclass
class HeadFrequencyStats:
    q_mean_complex: torch.Tensor
    q_abs_mean: torch.Tensor


def load_or_create_sample(
    sample_file: Path,
    sample_count: int,
    seed: int,
    layer_count: int,
    head_count: int,
) -> List[Tuple[int, int]]:
    if sample_file.exists():
        with sample_file.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        return [(int(pair[0]), int(pair[1])) for pair in data]

    if sample_count > layer_count * head_count:
        raise ValueError("sample_count exceeds total available heads")

    indices = [(layer, head) for layer in range(layer_count) for head in range(head_count)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(len(indices), generator=generator)
    selected = [indices[idx] for idx in perm[:sample_count].tolist()]

    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with sample_file.open("w", encoding="utf-8") as handle:
        json.dump([[layer, head] for layer, head in selected], handle, indent=2)

    return selected


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


def verify_rotary_alignment(local_rotary: torch.nn.Module, model_rotary: torch.nn.Module) -> None:
    """
    Asserts that the locally constructed RotaryEmbedding matches the model's live RotaryEmbedding.
    Checks inv_freq and scaling factors.
    """
    # Check inv_freq
    if hasattr(local_rotary, "inv_freq") and hasattr(model_rotary, "inv_freq"):
        local_freq = local_rotary.inv_freq.to("cpu")
        model_freq = model_rotary.inv_freq.to("cpu")
        
        if local_freq.shape != model_freq.shape:
             raise ValueError(
                f"Rotary embedding shape mismatch! Local: {local_freq.shape}, Model: {model_freq.shape}"
            )

        # Allow small float error
        if not torch.allclose(local_freq, model_freq, atol=1e-5):
            diff = (local_freq - model_freq).abs().max().item()
            raise ValueError(
                f"Rotary embedding inv_freq mismatch! Max diff: {diff}. "
                "This likely means the pruner's RoPE config does not match the model's."
            )
    
    # Check scaling factor if present (e.g. Qwen)
    # Llama usually bakes scaling into inv_freq or uses explicit implementation differences.
    # Ideally inv_freq equality covers most scaling issues for Llama3.



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


def compute_rotary_tables(
    rotary: Qwen3RotaryEmbedding,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, str]:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0]
    sin_table = sin_table[0]
    inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float64)
    freq_scale = compute_frequency_scaling(rotary, head_dim, dtype, device)
    style = getattr(rotary, "_rope_style", "half")
    return cos_table, sin_table, inv_freq, freq_scale, style


def build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_unrot: torch.Tensor,
    *,
    style: str = "half",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_complex = to_complex_pairs(k_unrot, style=style)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def score_keys_for_round(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
    freq_scale_sq: torch.Tensor,
    disable_top_n_high_freq: int = 0,
    simulate_bug_phase_offset: int = 0,
) -> torch.Tensor:
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)

    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)
    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(1)

    # Simulate bug 896cbca6 phase offset: bug caused phi to be offset by -Δ×ω
    if simulate_bug_phase_offset > 0:
        phase_offset = float(simulate_bug_phase_offset) * omega.view(1, 1, -1)
        phase = phase - phase_offset

    cos_phase = torch.cos(phase)

    # High-frequency ablation: mask top-n high-frequency components in position-dependent term only
    if disable_top_n_high_freq > 0:
        position_scale = freq_scale_sq.clone()
        position_scale[:disable_top_n_high_freq] = 0  # mask highest frequencies (omega[0:n])
        scale = position_scale.view(1, 1, -1)
    else:
        scale = freq_scale_sq.view(1, 1, -1)

    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)
    # additive term uses original freq_scale_sq (not affected by high-freq masking)
    additive = (extra * freq_scale_sq.view(1, -1)).sum(dim=1, keepdim=True)
    combined = base_scores + additive

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values


def save_head_frequency_stats(
    output_path: Path,
    sampled_heads: Sequence[Tuple[int, int]],
    stats_map: Dict[Tuple[int, int], HeadFrequencyStats],
    metadata: Dict[str, torch.Tensor | int | str | float],
) -> None:
    payload: Dict[str, object] = {
        "metadata": {
            **metadata,
            "sampled_heads": [[int(layer), int(head)] for layer, head in sampled_heads],
        },
        "stats": {},
    }
    for (layer, head), head_stats in stats_map.items():
        key = f"layer{layer:02d}_head{head:02d}"
        payload["stats"][key] = {
            "q_mean_real": head_stats.q_mean_complex.real.cpu(),
            "q_mean_imag": head_stats.q_mean_complex.imag.cpu(),
            "q_abs_mean": head_stats.q_abs_mean.cpu(),
        }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_path)


def load_head_frequency_stats(
    stats_path: Path,
    device: torch.device,
) -> tuple[Dict[str, object], Dict[Tuple[int, int], HeadFrequencyStats]]:
    payload = torch.load(stats_path, map_location=device)
    metadata = payload["metadata"]
    stats_raw: Dict[str, Dict[str, torch.Tensor]] = payload["stats"]
    sampled_heads = [tuple(item) for item in metadata["sampled_heads"]]
    stats: Dict[Tuple[int, int], HeadFrequencyStats] = {}
    for layer, head in sampled_heads:
        key = f"layer{layer:02d}_head{head:02d}"
        entry = stats_raw.get(key)
        if entry is None:
            continue
        q_mean_complex = torch.complex(
            entry["q_mean_real"].to(device=device, dtype=torch.float32),
            entry["q_mean_imag"].to(device=device, dtype=torch.float32),
        )
        q_abs_mean = entry["q_abs_mean"].to(device=device, dtype=torch.float32)
        stats[(int(layer), int(head))] = HeadFrequencyStats(
            q_mean_complex=q_mean_complex,
            q_abs_mean=q_abs_mean,
        )
    return metadata, stats
