"""Shared helpers for round-based sparse KV pruning."""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

from weian_development.attention_qk_analysis.freq_magnitude_plots import invert_rope
from weian_development.attention_qk_analysis.freq_magnitude_single_plot_meanvec_randomk import (
    to_complex_pairs,
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


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
) -> Qwen3RotaryEmbedding:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling
    rotary = Qwen3RotaryEmbedding(config=config, device=cache_device)
    rotary.to(dtype=dtype)
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


def compute_rotary_tables(
    rotary: Qwen3RotaryEmbedding,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0]
    sin_table = sin_table[0]
    inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float64)
    freq_scale = compute_frequency_scaling(rotary, head_dim, dtype, device)
    return cos_table, sin_table, inv_freq, freq_scale


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
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_complex = to_complex_pairs(k_unrot)
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
) -> torch.Tensor:
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)

    freq_scale_sq = freq_scale_sq.to(device=amp.device, dtype=torch.float32)
    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi.unsqueeze(1)
    cos_phase = torch.cos(phase)
    scale = freq_scale_sq.view(1, 1, -1)
    base_scores = (amp.unsqueeze(1) * scale * cos_phase).sum(dim=2)
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
