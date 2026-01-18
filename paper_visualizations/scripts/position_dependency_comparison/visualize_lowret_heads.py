"""Visualize low-ret heads from hybrid_sample_heads_lowret_top10.json."""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 14
color_scatter = (85 / 250, 104 / 250, 154 / 250)
face_color = (231 / 250, 231 / 250, 240 / 250)


def style_ax(ax):
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(rotated, cos, sin, scale):
    z = rotated / scale
    return z * cos - rotate_half(z) * sin


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    num_freq = tensor.shape[-1] // 2
    return torch.complex(tensor[..., :num_freq].float(), tensor[..., num_freq:].float())


def compute_attention_heatmap(q_block, k_block, seq_len, patch_size, q_tile, device):
    seq_q, head_dim = q_block.shape
    scale = head_dim ** -0.5

    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    q_pad = num_q_groups * patch_size - seq_len
    k_pad = num_k_groups * patch_size - seq_len
    if q_pad > 0:
        q_block = torch.cat([q_block, torch.zeros(q_pad, head_dim, device=device, dtype=q_block.dtype)], dim=0)
    if k_pad > 0:
        k_block = torch.cat([k_block, torch.zeros(k_pad, head_dim, device=device, dtype=k_block.dtype)], dim=0)

    seq_k_padded = k_block.shape[0]
    key_positions = torch.arange(seq_k_padded, device=device)
    key_valid = key_positions < seq_len

    pooled_groups = torch.zeros((num_q_groups, num_k_groups), device=device, dtype=torch.float32)
    k_t = k_block.t().contiguous()

    for q_start in range(0, seq_len, q_tile):
        q_end = min(q_start + q_tile, seq_len)
        indices = torch.arange(q_start, q_end, device=device)
        q_slice = q_block[q_start:q_end]

        scores = torch.matmul(q_slice, k_t) * scale
        scores = scores.to(torch.float32)

        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        scores = scores.masked_fill(future_mask, float("-inf"))
        scores = scores.masked_fill(~key_valid.view(1, -1), float("-inf"))

        scores_flat = scores.view(-1, num_k_groups * patch_size)
        row_max = scores_flat.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        stable = torch.exp(scores_flat - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (stable / row_sum).view(-1, num_k_groups, patch_size)

        key_mask = key_valid.view(1, num_k_groups, patch_size)
        weights = weights * key_mask
        pooled_k = weights.max(dim=-1).values

        query_groups = indices // patch_size
        for local_q, global_q_group in enumerate(query_groups.tolist()):
            pooled_groups[global_q_group] = torch.maximum(pooled_groups[global_q_group], pooled_k[local_q])

    valid_q_groups = math.ceil(seq_len / patch_size)
    valid_k_groups = math.ceil(seq_len / patch_size)
    pooled_groups = pooled_groups[:valid_q_groups, :valid_k_groups]

    row_min = pooled_groups.amin(dim=1, keepdim=True)
    row_max = pooled_groups.amax(dim=1, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_groups - row_min) / denom
    return norm.detach().cpu()


def compute_dominant_frequency(q_block, k_block, cos_table, sin_table, attention_scale):
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)
    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)
    amp_product = torch.abs(q_mean) * torch.abs(k_mean)
    dominant_freq = amp_product.argmax().item()
    return amp_product, dominant_freq


def compute_mean_resultant_length(complex_vals: torch.Tensor) -> float:
    mean_vec = complex_vals.mean()
    mean_norm = torch.abs(complex_vals).mean()
    if mean_norm < 1e-8:
        return 0.0
    return (torch.abs(mean_vec) / mean_norm).item()


def get_scatter_data(q_block, k_block, cos_table, sin_table, attention_scale, freq_idx):
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    q_vals = q_complex[:, freq_idx]
    k_vals = k_complex[:, freq_idx]

    R_q = compute_mean_resultant_length(q_vals)
    R_k = compute_mean_resultant_length(k_vals)

    max_points = 5000
    if len(q_vals) > max_points:
        indices = torch.randperm(len(q_vals))[:max_points]
        q_vals = q_vals[indices]
        k_vals = k_vals[indices]

    q_scale = torch.abs(q_vals).max().item()
    k_scale = torch.abs(k_vals).max().item()
    if q_scale > 1e-8:
        q_vals = q_vals / q_scale
    if k_scale > 1e-8:
        k_vals = k_vals / k_scale

    return (
        q_vals.real.cpu().numpy(),
        q_vals.imag.cpu().numpy(),
        k_vals.real.cpu().numpy(),
        k_vals.imag.cpu().numpy(),
        R_q,
        R_k,
    )


def main():
    mask_process_command("PD-L1_binder_lowret")

    # Load heads from json
    json_path = Path("weian_development/spec_sparse_simulator/hybrid_sample_heads_lowret_top10.json")
    heads = json.loads(json_path.read_text())
    num_heads = len(heads)

    trace_dir = Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34")
    model_path = Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B")
    device = torch.device("cuda:0")
    patch_size = 80
    q_tile = 512
    dpi = 200

    # Load data
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling

    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    inv_freq = rotary.inv_freq.to(torch.float64)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    head_dim = 128
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=torch.float32)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=torch.float32)
    sin_table = sin_table[0].to(dtype=torch.float32)

    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    # Create figure: 2 rows x num_heads cols
    fig = plt.figure(figsize=(3.5 * num_heads, 7), dpi=dpi)
    gs = GridSpec(2, num_heads, figure=fig, wspace=0.02, hspace=0.12,
                  left=0.04, right=0.99, top=0.94, bottom=0.03)

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
    })

    heatmaps = []
    scatter_data = []

    for col_idx, (layer, head) in enumerate(heads):
        print(f"Processing L{layer}H{head}...")

        q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
        k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            heatmap = compute_attention_heatmap(q_block, k_block, token_count, patch_size, q_tile, device)
        heatmaps.append(heatmap)

        amp_product, dom_freq = compute_dominant_frequency(q_block, k_block, cos_table, sin_table, attention_scale)
        print(f"  Dominant frequency: {dom_freq}")

        q_real, q_imag, k_real, k_imag, R_q, R_k = get_scatter_data(
            q_block, k_block, cos_table, sin_table, attention_scale, dom_freq
        )
        print(f"  R_Q={R_q:.4f}, R_K={R_k:.4f}")
        scatter_data.append({
            'q_real': q_real, 'q_imag': q_imag,
            'k_real': k_real, 'k_imag': k_imag,
            'dom_freq': dom_freq,
            'R_q': R_q, 'R_k': R_k,
            'layer': layer, 'head': head,
        })

    # Row 0: Attention Maps
    for col_idx, (layer, head) in enumerate(heads):
        ax_attn = fig.add_subplot(gs[0, col_idx])
        ax_attn.imshow(heatmaps[col_idx].numpy(), cmap="inferno", aspect="equal", origin="upper")
        ax_attn.set_title(f"L{layer}H{head}", fontsize=FONT_SIZE)
        ax_attn.set_xticks([])
        ax_attn.set_yticks([])
        for spine in ax_attn.spines.values():
            spine.set_visible(False)

    # Row 1: Scatter Plots
    for col_idx, data in enumerate(scatter_data):
        ax_scatter = fig.add_subplot(gs[1, col_idx])
        style_ax(ax_scatter)

        ax_scatter.scatter(data['q_real'], data['q_imag'], s=6, alpha=0.25,
                          color=color_scatter, label="Q", edgecolors="none")
        ax_scatter.scatter(data['k_real'], data['k_imag'], s=6, alpha=0.25,
                          color="gray", label="K", edgecolors="none")

        ax_scatter.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        ax_scatter.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)

        ax_scatter.set_xlim(-1.1, 1.1)
        ax_scatter.set_ylim(-1.1, 1.1)
        ax_scatter.set_aspect("equal", adjustable="box")
        ax_scatter.set_xticklabels([])
        ax_scatter.set_yticklabels([])
        ax_scatter.set_title(f"f={data['dom_freq']}", fontsize=FONT_SIZE)

        ax_scatter.text(0.95, 0.95, f"$R_Q$={data['R_q']:.3f}\n$R_K$={data['R_k']:.3f}",
                       transform=ax_scatter.transAxes, fontsize=FONT_SIZE - 2,
                       ha='right', va='top')

        if col_idx == 0:
            leg = ax_scatter.legend(loc="upper left", fontsize=FONT_SIZE - 2, frameon=False,
                                    markerscale=3, handletextpad=0.3)
            for lh in leg.legend_handles:
                lh.set_alpha(1.0)

    # Add row labels
    fig.text(0.02, 0.72, "Attention\nMap", ha='center', va='center',
             fontsize=FONT_SIZE, fontweight='bold', rotation=90)
    fig.text(0.02, 0.28, "Q/K @\nDom. Freq", ha='center', va='center',
             fontsize=FONT_SIZE, fontweight='bold', rotation=90)

    output_path = Path("paper_visualizations/outputs/position_dependency_comparison/lowret_heads_top10.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
