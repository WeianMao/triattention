"""Generate position dependency comparison figure.

Compares Relative-Position-Dependent vs Independent attention heads
across two dimensions: attention maps and dominant frequency Q/K scatter.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Tuple

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
FONT_SIZE = 16
color_dependent = (85 / 250, 104 / 250, 154 / 250)    # blue for dependent
color_independent = (187 / 250, 130 / 250, 90 / 250)  # warm brown for independent
face_color = (231 / 250, 231 / 250, 240 / 250)        # light gray-purple background


def style_ax(ax, grid_axis="both"):
    """Apply consistent styling to axes."""
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, alpha=0.7, color="white", linewidth=1.5)


# ============ Head Configurations ============
# Relative-Position-Dependent Heads
DEPENDENT_HEADS = [
    {"layer": 6, "head": 9, "label": "Local Attention"},
    {"layer": 9, "head": 20, "label": "Attention Sink"},
]

# Relative-Position-Independent Heads
INDEPENDENT_HEADS = [
    {"layer": 3, "head": 11, "label": "Vertical Stripes"},
    {"layer": 17, "head": 25, "label": "Retrieval Head"},
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate position dependency comparison figure")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Pooling window size for attention maps (larger = coarser)",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=512,
        help="Query tile size for memory-efficient attention computation",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for figure (default: outputs/position_dependency_comparison/)",
    )
    return parser.parse_args()


# ============ RoPE Utilities ============
def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(rotated, cos, sin, scale):
    z = rotated / scale
    return z * cos - rotate_half(z) * sin


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    num_freq = tensor.shape[-1] // 2
    return torch.complex(tensor[..., :num_freq].float(), tensor[..., num_freq:].float())


# ============ Attention Map Computation ============
def compute_attention_heatmap(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute pooled attention heatmap for a single head."""
    seq_q, head_dim = q_block.shape
    scale = head_dim ** -0.5

    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    # Padding
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

        # Causal mask
        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        scores = scores.masked_fill(future_mask, float("-inf"))
        scores = scores.masked_fill(~key_valid.view(1, -1), float("-inf"))

        # Softmax
        scores_flat = scores.view(-1, num_k_groups * patch_size)
        row_max = scores_flat.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        stable = torch.exp(scores_flat - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (stable / row_sum).view(-1, num_k_groups, patch_size)

        key_mask = key_valid.view(1, num_k_groups, patch_size)
        weights = weights * key_mask

        # Max pool over key dimension
        pooled_k = weights.max(dim=-1).values

        # Aggregate into query groups
        query_groups = indices // patch_size
        for local_q, global_q_group in enumerate(query_groups.tolist()):
            pooled_groups[global_q_group] = torch.maximum(
                pooled_groups[global_q_group], pooled_k[local_q]
            )

    # Normalize per row
    valid_q_groups = math.ceil(seq_len / patch_size)
    valid_k_groups = math.ceil(seq_len / patch_size)
    pooled_groups = pooled_groups[:valid_q_groups, :valid_k_groups]

    row_min = pooled_groups.amin(dim=1, keepdim=True)
    row_max = pooled_groups.amax(dim=1, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_groups - row_min) / denom
    return norm.detach().cpu()


# ============ Amplitude-based Dominant Frequency Selection ============
def compute_dominant_frequency(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
) -> Tuple[torch.Tensor, int]:
    """
    Compute dominant frequency using amplitude product |E[q]| * |E[k]|.
    Returns amplitude product per frequency and the dominant frequency index.
    """
    # Invert RoPE
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    # Convert to complex
    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    # Compute mean vectors
    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    # Amplitude product
    amp_product = torch.abs(q_mean) * torch.abs(k_mean)

    dominant_freq = amp_product.argmax().item()
    return amp_product, dominant_freq


# ============ Scatter Plot Data ============
def compute_mean_resultant_length(complex_vals: torch.Tensor) -> float:
    """Compute Mean Resultant Length: R = ||E[z]|| / E[||z||]"""
    mean_vec = complex_vals.mean()
    mean_norm = torch.abs(complex_vals).mean()
    if mean_norm < 1e-8:
        return 0.0
    return (torch.abs(mean_vec) / mean_norm).item()


def get_scatter_data(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    freq_idx: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Get raw Q/K scatter data for a specific frequency (no centering).

    Q and K are independently scaled to have similar range for visualization,
    but without shifting the mean (no centering).

    Returns: (q_real, q_imag, k_real, k_imag, R_q, R_k)
    """
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    q_vals = q_complex[:, freq_idx]
    k_vals = k_complex[:, freq_idx]

    # Compute Mean Resultant Length before subsampling
    R_q = compute_mean_resultant_length(q_vals)
    R_k = compute_mean_resultant_length(k_vals)

    # Subsample for visualization
    max_points = 5000
    if len(q_vals) > max_points:
        indices = torch.randperm(len(q_vals))[:max_points]
        q_vals = q_vals[indices]
        k_vals = k_vals[indices]

    # Independent scaling for Q and K (scale by max absolute value, no centering)
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


# ============ Main Figure Generation ============
def generate_figure(
    trace_dir: Path,
    model_path: Path,
    device: torch.device,
    patch_size: int,
    q_tile: int,
    dpi: int,
    output_path: Path,
    attn_cmap: str = "inferno",
) -> None:
    """Generate the comparison figure."""
    mask_process_command("PD-L1_binder_pos_dep")

    # Load data
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")

    # Load model config for RoPE
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling

    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    inv_freq = rotary.inv_freq.to(torch.float64)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # Build RoPE tables
    head_dim = 128
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=torch.float32)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=torch.float32)
    sin_table = sin_table[0].to(dtype=torch.float32)

    # Load Q/K tensors
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    all_heads = DEPENDENT_HEADS + INDEPENDENT_HEADS
    num_heads = len(all_heads)

    # Create figure: 2 rows x 4 cols (horizontal layout)
    # Row 0: attention maps, Row 1: scatter plots
    fig = plt.figure(figsize=(13, 6.5), dpi=dpi)
    gs = GridSpec(2, num_heads, figure=fig, wspace=0.01, hspace=0.05,
                  left=0.06, right=0.99, top=0.94, bottom=0.03)

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
        'xtick.labelsize': FONT_SIZE - 2,
        'ytick.labelsize': FONT_SIZE - 2,
    })

    # Store data for both rows
    heatmaps = []
    scatter_data = []

    for col_idx, head_info in enumerate(all_heads):
        layer = head_info["layer"]
        head = head_info["head"]
        label = head_info["label"]

        # Use unified color for all Q points (blue-purple)
        scatter_color = color_dependent

        print(f"Processing Layer {layer}, Head {head} ({label})...")

        q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
        k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

        # Compute attention map
        with torch.no_grad():
            heatmap = compute_attention_heatmap(q_block, k_block, token_count, patch_size, q_tile, device)
        heatmaps.append(heatmap)

        # Compute dominant frequency and scatter data
        amp_product, dom_freq = compute_dominant_frequency(
            q_block, k_block, cos_table, sin_table, attention_scale
        )
        print(f"  Dominant frequency (amplitude-based): {dom_freq}")

        q_real, q_imag, k_real, k_imag, R_q, R_k = get_scatter_data(
            q_block, k_block, cos_table, sin_table, attention_scale, dom_freq
        )
        print(f"  R_Q={R_q:.4f}, R_K={R_k:.4f}")
        scatter_data.append({
            'q_real': q_real, 'q_imag': q_imag,
            'k_real': k_real, 'k_imag': k_imag,
            'dom_freq': dom_freq, 'amp': amp_product[dom_freq].item(),
            'color': scatter_color,
            'R_q': R_q, 'R_k': R_k,
        })

    # Row 0: Attention Maps
    for col_idx, head_info in enumerate(all_heads):
        layer = head_info["layer"]
        head = head_info["head"]
        label = head_info["label"]

        ax_attn = fig.add_subplot(gs[0, col_idx])
        ax_attn.imshow(heatmaps[col_idx].numpy(), cmap=attn_cmap, aspect="equal", origin="upper")
        ax_attn.set_title(f"{label}", fontsize=FONT_SIZE)
        ax_attn.set_xticks([])
        ax_attn.set_yticks([])
        # Remove black border
        for spine in ax_attn.spines.values():
            spine.set_visible(False)


    # Row 1: Scatter Plots
    for col_idx, data in enumerate(scatter_data):
        ax_scatter = fig.add_subplot(gs[1, col_idx])
        style_ax(ax_scatter)

        ax_scatter.scatter(data['q_real'], data['q_imag'], s=6, alpha=0.25,
                          color=data['color'], label="Q", edgecolors="none")
        ax_scatter.scatter(data['k_real'], data['k_imag'], s=6, alpha=0.25,
                          color="gray", label="K", edgecolors="none")

        ax_scatter.axhline(0.0, color="gray", linewidth=0.8, alpha=0.5)
        ax_scatter.axvline(0.0, color="gray", linewidth=0.8, alpha=0.5)

        ax_scatter.set_xlim(-1.1, 1.1)
        ax_scatter.set_ylim(-1.1, 1.1)
        ax_scatter.set_aspect("equal", adjustable="box")
        ax_scatter.set_xticklabels([])
        ax_scatter.set_yticklabels([])
        ax_scatter.set_title("", fontsize=FONT_SIZE)

        # Add R values in top right corner
        ax_scatter.text(0.95, 0.95, f"$R_Q$={data['R_q']:.3f}\n$R_K$={data['R_k']:.3f}",
                       transform=ax_scatter.transAxes, fontsize=FONT_SIZE - 2,
                       ha='right', va='top')

        if col_idx == 0:
            leg = ax_scatter.legend(loc="upper left", fontsize=FONT_SIZE - 2, frameon=False,
                                    markerscale=3, handletextpad=0.3)
            # Make legend markers opaque
            for lh in leg.legend_handles:
                lh.set_alpha(1.0)

    # Add row labels on the left
    fig.text(0.04, 0.72, "Attention\nMap", ha='center', va='center',
             fontsize=FONT_SIZE, fontweight='bold', rotation=90)
    fig.text(0.04, 0.28, "Q/K @\nDom. Freq", ha='center', va='center',
             fontsize=FONT_SIZE, fontweight='bold', rotation=90)

    # Add top labels: Relative-Position-Dependent (left) and Independent (right) with arrow
    from matplotlib.patches import FancyArrowPatch

    # Place text at left and right edges
    txt_left = fig.text(0.06, 0.98, "Relative-Position-Dependent", ha='left', va='bottom',
                        fontsize=FONT_SIZE, fontweight='bold')
    txt_right = fig.text(0.99, 0.98, "Relative-Position-Independent", ha='right', va='bottom',
                         fontsize=FONT_SIZE, fontweight='bold')

    # Draw once to get text bounding boxes
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()

    # Get text bounds in figure coordinates
    bbox_left = txt_left.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    bbox_right = txt_right.get_window_extent(renderer).transformed(fig.transFigure.inverted())

    # Arrow starts after left text, ends before right text
    arrow_y = (bbox_left.y0 + bbox_left.y1) / 2
    arrow_start = bbox_left.x1 + 0.01
    arrow_end = bbox_right.x0 - 0.01

    arrow = FancyArrowPatch((arrow_start, arrow_y), (arrow_end, arrow_y),
                            transform=fig.transFigure,
                            arrowstyle='->', mutation_scale=15,
                            color='black', lw=1.5)
    fig.patches.append(arrow)

    # Add bottom labels: Concentrated (left) and Dispersed (right) with arrow
    txt_bottom_left = fig.text(0.06, -0.01, "Concentrated", ha='left', va='bottom',
                               fontsize=FONT_SIZE, fontweight='bold')
    txt_bottom_right = fig.text(0.99, -0.01, "Dispersed", ha='right', va='bottom',
                                fontsize=FONT_SIZE, fontweight='bold')

    # Get text bounds for bottom arrow
    bbox_bottom_left = txt_bottom_left.get_window_extent(renderer).transformed(fig.transFigure.inverted())
    bbox_bottom_right = txt_bottom_right.get_window_extent(renderer).transformed(fig.transFigure.inverted())

    arrow_bottom_y = (bbox_bottom_left.y0 + bbox_bottom_left.y1) / 2
    arrow_bottom_start = bbox_bottom_left.x1 + 0.01
    arrow_bottom_end = bbox_bottom_right.x0 - 0.01

    arrow_bottom = FancyArrowPatch((arrow_bottom_start, arrow_bottom_y), (arrow_bottom_end, arrow_bottom_y),
                                   transform=fig.transFigure,
                                   arrowstyle='->', mutation_scale=15,
                                   color='black', lw=1.5)
    fig.patches.append(arrow_bottom)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


def main() -> None:
    args = parse_args()

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/position_dependency_comparison")
        output_path = output_dir / "fig_position_dependency_comparison.png"
    else:
        output_path = args.output_path

    device = torch.device(args.device)

    generate_figure(
        trace_dir=args.trace_dir,
        model_path=args.model_path,
        device=device,
        patch_size=args.patch_size,
        q_tile=args.q_tile,
        dpi=args.dpi,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
