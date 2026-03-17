"""Generate combined figure: Q/K scatter for L32H27 + R distribution histogram.

Layout inspired by fig_freq_reconstruction_analysis.png
"""
from __future__ import annotations

import argparse
import json
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 16
LABEL_FONT_SIZE = 20
color_q = (85 / 250, 104 / 250, 154 / 250)    # blue for Q
color_k = (187 / 250, 130 / 250, 90 / 250)    # warm brown for K
face_color = (231 / 250, 231 / 250, 240 / 250)


def style_ax(ax, grid_axis="both"):
    """Apply consistent styling to axes."""
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
    ax.set_axisbelow(True)
    ax.grid(True, axis=grid_axis, alpha=0.5, color="white", linewidth=1.2)


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


# ============ R Computation ============
def compute_mean_resultant_length(complex_vals: torch.Tensor) -> float:
    mean_vec = complex_vals.mean()
    mean_norm = torch.abs(complex_vals).mean()
    if mean_norm < 1e-8:
        return 0.0
    return (torch.abs(mean_vec) / mean_norm).item()


def compute_dominant_frequency(q_complex: torch.Tensor, k_complex: torch.Tensor) -> int:
    q_amp = torch.abs(q_complex.mean(dim=0))
    k_amp = torch.abs(k_complex.mean(dim=0))
    amp_product = q_amp * k_amp
    return amp_product.argmax().item()


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined R distribution figure with scatter")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--dpi", type=int, default=200)
    parser.add_argument("--bins", type=int, default=40)
    parser.add_argument("--layer", type=int, default=32, help="Layer for scatter plot")
    parser.add_argument("--head", type=int, default=27, help="Head for scatter plot")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_r_scatter")

    if args.output_path is None:
        output_path = Path("paper_visualizations/outputs/r_distribution/r_distribution_with_scatter.png")
    else:
        output_path = args.output_path

    device = torch.device(args.device)

    # Load data
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")

    # Load model config for RoPE
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling

    rotary = Qwen3RotaryEmbedding(config=config, device=device)
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

    num_layers = q_tensor.shape[0]
    num_heads = q_tensor.shape[1]

    # ========== Compute scatter data for specific head ==========
    print(f"Computing scatter data for L{args.layer}H{args.head}...")
    q_block = q_tensor[args.layer, args.head, :token_count].to(device=device, dtype=torch.float32)
    k_block = k_tensor[args.layer, args.head, :token_count].to(device=device, dtype=torch.float32)

    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    dom_freq = compute_dominant_frequency(q_complex, k_complex)
    q_real, q_imag, k_real, k_imag, R_q_scatter, R_k_scatter = get_scatter_data(
        q_block, k_block, cos_table, sin_table, attention_scale, dom_freq
    )
    print(f"  Dominant freq: {dom_freq}, R_Q={R_q_scatter:.4f}, R_K={R_k_scatter:.4f}")

    # ========== Compute R for all heads ==========
    r_q_top1 = []
    r_k_top1 = []

    print("Computing R values for all heads...")
    for layer in range(num_layers):
        for head in range(num_heads):
            q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
            k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

            q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
            k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
            q_complex = to_complex_pairs(q_orig)
            k_complex = to_complex_pairs(k_orig)

            dom_freq = compute_dominant_frequency(q_complex, k_complex)
            r_q_top1.append(compute_mean_resultant_length(q_complex[:, dom_freq]))
            r_k_top1.append(compute_mean_resultant_length(k_complex[:, dom_freq]))

        if (layer + 1) % 10 == 0:
            print(f"  Processed {layer + 1}/{num_layers} layers")

    print(f"\nTop-1 Freq R:")
    print(f"  R_Q: [{min(r_q_top1):.4f}, {max(r_q_top1):.4f}], mean={np.mean(r_q_top1):.4f}")
    print(f"  R_K: [{min(r_k_top1):.4f}, {max(r_k_top1):.4f}], mean={np.mean(r_k_top1):.4f}")

    # ========== Create Figure ==========
    fig = plt.figure(figsize=(12, 5), dpi=args.dpi)
    gs = GridSpec(1, 2, figure=fig, wspace=0.3)

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
    })

    # ========== Panel A: Q/K scatter combined ==========
    ax_scatter = fig.add_subplot(gs[0, 0])
    style_ax(ax_scatter, grid_axis="both")
    ax_scatter.scatter(q_real, q_imag, s=8, alpha=0.3, color=color_q, edgecolors="none", label="Q")
    ax_scatter.scatter(k_real, k_imag, s=8, alpha=0.3, color=color_k, edgecolors="none", label="K")
    ax_scatter.axhline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax_scatter.axvline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax_scatter.set_xlim(-1.15, 1.15)
    ax_scatter.set_ylim(-1.15, 1.15)
    ax_scatter.set_aspect("equal", adjustable="box")
    ax_scatter.set_xticks([-1, 0, 1])
    ax_scatter.set_yticks([-1, 0, 1])
    ax_scatter.text(0.95, 0.95, f"$R_Q$={R_q_scatter:.2f}\n$R_K$={R_k_scatter:.2f}",
                    transform=ax_scatter.transAxes, fontsize=FONT_SIZE,
                    ha='right', va='top')
    leg = ax_scatter.legend(loc="lower left", fontsize=FONT_SIZE, frameon=False,
                            markerscale=2.5, handletextpad=0.2)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    ax_scatter.text(-0.15, 1.02, '(A)', transform=ax_scatter.transAxes,
                    fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom')

    # ========== Panel B: R distribution histogram ==========
    ax_hist = fig.add_subplot(gs[0, 1])
    style_ax(ax_hist, grid_axis="y")

    # Linear bins starting from 0.5
    r_min = 0.5
    r_max = 1.0
    r_bins = np.linspace(r_min, r_max, args.bins + 1)

    q_counts, _ = np.histogram(r_q_top1, bins=r_bins)
    k_counts, _ = np.histogram(r_k_top1, bins=r_bins)
    bin_centers = (r_bins[:-1] + r_bins[1:]) / 2

    ax_hist.plot(bin_centers, q_counts, color=color_q, linewidth=2.5, marker='o', markersize=4,
                 alpha=0.9, label=f'Q (mean={np.mean(r_q_top1):.2f})')
    ax_hist.plot(bin_centers, k_counts, color=color_k, linewidth=2.5, marker='s', markersize=4,
                 alpha=0.9, label=f'K (mean={np.mean(r_k_top1):.2f})')

    ax_hist.set_xlabel("R", fontsize=FONT_SIZE)
    ax_hist.set_ylabel("Count", fontsize=FONT_SIZE)
    ax_hist.legend(frameon=False, fontsize=FONT_SIZE, loc='upper left')
    ax_hist.text(-0.12, 1.02, '(B)', transform=ax_hist.transAxes,
                 fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
