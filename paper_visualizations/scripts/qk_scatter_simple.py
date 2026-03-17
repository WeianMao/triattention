"""Generate simplified Q/K scatter plot for dominant frequency.

Based on Panel A from r_distribution_with_scatter.py with modifications:
- No axis labels/ticks
- Larger QK legend
- No R_Q/R_K text
- Tight bounds around data + origin with small margin
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Force Times New Roman font for all text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'

import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 45  # Larger legend font
color_q = (85 / 250, 104 / 250, 154 / 250)    # blue for Q
color_k = (187 / 250, 130 / 250, 90 / 250)    # warm brown for K
face_color = (231 / 250, 231 / 250, 240 / 250)


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
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simplified Q/K scatter plot")
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
    parser.add_argument("--layer", type=int, default=32, help="Layer for scatter plot")
    parser.add_argument("--head", type=int, default=27, help="Head for scatter plot")
    parser.add_argument("--output-path", type=Path, default=None)
    parser.add_argument("--margin", type=float, default=0.08, help="Margin around data bounds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_qk_scatter_simple")

    if args.output_path is None:
        output_path = Path("paper_visualizations/outputs/r_distribution/qk_scatter_simple.png")
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

    # ========== Compute scatter data for specific head ==========
    print(f"Computing scatter data for L{args.layer}H{args.head}...")
    q_block = q_tensor[args.layer, args.head, :token_count].to(device=device, dtype=torch.float32)
    k_block = k_tensor[args.layer, args.head, :token_count].to(device=device, dtype=torch.float32)

    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    dom_freq = compute_dominant_frequency(q_complex, k_complex)
    q_real, q_imag, k_real, k_imag = get_scatter_data(
        q_block, k_block, cos_table, sin_table, attention_scale, dom_freq
    )
    print(f"  Dominant freq: {dom_freq}")

    # ========== Compute tight bounds including origin ==========
    all_x = np.concatenate([q_real, k_real, [0.0]])  # include origin
    all_y = np.concatenate([q_imag, k_imag, [0.0]])  # include origin

    x_min, x_max = all_x.min(), all_x.max()
    y_min, y_max = all_y.min(), all_y.max()

    # Make square and add margin
    x_range = x_max - x_min
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    # Center the square
    x_center = (x_max + x_min) / 2
    y_center = (y_max + y_min) / 2

    half_size = max_range / 2 * (1 + args.margin)

    xlim = (x_center - half_size, x_center + half_size)
    ylim = (y_center - half_size, y_center + half_size)

    print(f"  Data bounds: x=[{x_min:.3f}, {x_max:.3f}], y=[{y_min:.3f}, {y_max:.3f}]")
    print(f"  Plot limits: x={xlim}, y={ylim}")

    # ========== Create Figure ==========
    fig, ax = plt.subplots(figsize=(6, 6), dpi=args.dpi)

    # Set background color
    ax.set_facecolor(face_color)

    # Remove all spines
    for spine in ax.spines.values():
        spine.set_visible(False)

    # Remove all ticks and labels
    ax.set_xticks([])
    ax.set_yticks([])
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # Add manual grid lines at +1.0 and -1.0
    ax.set_axisbelow(True)
    for gx in [-1.0, 1.0]:
        if xlim[0] < gx < xlim[1]:
            ax.axvline(gx, color="white", linewidth=1.5, alpha=0.6, zorder=0)
    for gy in [-1.0, 1.0]:
        if ylim[0] < gy < ylim[1]:
            ax.axhline(gy, color="white", linewidth=1.5, alpha=0.6, zorder=0)

    # Scatter plot
    ax.scatter(q_real, q_imag, s=10, alpha=0.35, color=color_q, edgecolors="none", label="Q")
    ax.scatter(k_real, k_imag, s=10, alpha=0.35, color=color_k, edgecolors="none", label="K")

    # Origin cross lines (thicker)
    ax.axhline(0.0, color="gray", linewidth=2.5, alpha=0.7)
    ax.axvline(0.0, color="gray", linewidth=2.5, alpha=0.7)

    # Set limits
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_aspect("equal", adjustable="box")

    # Legend - larger, no frame, upper left corner (aggressive positioning)
    leg = ax.legend(loc="upper left", fontsize=FONT_SIZE, frameon=False,
                    markerscale=6.0, handletextpad=0.3, borderpad=0.0,
                    bbox_to_anchor=(-0.08, 1.0))
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
