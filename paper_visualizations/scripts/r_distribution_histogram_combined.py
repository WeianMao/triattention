"""Generate combined R value distribution histogram.

Compares 64-dim full vector R vs Top-1 frequency R,
with Q and K distinguished by different colors in the same subplot.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 14
LABEL_FONT_SIZE = 18
color_q = (85 / 250, 104 / 250, 154 / 250)    # blue for Q
color_k = (187 / 250, 130 / 250, 90 / 250)    # warm brown for K
face_color = (231 / 250, 231 / 250, 240 / 250)


def style_ax(ax):
    """Apply consistent styling to axes."""
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)


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
def compute_high_dim_R(complex_matrix: torch.Tensor) -> float:
    """Compute Mean Resultant Length for 64-dimensional complex vectors."""
    norms = torch.norm(complex_matrix.abs(), dim=1)
    mean_norm = norms.mean()
    mean_vec = complex_matrix.mean(dim=0)
    mean_vec_norm = torch.norm(mean_vec.abs())
    if mean_norm < 1e-8:
        return 0.0
    return (mean_vec_norm / mean_norm).item()


def compute_mean_resultant_length(complex_vals: torch.Tensor) -> float:
    """Compute Mean Resultant Length for 1D complex values."""
    mean_vec = complex_vals.mean()
    mean_norm = torch.abs(complex_vals).mean()
    if mean_norm < 1e-8:
        return 0.0
    return (torch.abs(mean_vec) / mean_norm).item()


def compute_dominant_frequency(q_complex: torch.Tensor, k_complex: torch.Tensor, use_mean_norm: bool = False) -> int:
    """Compute dominant frequency using amplitude product.

    Args:
        use_mean_norm: If False (default), use |E[z]| (norm of mean vector).
                       If True, use E[|z|] (mean of norms).
    """
    if use_mean_norm:
        # E[|q|] * E[|k|] - mean of norms
        q_amp = torch.abs(q_complex).mean(dim=0)
        k_amp = torch.abs(k_complex).mean(dim=0)
    else:
        # |E[q]| * |E[k]| - norm of mean vector
        q_amp = torch.abs(q_complex.mean(dim=0))
        k_amp = torch.abs(k_complex.mean(dim=0))
    amp_product = q_amp * k_amp
    return amp_product.argmax().item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate combined R distribution histogram")
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
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument("--bins", type=int, default=20, help="Number of histogram bins")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for figure",
    )
    parser.add_argument(
        "--use-mean-norm",
        action="store_true",
        help="Use E[|z|] (mean of norms) instead of |E[z]| (norm of mean) for dominant freq selection",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_r_hist_comb")

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/r_distribution")
        output_path = output_dir / "r_distribution_histogram_combined.png"
    else:
        output_path = args.output_path

    device = torch.device(args.device)

    # Load data
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {args.trace_dir}")

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
    print(f"Model: {num_layers} layers, {num_heads} heads per layer")

    # Compute R for all heads
    r_q_64dim = []
    r_k_64dim = []
    r_q_top1 = []
    r_k_top1 = []

    print("Computing R values for all heads...")
    for layer in range(num_layers):
        for head in range(num_heads):
            q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
            k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

            # Invert RoPE
            q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
            k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

            # Convert to complex
            q_complex = to_complex_pairs(q_orig)
            k_complex = to_complex_pairs(k_orig)

            # 64-dim R
            r_q_64dim.append(compute_high_dim_R(q_complex))
            r_k_64dim.append(compute_high_dim_R(k_complex))

            # Top-1 frequency R
            dom_freq = compute_dominant_frequency(q_complex, k_complex, use_mean_norm=args.use_mean_norm)
            r_q_top1.append(compute_mean_resultant_length(q_complex[:, dom_freq]))
            r_k_top1.append(compute_mean_resultant_length(k_complex[:, dom_freq]))

        if (layer + 1) % 10 == 0:
            print(f"  Processed {layer + 1}/{num_layers} layers")

    print(f"Total heads: {len(r_q_64dim)}")
    print(f"\n64-dim R:")
    print(f"  R_Q range: [{min(r_q_64dim):.4f}, {max(r_q_64dim):.4f}], mean: {np.mean(r_q_64dim):.4f}")
    print(f"  R_K range: [{min(r_k_64dim):.4f}, {max(r_k_64dim):.4f}], mean: {np.mean(r_k_64dim):.4f}")
    print(f"\nTop-1 Freq R:")
    print(f"  R_Q range: [{min(r_q_top1):.4f}, {max(r_q_top1):.4f}], mean: {np.mean(r_q_top1):.4f}")
    print(f"  R_K range: [{min(r_k_top1):.4f}, {max(r_k_top1):.4f}], mean: {np.mean(r_k_top1):.4f}")

    # Create figure: single plot (square)
    fig, ax = plt.subplots(1, 1, figsize=(6, 6), dpi=args.dpi)

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
    })

    style_ax(ax)

    # Linear bins starting from 0.5
    r_min = 0.5
    r_max = 1.0
    r_bins = np.linspace(r_min, r_max, args.bins + 1)

    # Compute histogram counts
    q_counts, _ = np.histogram(r_q_top1, bins=r_bins)
    k_counts, _ = np.histogram(r_k_top1, bins=r_bins)

    # Compute bin centers
    bin_centers = (r_bins[:-1] + r_bins[1:]) / 2

    # Draw line plots
    ax.plot(bin_centers, q_counts, color=color_q, linewidth=2.5, marker='o', markersize=6,
            alpha=0.9, label=f'Q (mean={np.mean(r_q_top1):.2f})')
    ax.plot(bin_centers, k_counts, color=color_k, linewidth=2.5, marker='s', markersize=6,
            alpha=0.9, label=f'K (mean={np.mean(r_k_top1):.2f})')

    ax.set_xlabel("R value", fontsize=FONT_SIZE)
    ax.set_ylabel("Count", fontsize=FONT_SIZE)
    method = "E[|z|]" if args.use_mean_norm else "|E[z]|"
    ax.set_title(f"Top-1 Frequency R Distribution (select by {method})", fontsize=FONT_SIZE + 2)
    ax.legend(frameon=False, fontsize=FONT_SIZE, loc='upper left')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
