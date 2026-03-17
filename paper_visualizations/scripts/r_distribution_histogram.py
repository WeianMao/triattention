"""Generate R value distribution histogram for all heads.

Computes the 64-dimensional Mean Resultant Length R for all heads
in the pre-RoPE space and visualizes the distribution.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
color_mean = '#E24A33'                         # red for mean line
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


# ============ High-dimensional R Computation ============
def compute_high_dim_R(complex_matrix: torch.Tensor) -> float:
    """
    Compute Mean Resultant Length for 64-dimensional complex vectors.

    Args:
        complex_matrix: shape [seq_len, 64] complex tensor

    Returns:
        R = ||mean(v_i)||_2 / mean(||v_i||_2)
    """
    # L2 norm of each position's 64-dim complex vector
    norms = torch.norm(complex_matrix.abs(), dim=1)  # [seq_len]
    mean_norm = norms.mean()

    # Mean vector across all positions
    mean_vec = complex_matrix.mean(dim=0)  # [64]
    mean_vec_norm = torch.norm(mean_vec.abs())

    if mean_norm < 1e-8:
        return 0.0
    return (mean_vec_norm / mean_norm).item()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate R distribution histogram")
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
    parser.add_argument("--bins", type=int, default=30, help="Number of histogram bins")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for figure",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_r_hist")

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/r_distribution")
        output_path = output_dir / "r_distribution_histogram.png"
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
    r_q_values = []
    r_k_values = []

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

            # Compute high-dim R
            r_q = compute_high_dim_R(q_complex)
            r_k = compute_high_dim_R(k_complex)

            r_q_values.append(r_q)
            r_k_values.append(r_k)

        if (layer + 1) % 10 == 0:
            print(f"  Processed {layer + 1}/{num_layers} layers")

    print(f"Total heads: {len(r_q_values)}")
    print(f"R_Q range: [{min(r_q_values):.4f}, {max(r_q_values):.4f}]")
    print(f"R_K range: [{min(r_k_values):.4f}, {max(r_k_values):.4f}]")

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=args.dpi)
    fig.patch.set_facecolor(face_color)

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
    })

    import numpy as np
    r_q_arr = np.array(r_q_values)
    r_k_arr = np.array(r_k_values)

    # R_Q histogram
    ax_q = axes[0]
    style_ax(ax_q)
    ax_q.hist(r_q_values, bins=args.bins, color=color_q, edgecolor="white", alpha=0.85, linewidth=0.8)
    ax_q.axvline(r_q_arr.mean(), color=color_mean, linestyle='--', linewidth=2.5,
                 label=f'Mean = {r_q_arr.mean():.2f}')
    ax_q.set_yscale("log")
    ax_q.yaxis.set_minor_locator(plt.NullLocator())
    ax_q.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax_q.set_xlabel("R value", fontsize=FONT_SIZE)
    ax_q.set_ylabel("Count", fontsize=FONT_SIZE)
    ax_q.legend(frameon=False, fontsize=FONT_SIZE, loc='upper left')
    ax_q.text(-0.08, 1.05, '(A)', transform=ax_q.transAxes, fontsize=LABEL_FONT_SIZE,
              fontweight='bold', va='bottom')

    # R_K histogram
    ax_k = axes[1]
    style_ax(ax_k)
    ax_k.hist(r_k_values, bins=args.bins, color=color_k, edgecolor="white", alpha=0.85, linewidth=0.8)
    ax_k.axvline(r_k_arr.mean(), color=color_mean, linestyle='--', linewidth=2.5,
                 label=f'Mean = {r_k_arr.mean():.2f}')
    ax_k.set_yscale("log")
    ax_k.yaxis.set_minor_locator(plt.NullLocator())
    ax_k.yaxis.set_major_formatter(plt.ScalarFormatter())
    ax_k.set_xlabel("R value", fontsize=FONT_SIZE)
    ax_k.set_ylabel("Count", fontsize=FONT_SIZE)
    ax_k.legend(frameon=False, fontsize=FONT_SIZE, loc='upper left')
    ax_k.text(-0.08, 1.05, '(B)', transform=ax_k.transAxes, fontsize=LABEL_FONT_SIZE,
              fontweight='bold', va='bottom')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight', facecolor=face_color)
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
