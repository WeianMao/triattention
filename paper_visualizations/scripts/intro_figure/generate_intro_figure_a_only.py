"""Generate simplified Figure A - Q/K distribution scatter plot only.

Removes R values and Pre-RoPE label from the original figure.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
LABEL_FONT_SIZE = 18
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
    ax.grid(True, axis=grid_axis, alpha=0.7, color="white", linewidth=1.5)


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


def compute_dominant_frequency(q_complex: torch.Tensor, k_complex: torch.Tensor, use_mean_norm: bool = True) -> int:
    """Compute dominant frequency using amplitude product."""
    if use_mean_norm:
        q_amp = torch.abs(q_complex).mean(dim=0)
        k_amp = torch.abs(k_complex).mean(dim=0)
    else:
        q_amp = torch.abs(q_complex.mean(dim=0))
        k_amp = torch.abs(k_complex.mean(dim=0))
    amp_product = q_amp * k_amp
    return amp_product.argmax().item()


def get_scatter_data_pre_rope(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    freq_idx: int,
    max_points: int = 5000,
    filter_radius: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Get Q/K scatter data for a specific frequency (Pre-RoPE)."""
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    q_vals = q_complex[:, freq_idx]
    k_vals = k_complex[:, freq_idx]

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

    # Filter out points within filter_radius from origin
    q_real = q_vals.real.cpu().numpy()
    q_imag = q_vals.imag.cpu().numpy()
    k_real = k_vals.real.cpu().numpy()
    k_imag = k_vals.imag.cpu().numpy()

    q_dist = np.sqrt(q_real**2 + q_imag**2)
    k_dist = np.sqrt(k_real**2 + k_imag**2)

    q_mask = q_dist >= filter_radius
    k_mask = k_dist >= filter_radius

    return (
        q_real[q_mask],
        q_imag[q_mask],
        k_real[k_mask],
        k_imag[k_mask],
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate simplified Figure A - Q/K scatter only")
    parser.add_argument(
        "--trace-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
            Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0010_trace23"),
            Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0012_trace61"),
        ],
        help="Trace directories for scatter plot",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument("--scatter-layer", type=int, default=3, help="Layer for scatter plot")
    parser.add_argument("--scatter-head", type=int, default=0, help="Head for scatter plot")
    parser.add_argument("--output-path", type=Path, default=None, help="Output path for figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_intro_fig_a")

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/intro_figure")
        output_path = output_dir / "fig_intro_a_only.png"
    else:
        output_path = args.output_path

    device = torch.device(args.device)
    np.random.seed(42)

    # ========== Load Model Config ==========
    print("Loading model config...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling

    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # ========== Collect Q/K Scatter Data ==========
    print(f"\n=== Q/K Scatter (L{args.scatter_layer}H{args.scatter_head}, {len(args.trace_dirs)} traces) ===")
    all_q_real, all_q_imag = [], []
    all_k_real, all_k_imag = [], []

    fixed_dom_freq = None

    for trace_dir in args.trace_dirs:
        print(f"  Loading {trace_dir.name}...")
        qk_path = trace_dir / "qk.pt"
        meta_path = trace_dir / "metadata.json"

        if not qk_path.exists() or not meta_path.exists():
            print(f"    WARNING: Missing files in {trace_dir}, skipping")
            continue

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        token_count = int(meta["sequence_length"])

        head_dim = 128
        position_ids = torch.arange(token_count, device=device).unsqueeze(0)
        base = torch.zeros(1, token_count, head_dim, device=device, dtype=torch.float32)
        cos_table, sin_table = rotary(base, position_ids)
        cos_table = cos_table[0].to(dtype=torch.float32)
        sin_table = sin_table[0].to(dtype=torch.float32)

        data = torch.load(qk_path, map_location="cpu")
        q_tensor = data["q"]
        k_tensor = data["k"]

        q_block = q_tensor[args.scatter_layer, args.scatter_head, :token_count].to(device=device, dtype=torch.float32)
        k_block = k_tensor[args.scatter_layer, args.scatter_head, :token_count].to(device=device, dtype=torch.float32)

        # Compute dominant frequency only from first trace
        q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
        k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
        q_complex = to_complex_pairs(q_orig)
        k_complex = to_complex_pairs(k_orig)

        if fixed_dom_freq is None:
            fixed_dom_freq = compute_dominant_frequency(q_complex, k_complex, use_mean_norm=True)
            print(f"  Using fixed dominant frequency: {fixed_dom_freq}")
        dom_freq = fixed_dom_freq

        # Get scatter data (without R values)
        q_real, q_imag, k_real, k_imag = get_scatter_data_pre_rope(
            q_block, k_block, cos_table, sin_table, attention_scale, dom_freq, max_points=3000
        )

        all_q_real.extend(q_real)
        all_q_imag.extend(q_imag)
        all_k_real.extend(k_real)
        all_k_imag.extend(k_imag)

    print(f"  Combined: {len(all_q_real)} Q points, {len(all_k_real)} K points")

    # ========== Create Figure ==========
    print("\n=== Creating figure ===")
    fig, ax = plt.subplots(1, 1, figsize=(5, 5), dpi=args.dpi)
    plt.subplots_adjust(left=0.15, right=0.95, top=0.92, bottom=0.12)

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
    })

    style_ax(ax)

    ax.scatter(all_q_real, all_q_imag, s=6, alpha=0.25, color=color_q, edgecolors="none", label="Q")
    ax.scatter(all_k_real, all_k_imag, s=6, alpha=0.25, color=color_k, edgecolors="none", label="K")
    ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax.set_xlim(-1.15, 1.15)
    ax.set_ylim(-1.15, 1.15)
    ax.set_box_aspect(1)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xlabel("Re", fontsize=FONT_SIZE)
    ax.set_ylabel("Im", fontsize=FONT_SIZE)

    # Legend without R values
    leg = ax.legend(loc="lower left", fontsize=FONT_SIZE, frameon=False, markerscale=2.5, handletextpad=0.2)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)

    # Title without Pre-RoPE
    ax.set_title('Q/K Distribution', fontsize=FONT_SIZE, pad=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
