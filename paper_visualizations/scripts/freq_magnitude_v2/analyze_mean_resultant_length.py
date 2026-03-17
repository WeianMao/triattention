"""Analyze Mean Resultant Length (R) for Q and K vectors.

This script computes and visualizes R_Q, R_K, and R_Q*R_K statistics across layers.
R (Mean Resultant Length) measures phase concentration: R = ||E[z]|| / E[||z||]

Output: Multiple figures showing R statistics vs layer index
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Mean Resultant Length for Q/K vectors"
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=None,
        help="Path to full_model_correlation_results.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--threshold", type=float, default=0.55, help="Correlation threshold")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Output directory for figures",
    )
    return parser.parse_args()


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(rotated, cos, sin, scale):
    z = rotated / scale
    return z * cos - rotate_half(z) * sin


def to_complex_pairs(tensor):
    num_freq = tensor.shape[-1] // 2
    return torch.complex(tensor[:, :num_freq].float(), tensor[:, num_freq:].float())


def compute_dominant_frequency(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
) -> int:
    """Find dominant frequency: argmax(|E[q_f]| * |E[k_f]|)."""
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)
    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)
    amp_product = torch.abs(q_mean) * torch.abs(k_mean)
    return amp_product.argmax().item()


def compute_mean_resultant_length(complex_vals: torch.Tensor) -> float:
    """Compute Mean Resultant Length: R = ||E[z]|| / E[||z||]."""
    mean_vec = complex_vals.mean()
    mean_norm = torch.abs(complex_vals).mean()
    if mean_norm < 1e-8:
        return 0.0
    return (torch.abs(mean_vec) / mean_norm).item()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_R_analysis")

    # Load correlation results
    if args.results_json:
        results_path = args.results_json
    else:
        results_path = Path("paper_visualizations/outputs/freq_magnitude_v2/full_model_correlation_results.json")

    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    num_layers = max(r['layer'] for r in results) + 1
    num_heads = max(r['head'] for r in results) + 1
    threshold = args.threshold

    # Per-layer correlation stats
    layer_above_thr_pct = []
    for layer in range(num_layers):
        layer_pearson = [r['ind_pearson'] for r in results if r['layer'] == layer]
        above_thr = sum(1 for p in layer_pearson if p > threshold)
        pct = above_thr / len(layer_pearson) * 100
        layer_above_thr_pct.append(pct)

    # Load trace data
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {args.trace_dir}")

    meta = json.loads(meta_path.read_text())
    token_count = int(meta["sequence_length"])

    device = torch.device(args.device)
    dtype = torch.float32

    # Load model config
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling

    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # Load Q/K tensors
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]
    head_dim = q_tensor.shape[-1]

    # Build RoPE tables
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=dtype)
    sin_table = sin_table[0].to(dtype=dtype)

    # Compute per-layer R_Q, R_K, R_Q*R_K
    print("Computing per-layer R_Q, R_K, R_Q*R_K...")
    layer_R_Q_values = {layer: [] for layer in range(num_layers)}
    layer_R_K_values = {layer: [] for layer in range(num_layers)}
    layer_R_QK_values = {layer: [] for layer in range(num_layers)}

    for layer in range(num_layers):
        for head in range(num_heads):
            q_block_lh = q_tensor[layer, head, :token_count].to(device=device, dtype=dtype)
            k_block_lh = k_tensor[layer, head, :token_count].to(device=device, dtype=dtype)

            # Find dominant frequency
            dominant_freq = compute_dominant_frequency(
                q_block_lh, k_block_lh, cos_table, sin_table, attention_scale
            )

            # Extract Q and K at dominant frequency and compute R
            q_orig = invert_rope(q_block_lh, cos_table, sin_table, attention_scale)
            k_orig = invert_rope(k_block_lh, cos_table, sin_table, attention_scale)
            q_complex = to_complex_pairs(q_orig)
            k_complex = to_complex_pairs(k_orig)
            q_dominant = q_complex[:, dominant_freq]
            k_dominant = k_complex[:, dominant_freq]
            R_q = compute_mean_resultant_length(q_dominant)
            R_k = compute_mean_resultant_length(k_dominant)

            layer_R_Q_values[layer].append(R_q)
            layer_R_K_values[layer].append(R_k)
            layer_R_QK_values[layer].append(R_q * R_k)

        print(f"  Layer {layer + 1}/{num_layers} done")

    # Compute per-layer statistics
    layer_mean_R_Q = [np.mean(layer_R_Q_values[layer]) for layer in range(num_layers)]
    layer_mean_R_K = [np.mean(layer_R_K_values[layer]) for layer in range(num_layers)]
    layer_mean_R_QK = [np.mean(layer_R_QK_values[layer]) for layer in range(num_layers)]

    print(f"R_Q range: [{min(layer_mean_R_Q):.3f}, {max(layer_mean_R_Q):.3f}]")
    print(f"R_K range: [{min(layer_mean_R_K):.3f}, {max(layer_mean_R_K):.3f}]")
    print(f"R_Q*R_K range: [{min(layer_mean_R_QK):.3f}, {max(layer_mean_R_QK):.3f}]")

    # Compute threshold statistics
    R_thresholds = [0.96, 0.97, 0.98, 0.99]
    layer_R_Q_above_thr = {}
    layer_R_QK_above_thr = {}
    for R_thr in R_thresholds:
        layer_R_Q_above_thr[R_thr] = []
        layer_R_QK_above_thr[R_thr] = []
        for layer in range(num_layers):
            above_Q = sum(1 for r in layer_R_Q_values[layer] if r > R_thr)
            above_QK = sum(1 for r in layer_R_QK_values[layer] if r > R_thr)
            pct_Q = above_Q / len(layer_R_Q_values[layer]) * 100
            pct_QK = above_QK / len(layer_R_QK_values[layer]) * 100
            layer_R_Q_above_thr[R_thr].append(pct_Q)
            layer_R_QK_above_thr[R_thr].append(pct_QK)
        print(f"R_Q > {R_thr}: min={min(layer_R_Q_above_thr[R_thr]):.1f}%, max={max(layer_R_Q_above_thr[R_thr]):.1f}%")
        print(f"R_Q*R_K > {R_thr}: min={min(layer_R_QK_above_thr[R_thr]):.1f}%, max={max(layer_R_QK_above_thr[R_thr]):.1f}%")

    # ========== Plotting ==========
    color_bar = (187 / 250, 130 / 250, 90 / 250)
    color_recon = (85 / 250, 104 / 250, 154 / 250)
    face_color = (231 / 250, 231 / 250, 240 / 250)
    FONT_SIZE = 14

    def style_ax(ax):
        ax.set_facecolor(face_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    output_dir = args.output_dir or Path("paper_visualizations/outputs/freq_magnitude_v2")
    output_dir.mkdir(parents=True, exist_ok=True)
    layers_arr = np.arange(num_layers)

    # Helper function to create single panel figure
    def create_panel_figure(y_data, y_label, output_name, ylim=(0, 100)):
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        style_ax(ax)
        ax.grid(True, axis='y', alpha=0.7, color='white', linewidth=1.5)

        # Bar chart (left Y axis)
        ax.bar(layers_arr, layer_above_thr_pct, color=color_bar, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.8)
        ax.set_xlabel('Layer Index', fontsize=FONT_SIZE)
        ax.set_ylabel(f'% Heads with $\\bar{{r}}$ > {threshold:.2f}', fontsize=FONT_SIZE)
        ax.set_xticks(layers_arr[::2])
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, num_layers - 0.5)

        # Curve (right Y axis)
        ax2 = ax.twinx()
        ax2.plot(layers_arr, y_data, color=color_recon, linewidth=2.5,
                 marker='o', markersize=4, zorder=1)
        ax2.set_ylabel(y_label, fontsize=FONT_SIZE, color=color_recon)
        ax2.tick_params(axis='y', labelsize=FONT_SIZE, labelcolor=color_recon)
        ax2.set_ylim(ylim)
        ax2.spines['right'].set_visible(False)
        ax2.spines['top'].set_visible(False)
        ax2.spines['left'].set_visible(False)
        ax2.spines['bottom'].set_visible(False)

        # Legend
        handles = [Patch(facecolor=color_bar, alpha=0.85, edgecolor='white'),
                   plt.Line2D([0], [0], color=color_recon, linewidth=2.5, marker='o', markersize=4)]
        labels = [f'% Heads with $\\bar{{r}}$ > {threshold:.2f}', y_label]
        ax2.legend(handles, labels, frameon=True, fontsize=FONT_SIZE, loc='lower left',
                   facecolor='white', edgecolor='none', framealpha=0.7)

        plt.tight_layout()
        fig.savefig(output_dir / output_name, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_dir / output_name}")

    # Generate Mean R figures
    create_panel_figure(layer_mean_R_Q, 'Mean $R_Q$', 'fig_panel_c_mean_R.png', ylim=(0.9, 1.0))
    create_panel_figure(layer_mean_R_QK, 'Mean $R_Q \\cdot R_K$', 'fig_panel_c_mean_R_QK.png', ylim=(0.9, 1.0))

    # Generate R_Q threshold figures
    for R_thr in R_thresholds:
        create_panel_figure(
            layer_R_Q_above_thr[R_thr],
            f'% $R_Q$ > {R_thr}',
            f'fig_panel_c_R_thr_{R_thr}.png'
        )

    # Generate R_Q*R_K threshold figures
    for R_thr in R_thresholds:
        create_panel_figure(
            layer_R_QK_above_thr[R_thr],
            f'% $R_Q \\cdot R_K$ > {R_thr}',
            f'fig_panel_c_R_QK_thr_{R_thr}.png'
        )

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
