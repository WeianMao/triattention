"""Analyze Spearman correlation for attention reconstruction.

This script computes and visualizes Spearman rho statistics across layers,
comparing with Pearson r to see if Spearman provides better layer differentiation.

Output: Multiple figures showing Spearman statistics vs layer index
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np
import torch
from scipy import stats
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Spearman correlation for attention reconstruction"
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
        help="Path to full_model_correlation_results.json (for Pearson baseline)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--threshold", type=float, default=0.55, help="Correlation threshold for Pearson bars")
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


def compute_per_query_spearman(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    inv_freq: torch.Tensor,
    device: torch.device,
) -> float:
    """Compute per-query Spearman correlation with log-spaced distance sampling."""
    token_count = q_block.shape[0]
    head_dim = q_block.shape[-1]
    num_freq = head_dim // 2

    # Invert RoPE
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    # Complex representation
    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    # Mean vectors
    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    phi_f = torch.angle(q_mean) - torch.angle(k_mean)
    amplitude = torch.abs(q_mean) * torch.abs(k_mean)
    omega = inv_freq[:num_freq].to(device=device, dtype=torch.float32)

    # Precompute reconstruction values for all distances
    all_distances = torch.arange(1, token_count, device=device, dtype=torch.float32)
    recon_all = (torch.cos(all_distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)) * amplitude.unsqueeze(0)).sum(dim=1)
    recon_dict = {int(d): recon_all[i].item() for i, d in enumerate(all_distances.long().tolist())}

    # Per-Query correlation with log-spaced sampling
    q_float = q_block.float()
    k_float = k_block.float()
    min_history = 50
    num_log_samples = 50
    num_query_samples = 500

    # Sample query positions
    query_positions = torch.unique(torch.logspace(
        math.log10(min_history), math.log10(token_count - 1), num_query_samples, device=device
    ).long())
    query_positions = query_positions[(query_positions >= min_history) & (query_positions < token_count)]

    per_query_spearmans = []

    for query_pos in query_positions.tolist():
        log_distances = torch.unique(torch.logspace(0, math.log10(query_pos), num_log_samples, device=device).long())
        log_distances = log_distances[(log_distances >= 1) & (log_distances <= query_pos)]

        if len(log_distances) < 3:
            continue

        key_positions = query_pos - log_distances
        q_vec = q_float[query_pos]
        k_vecs = k_float[key_positions]
        actual_scores = (q_vec.unsqueeze(0) * k_vecs).sum(dim=1)

        predicted_scores = torch.tensor([recon_dict[int(d)] for d in log_distances.tolist()], device=device)

        # Compute Spearman
        rho, _ = stats.spearmanr(actual_scores.cpu().numpy(), predicted_scores.cpu().numpy())
        if not np.isnan(rho):
            per_query_spearmans.append(rho)

    return np.mean(per_query_spearmans) if per_query_spearmans else 0.0


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_spearman")

    # Load Pearson results for baseline comparison
    if args.results_json:
        results_path = args.results_json
    else:
        results_path = Path("paper_visualizations/outputs/freq_magnitude_v2/full_model_correlation_results.json")

    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    with open(results_path) as f:
        pearson_results = json.load(f)

    num_layers = max(r['layer'] for r in pearson_results) + 1
    num_heads = max(r['head'] for r in pearson_results) + 1
    threshold = args.threshold

    # Per-layer Pearson stats (baseline)
    layer_above_thr_pct = []
    for layer in range(num_layers):
        layer_pearson = [r['ind_pearson'] for r in pearson_results if r['layer'] == layer]
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
    inv_freq = rotary.inv_freq.to(torch.float64)
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

    # Compute per-head Spearman
    print("Computing per-head Spearman correlation...")
    layer_spearman_values = {layer: [] for layer in range(num_layers)}

    for layer in range(num_layers):
        for head in range(num_heads):
            q_block_lh = q_tensor[layer, head, :token_count].to(device=device, dtype=dtype)
            k_block_lh = k_tensor[layer, head, :token_count].to(device=device, dtype=dtype)

            spearman = compute_per_query_spearman(
                q_block_lh, k_block_lh, cos_table, sin_table,
                attention_scale, inv_freq, device
            )
            layer_spearman_values[layer].append(spearman)

        print(f"  Layer {layer + 1}/{num_layers} done")

    # Compute per-layer statistics
    layer_mean_spearman = [np.mean(layer_spearman_values[layer]) for layer in range(num_layers)]
    print(f"Spearman range: [{min(layer_mean_spearman):.3f}, {max(layer_mean_spearman):.3f}]")

    # Compute threshold statistics
    spearman_thresholds = [0.55, 0.60, 0.65, 0.70]
    layer_spearman_above_thr = {}
    for thr in spearman_thresholds:
        layer_spearman_above_thr[thr] = []
        for layer in range(num_layers):
            above = sum(1 for s in layer_spearman_values[layer] if s > thr)
            pct = above / len(layer_spearman_values[layer]) * 100
            layer_spearman_above_thr[thr].append(pct)
        print(f"Spearman > {thr}: min={min(layer_spearman_above_thr[thr]):.1f}%, max={max(layer_spearman_above_thr[thr]):.1f}%")

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

    # Helper function
    def create_panel_figure(y_data, y_label, output_name, ylim=(0, 100)):
        fig = plt.figure(figsize=(12, 3))
        ax = fig.add_subplot(111)
        style_ax(ax)
        ax.grid(True, axis='y', alpha=0.7, color='white', linewidth=1.5)

        # Bar chart (Pearson baseline)
        ax.bar(layers_arr, layer_above_thr_pct, color=color_bar, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.8)
        ax.set_xlabel('Layer Index', fontsize=FONT_SIZE)
        ax.set_ylabel(f'% Heads with Pearson $\\bar{{r}}$ > {threshold:.2f}', fontsize=FONT_SIZE)
        ax.set_xticks(layers_arr[::2])
        ax.set_ylim(0, 100)
        ax.set_xlim(-0.5, num_layers - 0.5)

        # Spearman curve
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
        labels = [f'% Heads with Pearson $\\bar{{r}}$ > {threshold:.2f}', y_label]
        ax2.legend(handles, labels, frameon=True, fontsize=FONT_SIZE, loc='lower left',
                   facecolor='white', edgecolor='none', framealpha=0.7)

        plt.tight_layout()
        fig.savefig(output_dir / output_name, dpi=args.dpi, bbox_inches='tight')
        plt.close(fig)
        print(f"Saved: {output_dir / output_name}")

    # Generate Mean Spearman figure
    create_panel_figure(layer_mean_spearman, 'Mean Spearman $\\bar{\\rho}$',
                        'fig_panel_c_mean_spearman.png', ylim=(0, 1))

    # Generate Spearman threshold figures
    for thr in spearman_thresholds:
        create_panel_figure(
            layer_spearman_above_thr[thr],
            f'% Spearman $\\rho$ > {thr}',
            f'fig_panel_c_spearman_thr_{thr}.png'
        )

    # Save Spearman results
    spearman_results = []
    for layer in range(num_layers):
        for head in range(num_heads):
            spearman_results.append({
                'layer': layer,
                'head': head,
                'spearman': layer_spearman_values[layer][head],
            })

    results_output = output_dir / "spearman_correlation_results.json"
    with open(results_output, 'w') as f:
        json.dump(spearman_results, f, indent=2)
    print(f"\nSpearman results saved to {results_output}")

    print("\nAll figures generated successfully!")


if __name__ == "__main__":
    main()
