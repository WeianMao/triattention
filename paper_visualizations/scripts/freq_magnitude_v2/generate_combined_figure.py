"""Generate combined figure for frequency-magnitude reconstruction analysis.

This script generates the main paper figure with three panels:
- (A) Reconstruction curve with GT error band for a representative head
- (B) Histogram of Individual-level Pearson across all 1152 heads
- (C) Per-layer percentage of heads above correlation threshold

Output: fig_freq_reconstruction_analysis.png
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
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
import numpy as np
import torch
from scipy import stats
from scipy.ndimage import gaussian_filter1d
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combined figure for frequency-magnitude reconstruction"
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
        help="Path to full_model_correlation_results.json (default: auto-detect in outputs)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--layer", type=int, default=0, help="Layer index for panel A")
    parser.add_argument("--head", type=int, default=0, help="Head index for panel A")
    parser.add_argument("--threshold", type=float, default=0.55, help="Correlation threshold for panel C")
    parser.add_argument("--max-distance", type=int, default=5000, help="Maximum token distance")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for the figure",
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


def compute_panel_a_data(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    inv_freq: torch.Tensor,
    max_dist: int,
    device: torch.device,
    num_samples: int = 200,
) -> dict:
    """Compute all data needed for panel A."""
    token_count = q_block.shape[0]
    head_dim = q_block.shape[-1]
    num_freq = head_dim // 2
    dtype = q_block.dtype

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

    actual_max_dist = min(max_dist, token_count - 1)
    distances = torch.arange(1, actual_max_dist + 1, device=device, dtype=torch.float32)

    # Reconstruction
    phase_matrix = distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)
    reconstructed = (torch.cos(phase_matrix) * amplitude.unsqueeze(0)).sum(dim=1)

    # GT curve (FFT)
    fft_len = 1 << (2 * token_count - 1).bit_length()
    q_fft = torch.fft.rfft(q_block.float(), n=fft_len, dim=0)
    k_fft = torch.fft.rfft(k_block.float(), n=fft_len, dim=0)
    corr = torch.fft.irfft((torch.conj(k_fft) * q_fft).sum(dim=1), n=fft_len, dim=0)
    all_counts = (token_count - torch.arange(0, actual_max_dist + 1, device=device, dtype=torch.float32)).clamp_min(1.0)
    gt_scores = corr[distances.long()] / all_counts[distances.long()]

    # GT with std (sampled for error band)
    sample_distances = torch.unique(torch.logspace(0, math.log10(actual_max_dist), num_samples, device=device).long())
    sample_distances = sample_distances[(sample_distances >= 1) & (sample_distances <= actual_max_dist)]

    q_float = q_block.float()
    k_float = k_block.float()
    gt_means_list, gt_stds_list = [], []
    for delta in sample_distances.long().tolist():
        if delta >= token_count:
            gt_means_list.append(0.0)
            gt_stds_list.append(0.0)
            continue
        q_slice = q_float[delta:]
        k_slice = k_float[:token_count - delta]
        scores = (q_slice * k_slice).sum(dim=1)
        gt_means_list.append(scores.mean().item())
        gt_stds_list.append(scores.std().item())

    # Individual correlation
    log_distances = torch.unique(torch.logspace(0, math.log10(actual_max_dist), 500, device=device).long())
    log_distances = log_distances[log_distances >= 1].float()
    recon_at_log = (torch.cos(log_distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)) * amplitude.unsqueeze(0)).sum(dim=1)
    dist_to_recon = {int(d): recon_at_log[i].item() for i, d in enumerate(log_distances.long().tolist())}

    all_actual, all_predicted = [], []
    for delta in log_distances.long().tolist():
        if delta >= token_count:
            continue
        q_slice = q_float[delta:]
        k_slice = k_float[:token_count - delta]
        scores = (q_slice * k_slice).sum(dim=1)
        all_actual.extend(scores.cpu().tolist())
        all_predicted.extend([dist_to_recon[delta]] * len(scores))

    all_actual_arr = np.array(all_actual)
    all_predicted_arr = np.array(all_predicted)
    if len(all_actual_arr) > 200000:
        indices = np.random.choice(len(all_actual_arr), 200000, replace=False)
        all_actual_arr = all_actual_arr[indices]
        all_predicted_arr = all_predicted_arr[indices]

    ind_pearson, _ = stats.pearsonr(all_actual_arr, all_predicted_arr)
    mean_pearson, _ = stats.pearsonr(gt_scores.cpu().numpy(), reconstructed.cpu().numpy())

    return {
        'distances': distances.cpu().numpy(),
        'reconstructed': reconstructed.cpu().numpy(),
        'gt_scores': gt_scores.cpu().numpy(),
        'sample_distances': sample_distances.cpu().numpy(),
        'gt_means': np.array(gt_means_list),
        'gt_stds': np.array(gt_stds_list),
        'ind_pearson': ind_pearson,
        'mean_pearson': mean_pearson,
    }


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_combined_fig")

    np.random.seed(42)

    # Load correlation results for panels B and C
    if args.results_json:
        results_path = args.results_json
    else:
        results_path = Path("paper_visualizations/outputs/freq_magnitude_v2/full_model_correlation_results.json")

    if not results_path.exists():
        raise FileNotFoundError(f"Results JSON not found: {results_path}")

    with open(results_path) as f:
        results = json.load(f)

    all_pearson = np.array([r['ind_pearson'] for r in results])
    num_layers = max(r['layer'] for r in results) + 1
    num_heads = max(r['head'] for r in results) + 1
    threshold = args.threshold

    # Per-layer stats
    layer_above_thr_pct = []
    for layer in range(num_layers):
        layer_pearson = [r['ind_pearson'] for r in results if r['layer'] == layer]
        above_thr = sum(1 for p in layer_pearson if p > threshold)
        pct = above_thr / len(layer_pearson) * 100
        layer_above_thr_pct.append(pct)

    # Load trace data for panel A
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

    q_block = q_tensor[args.layer, args.head, :token_count].to(device=device, dtype=dtype)
    k_block = k_tensor[args.layer, args.head, :token_count].to(device=device, dtype=dtype)

    # Build RoPE tables
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=dtype)
    sin_table = sin_table[0].to(dtype=dtype)

    # Compute panel A data
    panel_a = compute_panel_a_data(
        q_block, k_block, cos_table, sin_table,
        attention_scale, inv_freq, args.max_distance, device
    )

    # ========== Plotting ==========
    color_gt = (187 / 250, 130 / 250, 90 / 250)
    color_recon = (85 / 250, 104 / 250, 154 / 250)
    color_bar = (187 / 250, 130 / 250, 90 / 250)
    face_color = (231 / 250, 231 / 250, 240 / 250)

    FONT_SIZE = 14
    LABEL_FONT_SIZE = 18  # For (A), (B), (C) labels
    LABEL_FONT = 'DejaVu Sans'

    def style_ax(ax):
        ax.set_facecolor(face_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
        ax.set_axisbelow(True)
        ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    fig = plt.figure(figsize=(12, 8))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.2, 0.8], hspace=0.28, wspace=0.22)

    # (A) Reconstruction curve
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)

    ax_a.fill_between(panel_a['sample_distances'],
                      panel_a['gt_means'] - panel_a['gt_stds'],
                      panel_a['gt_means'] + panel_a['gt_stds'],
                      color=color_gt, alpha=0.25, linewidth=0, label='GT ± std')
    ax_a.plot(panel_a['distances'], panel_a['gt_scores'],
              color=color_gt, linestyle='--', linewidth=2.5, alpha=0.9, label='Ground Truth')
    ax_a.plot(panel_a['distances'], panel_a['reconstructed'],
              color=color_recon, linestyle=':', linewidth=2.5, alpha=0.9, label='Reconstructed')
    ax_a.set_xscale('log')
    ax_a.set_xlabel(r'QK Relative Position $\Delta$', fontsize=FONT_SIZE)
    ax_a.set_ylabel(r'$\langle q, k \rangle_\Delta$', fontsize=FONT_SIZE)
    ax_a.legend(frameon=False, fontsize=FONT_SIZE, loc='upper right')
    ax_a.text(0.03, 0.03,
              f"Individual Pearson $\\rho$ = {panel_a['ind_pearson']:.4f}\nTrendline Pearson $r$ = {panel_a['mean_pearson']:.4f}",
              transform=ax_a.transAxes, fontsize=FONT_SIZE,
              verticalalignment='bottom', horizontalalignment='left',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    ax_a.text(-0.08, 1.05, '(A)', transform=ax_a.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom', fontname=LABEL_FONT)

    # (B) Histogram
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)

    bin_edges = np.arange(-0.2, 1.025, 0.05)
    ax_b.hist(all_pearson, bins=bin_edges, color=color_recon, alpha=0.85, edgecolor='white', linewidth=0.8)
    ax_b.axvline(all_pearson.mean(), color='#E24A33', linestyle='--', linewidth=2.5,
                 label=f'Mean = {all_pearson.mean():.3f}')
    ax_b.set_xlabel('Individual Pearson $\\rho$', fontsize=FONT_SIZE)
    ax_b.set_ylabel('Count', fontsize=FONT_SIZE)
    ax_b.legend(frameon=False, fontsize=FONT_SIZE, loc='upper left')
    ax_b.set_xticks(np.arange(0, 1.1, 0.2))
    ax_b.set_xlim(-0.25, 1.05)
    ax_b.text(-0.08, 1.05, '(B)', transform=ax_b.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom', fontname=LABEL_FONT)

    # (C) Per-layer percentage
    ax_c = fig.add_subplot(gs[1, :])
    style_ax(ax_c)
    ax_c.grid(True, axis='y', alpha=0.7, color='white', linewidth=1.5)

    layers_arr = np.arange(num_layers)
    ax_c.bar(layers_arr, layer_above_thr_pct, color=color_bar, alpha=0.85, edgecolor='white', linewidth=0.5, width=0.8)

    smoothed = gaussian_filter1d(layer_above_thr_pct, sigma=2)
    ax_c.plot(layers_arr, smoothed, color=color_recon, linewidth=2.5, label='Smoothed trend')

    ax_c.set_xlabel('Layer Index', fontsize=FONT_SIZE)
    ax_c.set_ylabel(f'% Heads with $\\rho$ > {threshold}', fontsize=FONT_SIZE)
    ax_c.set_xticks(layers_arr[::2])
    ax_c.set_ylim(0, 100)

    handles, labels = ax_c.get_legend_handles_labels()
    handles.insert(0, Patch(facecolor=color_bar, alpha=0.85, edgecolor='white', label='Per-layer percentage'))
    labels.insert(0, 'Per-layer percentage')
    ax_c.legend(handles, labels, frameon=False, fontsize=FONT_SIZE, loc='upper right', bbox_to_anchor=(1.0, 1.05))
    ax_c.text(-0.04, 1.08, '(C)', transform=ax_c.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom', fontname=LABEL_FONT)

    plt.tight_layout()

    # Save
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = Path("paper_visualizations/outputs/freq_magnitude_v2/fig_freq_reconstruction_analysis.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Figure saved to {output_path}")
    print(f"\nPanel (A): Layer {args.layer}, Head {args.head}")
    print(f"  Individual Pearson r = {panel_a['ind_pearson']:.4f}")
    print(f"  Mean Pearson r = {panel_a['mean_pearson']:.4f}")
    print(f"\nPanel (B): {len(all_pearson)} heads")
    print(f"  Mean = {all_pearson.mean():.4f}, Std = {all_pearson.std():.4f}")
    print(f"\nPanel (C): Threshold = {threshold}")


if __name__ == "__main__":
    main()
