"""Generate Introduction figure with 5 subplots.

(a) Pre-RoPE Q/K distribution (L32H27, across 3 traces)
(b) R concentration histogram (Q/K merged, bar chart)
(c) Reconstruction curve Case 1 (L0H0)
(d) Reconstruction curve Case 2 (L1H18)
(e) Pearson correlation histogram
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from scipy import stats
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
color_mean = '#E24A33'                         # red for mean line
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


# ============ R Computation ============
def compute_mean_resultant_length(complex_vals: torch.Tensor) -> float:
    """Compute Mean Resultant Length: R = ||E[z]|| / E[||z||]"""
    mean_vec = complex_vals.mean()
    mean_norm = torch.abs(complex_vals).mean()
    if mean_norm < 1e-8:
        return 0.0
    return (torch.abs(mean_vec) / mean_norm).item()


def compute_dominant_frequency(q_complex: torch.Tensor, k_complex: torch.Tensor, use_mean_norm: bool = True) -> int:
    """Compute dominant frequency using amplitude product.

    Args:
        use_mean_norm: If True, use E[|z|] (mean of norms).
                       If False, use |E[z]| (norm of mean vector).
    """
    if use_mean_norm:
        q_amp = torch.abs(q_complex).mean(dim=0)
        k_amp = torch.abs(k_complex).mean(dim=0)
    else:
        q_amp = torch.abs(q_complex.mean(dim=0))
        k_amp = torch.abs(k_complex.mean(dim=0))
    amp_product = q_amp * k_amp
    return amp_product.argmax().item()


def get_scatter_data(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    freq_idx: int,
    max_points: int = 5000,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Get Q/K scatter data for a specific frequency."""
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    q_vals = q_complex[:, freq_idx]
    k_vals = k_complex[:, freq_idx]

    R_q = compute_mean_resultant_length(q_vals)
    R_k = compute_mean_resultant_length(k_vals)

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


# ============ Reconstruction Computation ============
def compute_reconstruction_data(
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
    """Compute reconstruction curve data."""
    token_count = q_block.shape[0]
    head_dim = q_block.shape[-1]
    num_freq = head_dim // 2

    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    phi_f = torch.angle(q_mean) - torch.angle(k_mean)
    amplitude = torch.abs(q_mean) * torch.abs(k_mean)
    omega = inv_freq[:num_freq].to(device=device, dtype=torch.float32)

    actual_max_dist = min(max_dist, token_count - 1)
    distances = torch.arange(1, actual_max_dist + 1, device=device, dtype=torch.float32)

    phase_matrix = distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)
    reconstructed = (torch.cos(phase_matrix) * amplitude.unsqueeze(0)).sum(dim=1)

    fft_len = 1 << (2 * token_count - 1).bit_length()
    q_fft = torch.fft.rfft(q_block.float(), n=fft_len, dim=0)
    k_fft = torch.fft.rfft(k_block.float(), n=fft_len, dim=0)
    corr = torch.fft.irfft((torch.conj(k_fft) * q_fft).sum(dim=1), n=fft_len, dim=0)
    all_counts = (token_count - torch.arange(0, actual_max_dist + 1, device=device, dtype=torch.float32)).clamp_min(1.0)
    gt_scores = corr[distances.long()] / all_counts[distances.long()]

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

    min_history = 50
    num_log_samples = 50
    num_query_samples = 500
    per_query_pearsons = []

    all_distances_recon = torch.arange(1, token_count, device=device, dtype=torch.float32)
    recon_all = (torch.cos(all_distances_recon.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)) * amplitude.unsqueeze(0)).sum(dim=1)
    recon_dict = {int(d): recon_all[i].item() for i, d in enumerate(all_distances_recon.long().tolist())}

    query_positions = torch.unique(torch.logspace(
        math.log10(min_history), math.log10(token_count - 1), num_query_samples, device=device
    ).long())
    query_positions = query_positions[(query_positions >= min_history) & (query_positions < token_count)]

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

        r, _ = stats.pearsonr(actual_scores.cpu().numpy(), predicted_scores.cpu().numpy())
        if not np.isnan(r):
            per_query_pearsons.append(r)

    per_query_pearson = np.mean(per_query_pearsons) if per_query_pearsons else 0.0

    return {
        'distances': distances.cpu().numpy(),
        'reconstructed': reconstructed.cpu().numpy(),
        'gt_scores': gt_scores.cpu().numpy(),
        'sample_distances': sample_distances.cpu().numpy(),
        'gt_means': np.array(gt_means_list),
        'gt_stds': np.array(gt_stds_list),
        'per_query_pearson': per_query_pearson,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Introduction figure with 5 subplots")
    parser.add_argument(
        "--trace-dirs",
        type=Path,
        nargs="+",
        default=[
            Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
            Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0010_trace23"),
            Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0012_trace61"),
        ],
        help="Trace directories for subplot (a)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument(
        "--results-json",
        type=Path,
        default=Path("paper_visualizations/outputs/freq_magnitude_v2/full_model_correlation_results.json"),
        help="Path to correlation results JSON for subplot (e)",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument("--max-distance", type=int, default=5000, help="Maximum token distance for reconstruction")
    parser.add_argument("--scatter-layer", type=int, default=32, help="Layer for scatter plot")
    parser.add_argument("--scatter-head", type=int, default=27, help="Head for scatter plot")
    parser.add_argument("--output-path", type=Path, default=None, help="Output path for figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_intro_fig")

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/intro_figure")
        output_path = output_dir / "fig_intro_combined.png"
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
    inv_freq = rotary.inv_freq.to(torch.float64)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # ========== Subplot (a): Q/K Scatter across 3 traces ==========
    print(f"\n=== Subplot (a): Q/K Scatter (L{args.scatter_layer}H{args.scatter_head}, {len(args.trace_dirs)} traces) ===")
    all_q_real, all_q_imag = [], []
    all_k_real, all_k_imag = [], []
    all_R_q, all_R_k = [], []

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

        q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
        k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
        q_complex = to_complex_pairs(q_orig)
        k_complex = to_complex_pairs(k_orig)

        dom_freq = compute_dominant_frequency(q_complex, k_complex, use_mean_norm=True)
        q_real, q_imag, k_real, k_imag, R_q, R_k = get_scatter_data(
            q_block, k_block, cos_table, sin_table, attention_scale, dom_freq, max_points=3000
        )
        print(f"    Dom freq: {dom_freq}, R_Q={R_q:.4f}, R_K={R_k:.4f}")

        all_q_real.extend(q_real)
        all_q_imag.extend(q_imag)
        all_k_real.extend(k_real)
        all_k_imag.extend(k_imag)
        all_R_q.append(R_q)
        all_R_k.append(R_k)

    mean_R_q = np.mean(all_R_q)
    mean_R_k = np.mean(all_R_k)
    print(f"  Combined: {len(all_q_real)} points, mean R_Q={mean_R_q:.4f}, mean R_K={mean_R_k:.4f}")

    # ========== Subplot (b): R Histogram ==========
    print("\n=== Subplot (b): R Histogram (all heads) ===")
    # Use the first trace for R computation
    primary_trace = args.trace_dirs[0]
    qk_path = primary_trace / "qk.pt"
    meta_path = primary_trace / "metadata.json"
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

    num_layers = q_tensor.shape[0]
    num_heads = q_tensor.shape[1]
    print(f"  Model: {num_layers} layers, {num_heads} heads")

    r_q_values = []
    r_k_values = []

    for layer in range(num_layers):
        for head in range(num_heads):
            q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
            k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

            q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
            k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

            q_complex = to_complex_pairs(q_orig)
            k_complex = to_complex_pairs(k_orig)

            dom_freq = compute_dominant_frequency(q_complex, k_complex, use_mean_norm=True)
            r_q_values.append(compute_mean_resultant_length(q_complex[:, dom_freq]))
            r_k_values.append(compute_mean_resultant_length(k_complex[:, dom_freq]))

        if (layer + 1) % 10 == 0:
            print(f"    Processed {layer + 1}/{num_layers} layers")

    print(f"  R_Q: [{min(r_q_values):.4f}, {max(r_q_values):.4f}], mean={np.mean(r_q_values):.4f}")
    print(f"  R_K: [{min(r_k_values):.4f}, {max(r_k_values):.4f}], mean={np.mean(r_k_values):.4f}")

    # ========== Subplot (c) & (d): Reconstruction curves ==========
    print("\n=== Subplot (c): Reconstruction L0H0 ===")
    q_block_c = q_tensor[0, 0, :token_count].to(device=device, dtype=torch.float32)
    k_block_c = k_tensor[0, 0, :token_count].to(device=device, dtype=torch.float32)
    recon_c = compute_reconstruction_data(
        q_block_c, k_block_c, cos_table, sin_table, attention_scale, inv_freq, args.max_distance, device
    )
    print(f"  Pearson r = {recon_c['per_query_pearson']:.4f}")

    print("\n=== Subplot (d): Reconstruction L1H18 ===")
    q_block_d = q_tensor[1, 18, :token_count].to(device=device, dtype=torch.float32)
    k_block_d = k_tensor[1, 18, :token_count].to(device=device, dtype=torch.float32)
    recon_d = compute_reconstruction_data(
        q_block_d, k_block_d, cos_table, sin_table, attention_scale, inv_freq, args.max_distance, device
    )
    print(f"  Pearson r = {recon_d['per_query_pearson']:.4f}")

    # ========== Subplot (e): Pearson histogram ==========
    print("\n=== Subplot (e): Pearson Histogram ===")
    if not args.results_json.exists():
        raise FileNotFoundError(f"Results JSON not found: {args.results_json}")

    with open(args.results_json) as f:
        results = json.load(f)

    all_pearson = np.array([r['ind_pearson'] for r in results])
    print(f"  {len(all_pearson)} heads, mean={all_pearson.mean():.4f}, std={all_pearson.std():.4f}")

    # ========== Create Figure ==========
    print("\n=== Creating figure ===")
    fig = plt.figure(figsize=(20, 4), dpi=args.dpi)
    gs = GridSpec(1, 5, figure=fig, wspace=0.28)

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
    })

    # ========== (a) Q/K Scatter ==========
    ax_a = fig.add_subplot(gs[0, 0])
    style_ax(ax_a)

    ax_a.scatter(all_q_real, all_q_imag, s=6, alpha=0.25, color=color_q, edgecolors="none", label="Q")
    ax_a.scatter(all_k_real, all_k_imag, s=6, alpha=0.25, color=color_k, edgecolors="none", label="K")
    ax_a.axhline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax_a.axvline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax_a.set_xlim(-1.15, 1.15)
    ax_a.set_ylim(-1.15, 1.15)
    ax_a.set_aspect("equal", adjustable="box")
    ax_a.set_xticks([-1, 0, 1])
    ax_a.set_yticks([-1, 0, 1])
    ax_a.text(0.95, 0.95, f"$R_Q$={mean_R_q:.2f}\n$R_K$={mean_R_k:.2f}",
              transform=ax_a.transAxes, fontsize=FONT_SIZE, ha='right', va='top')
    leg = ax_a.legend(loc="lower left", fontsize=FONT_SIZE, frameon=False, markerscale=2.5, handletextpad=0.2)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    ax_a.text(-0.15, 1.02, '(a)', transform=ax_a.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom')

    # ========== (b) R Histogram ==========
    ax_b = fig.add_subplot(gs[0, 1])
    style_ax(ax_b)

    bin_edges = np.linspace(0.5, 1.0, 21)
    ax_b.hist([r_q_values, r_k_values], bins=bin_edges,
              color=[color_q, color_k], label=['Q', 'K'],
              alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_b.set_xlabel("R", fontsize=FONT_SIZE)
    ax_b.set_ylabel("Count", fontsize=FONT_SIZE)
    ax_b.legend(frameon=False, fontsize=FONT_SIZE, loc='upper left')
    ax_b.text(-0.12, 1.02, '(b)', transform=ax_b.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom')

    # ========== (c) Reconstruction L0H0 ==========
    ax_c = fig.add_subplot(gs[0, 2])
    style_ax(ax_c)

    ax_c.fill_between(recon_c['sample_distances'],
                      recon_c['gt_means'] - recon_c['gt_stds'],
                      recon_c['gt_means'] + recon_c['gt_stds'],
                      color=color_k, alpha=0.25, linewidth=0, label='GT Band')
    ax_c.plot(recon_c['distances'], recon_c['gt_scores'],
              color=color_k, linestyle='--', linewidth=2, alpha=0.9, label='GT Mean')
    ax_c.plot(recon_c['distances'], recon_c['reconstructed'],
              color=color_q, linestyle=':', linewidth=2, alpha=0.9, label='Recon')
    ax_c.set_xscale('log')
    ax_c.set_xlabel(r'$\Delta$', fontsize=FONT_SIZE)
    ax_c.set_ylabel(r'$\langle q, k \rangle$', fontsize=FONT_SIZE)
    ax_c.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper right')
    ax_c.text(0.03, 0.03, f"$\\bar{{r}}$={recon_c['per_query_pearson']:.2f}",
              transform=ax_c.transAxes, fontsize=FONT_SIZE,
              va='bottom', ha='left',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    ax_c.text(-0.12, 1.02, '(c)', transform=ax_c.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom')

    # ========== (d) Reconstruction L1H18 ==========
    ax_d = fig.add_subplot(gs[0, 3])
    style_ax(ax_d)

    ax_d.fill_between(recon_d['sample_distances'],
                      recon_d['gt_means'] - recon_d['gt_stds'],
                      recon_d['gt_means'] + recon_d['gt_stds'],
                      color=color_k, alpha=0.25, linewidth=0, label='GT Band')
    ax_d.plot(recon_d['distances'], recon_d['gt_scores'],
              color=color_k, linestyle='--', linewidth=2, alpha=0.9, label='GT Mean')
    ax_d.plot(recon_d['distances'], recon_d['reconstructed'],
              color=color_q, linestyle=':', linewidth=2, alpha=0.9, label='Recon')
    ax_d.set_xscale('log')
    ax_d.set_xlabel(r'$\Delta$', fontsize=FONT_SIZE)
    ax_d.set_ylabel(r'$\langle q, k \rangle$', fontsize=FONT_SIZE)
    ax_d.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='upper right')
    ax_d.text(0.03, 0.03, f"$\\bar{{r}}$={recon_d['per_query_pearson']:.2f}",
              transform=ax_d.transAxes, fontsize=FONT_SIZE,
              va='bottom', ha='left',
              bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8, edgecolor='none'))
    ax_d.text(-0.12, 1.02, '(d)', transform=ax_d.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom')

    # ========== (e) Pearson Histogram ==========
    ax_e = fig.add_subplot(gs[0, 4])
    style_ax(ax_e)

    bin_edges_e = np.arange(-0.2, 1.025, 0.05)
    ax_e.hist(all_pearson, bins=bin_edges_e, color=color_q, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax_e.axvline(all_pearson.mean(), color=color_mean, linestyle='--', linewidth=2.5,
                 label=f'Mean={all_pearson.mean():.2f}')
    ax_e.set_xlabel('Pearson $\\bar{r}$', fontsize=FONT_SIZE)
    ax_e.set_ylabel('Count', fontsize=FONT_SIZE)
    ax_e.legend(frameon=False, fontsize=FONT_SIZE, loc='upper left')
    ax_e.set_xticks(np.arange(0, 1.1, 0.2))
    ax_e.set_xlim(-0.25, 1.05)
    ax_e.text(-0.12, 1.02, '(e)', transform=ax_e.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom')

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
