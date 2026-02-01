"""Generate Introduction figure with 4 subplots (v2).

(a) Pre-RoPE Q/K distribution (L32H27, across 3 traces)
(b) Post-RoPE Q/K distribution (same head, after RoPE rotation)
(c) R concentration histogram (Q/K merged, bar chart, percentage Y-axis)
(d) Reconstruction curve Case 1 (L0H0)
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

# Force Times New Roman font for all text
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.serif'] = ['Times New Roman']
plt.rcParams['mathtext.fontset'] = 'stix'  # STIX fonts match Times New Roman for math
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
LABEL_FONT = 'Times New Roman'
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


def get_scatter_data_pre_rope(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    freq_idx: int,
    max_points: int = 5000,
    filter_radius: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Get Q/K scatter data for a specific frequency (Pre-RoPE)."""
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
        R_q,
        R_k,
    )


def get_scatter_data_post_rope(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    freq_idx: int,
    max_points: int = 5000,
    filter_radius: float = 0.3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, float]:
    """Get Q/K scatter data for a specific frequency (Post-RoPE, no invert)."""
    # Directly use the rotated Q/K (no invert_rope)
    q_complex = to_complex_pairs(q_block)
    k_complex = to_complex_pairs(k_block)

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
    parser = argparse.ArgumentParser(description="Generate Introduction figure with 4 subplots (v2)")
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
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument("--max-distance", type=int, default=5000, help="Maximum token distance for reconstruction")
    parser.add_argument("--scatter-layer", type=int, default=3, help="Layer for scatter plot")
    parser.add_argument("--scatter-head", type=int, default=0, help="Head for scatter plot")
    parser.add_argument("--output-path", type=Path, default=None, help="Output path for figure")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_intro_fig_v2")

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/intro_figure")
        output_path = output_dir / "fig_intro_combined_v2.png"
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

    # ========== Subplot (a): Pre-RoPE Q/K Scatter across 3 traces ==========
    print(f"\n=== Subplot (a): Pre-RoPE Q/K Scatter (L{args.scatter_layer}H{args.scatter_head}, {len(args.trace_dirs)} traces) ===")
    all_q_real_pre, all_q_imag_pre = [], []
    all_k_real_pre, all_k_imag_pre = [], []
    all_R_q_pre, all_R_k_pre = [], []

    # Also collect Post-RoPE data for subplot (b)
    all_q_real_post, all_q_imag_post = [], []
    all_k_real_post, all_k_imag_post = [], []
    all_R_q_post, all_R_k_post = [], []

    # Use first trace to determine dominant frequency (fix for GQA consistency)
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

        # Pre-RoPE: compute dominant frequency only from first trace
        q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
        k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
        q_complex_pre = to_complex_pairs(q_orig)
        k_complex_pre = to_complex_pairs(k_orig)

        if fixed_dom_freq is None:
            fixed_dom_freq = compute_dominant_frequency(q_complex_pre, k_complex_pre, use_mean_norm=True)
            print(f"  Using fixed dominant frequency: {fixed_dom_freq}")
        dom_freq = fixed_dom_freq

        # Get Pre-RoPE scatter data
        q_real, q_imag, k_real, k_imag, R_q, R_k = get_scatter_data_pre_rope(
            q_block, k_block, cos_table, sin_table, attention_scale, dom_freq, max_points=3000
        )
        print(f"    Pre-RoPE: R_Q={R_q:.4f}, R_K={R_k:.4f}")

        all_q_real_pre.extend(q_real)
        all_q_imag_pre.extend(q_imag)
        all_k_real_pre.extend(k_real)
        all_k_imag_pre.extend(k_imag)
        all_R_q_pre.append(R_q)
        all_R_k_pre.append(R_k)

        # Get Post-RoPE scatter data (same freq_idx for comparison)
        q_real_post, q_imag_post, k_real_post, k_imag_post, R_q_post, R_k_post = get_scatter_data_post_rope(
            q_block, k_block, dom_freq, max_points=3000
        )
        print(f"    Post-RoPE: R_Q={R_q_post:.4f}, R_K={R_k_post:.4f}")

        all_q_real_post.extend(q_real_post)
        all_q_imag_post.extend(q_imag_post)
        all_k_real_post.extend(k_real_post)
        all_k_imag_post.extend(k_imag_post)
        all_R_q_post.append(R_q_post)
        all_R_k_post.append(R_k_post)

    mean_R_q_pre = np.mean(all_R_q_pre)
    mean_R_k_pre = np.mean(all_R_k_pre)
    mean_R_q_post = np.mean(all_R_q_post)
    mean_R_k_post = np.mean(all_R_k_post)
    print(f"  Pre-RoPE Combined: {len(all_q_real_pre)} points, mean R_Q={mean_R_q_pre:.4f}, mean R_K={mean_R_k_pre:.4f}")
    print(f"  Post-RoPE Combined: {len(all_q_real_post)} points, mean R_Q={mean_R_q_post:.4f}, mean R_K={mean_R_k_post:.4f}")

    # ========== Subplot (c): R Histogram (with percentage Y-axis) ==========
    print("\n=== Subplot (c): R Histogram (all heads, percentage Y-axis) ===")
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
    total_heads = num_layers * num_heads
    print(f"  Model: {num_layers} layers, {num_heads} heads, total {total_heads} heads")

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

    # ========== Subplot (d): Reconstruction curve L0H0 ==========
    print("\n=== Subplot (d): Reconstruction L0H0 ===")
    q_block_d = q_tensor[0, 0, :token_count].to(device=device, dtype=torch.float32)
    k_block_d = k_tensor[0, 0, :token_count].to(device=device, dtype=torch.float32)
    recon_d = compute_reconstruction_data(
        q_block_d, k_block_d, cos_table, sin_table, attention_scale, inv_freq, args.max_distance, device
    )
    print(f"  Pearson r = {recon_d['per_query_pearson']:.4f}")

    # ========== Create Figure ==========
    print("\n=== Creating figure ===")
    # 4 subplots: (a) Pre-RoPE, (b) Post-RoPE, (c) R Histogram, (d) Reconstruction
    fig, axes = plt.subplots(1, 4, figsize=(16, 4), dpi=args.dpi)
    plt.subplots_adjust(wspace=0.18, left=0.04, right=0.98, top=0.88, bottom=0.14)

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
    })

    # ========== (a) Pre-RoPE Q/K Scatter ==========
    ax_a = axes[0]
    style_ax(ax_a)

    ax_a.scatter(all_q_real_pre, all_q_imag_pre, s=6, alpha=0.25, color=color_q, edgecolors="none", label="Q")
    ax_a.scatter(all_k_real_pre, all_k_imag_pre, s=6, alpha=0.25, color=color_k, edgecolors="none", label="K")
    ax_a.axhline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax_a.axvline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax_a.set_xlim(-1.15, 1.15)
    ax_a.set_ylim(-1.15, 1.15)
    ax_a.set_box_aspect(1)
    ax_a.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_a.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_a.set_xlabel("Re", fontsize=FONT_SIZE)
    ax_a.set_ylabel("Im", fontsize=FONT_SIZE)
    ax_a.text(0.95, 0.95, f"$R_Q$={mean_R_q_pre:.2f}\n$R_K$={mean_R_k_pre:.2f}",
              transform=ax_a.transAxes, fontsize=FONT_SIZE, ha='right', va='top')
    leg = ax_a.legend(loc="lower left", fontsize=FONT_SIZE, frameon=False, markerscale=2.5, handletextpad=0.2)
    for lh in leg.legend_handles:
        lh.set_alpha(1.0)
    ax_a.text(-0.08, 1.05, '(A)', transform=ax_a.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom', fontname=LABEL_FONT)
    ax_a.set_title('Pre-RoPE Q/K', fontsize=FONT_SIZE, pad=8)

    # ========== (b) Post-RoPE Q/K Scatter ==========
    ax_b = axes[1]
    style_ax(ax_b)

    ax_b.scatter(all_q_real_post, all_q_imag_post, s=6, alpha=0.25, color=color_q, edgecolors="none", label="Q")
    ax_b.scatter(all_k_real_post, all_k_imag_post, s=6, alpha=0.25, color=color_k, edgecolors="none", label="K")
    ax_b.axhline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax_b.axvline(0.0, color="gray", linewidth=0.8, alpha=0.4)
    ax_b.set_xlim(-1.15, 1.15)
    ax_b.set_ylim(-1.15, 1.15)
    ax_b.set_box_aspect(1)
    ax_b.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax_b.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_b.set_xlabel("Re", fontsize=FONT_SIZE)
    ax_b.set_ylabel("Im", fontsize=FONT_SIZE)
    leg_b = ax_b.legend(loc="lower left", fontsize=FONT_SIZE, frameon=False, markerscale=2.5, handletextpad=0.2)
    for lh in leg_b.legend_handles:
        lh.set_alpha(1.0)
    ax_b.text(-0.08, 1.05, '(B)', transform=ax_b.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom', fontname=LABEL_FONT)
    ax_b.set_title('Post-RoPE Q/K', fontsize=FONT_SIZE, pad=8)

    # ========== (c) R Histogram (percentage Y-axis) ==========
    ax_c = axes[2]
    style_ax(ax_c)

    # Use 0.05 bin width like reference (0.5-1.0 range = 10 bins)
    bin_edges = np.arange(0.5, 1.025, 0.05)

    # Compute histogram counts and convert to percentage
    counts_q, _ = np.histogram(r_q_values, bins=bin_edges)
    counts_k, _ = np.histogram(r_k_values, bins=bin_edges)
    pct_q = counts_q / total_heads * 100
    pct_k = counts_k / total_heads * 100

    # Plot as grouped bar chart
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    bar_width = 0.02
    ax_c.bar(bin_centers - bar_width/2, pct_q, width=bar_width, color=color_q, label='Q', alpha=0.85, edgecolor='white', linewidth=0.8)
    ax_c.bar(bin_centers + bar_width/2, pct_k, width=bar_width, color=color_k, label='K', alpha=0.85, edgecolor='white', linewidth=0.8)

    ax_c.set_xlabel("Concentration $R$", fontsize=FONT_SIZE)
    ax_c.set_ylabel("Percentage (%)", fontsize=FONT_SIZE, labelpad=2)
    ax_c.set_box_aspect(1)
    ax_c.set_xlim(0.45, 1.05)
    ax_c.set_xticks([0.5, 0.6, 0.7, 0.8, 0.9, 1.0])
    # Set y-axis ticks as percentages
    ax_c.set_yticks([0, 20, 40, 60, 80, 100])
    ax_c.legend(frameon=False, fontsize=FONT_SIZE, loc='upper left')
    ax_c.text(-0.08, 1.05, '(C)', transform=ax_c.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom', fontname=LABEL_FONT)
    ax_c.set_title('Concentration Distribution', fontsize=FONT_SIZE, pad=8)

    # ========== (d) Reconstruction L0H0 ==========
    ax_d = axes[3]
    style_ax(ax_d)

    ax_d.fill_between(recon_d['sample_distances'],
                      recon_d['gt_means'] - recon_d['gt_stds'],
                      recon_d['gt_means'] + recon_d['gt_stds'],
                      color=color_k, alpha=0.25, linewidth=0)
    ax_d.plot(recon_d['distances'], recon_d['gt_scores'],
              color=color_k, linestyle='--', linewidth=2.5, alpha=0.9, label='Ground Truth')
    ax_d.plot(recon_d['distances'], recon_d['reconstructed'],
              color=color_q, linestyle=':', linewidth=2.5, alpha=0.9, label='Trig. Recon.')
    ax_d.set_xscale('log')
    ax_d.set_xlim(1, 5000)
    ax_d.set_xticks([1, 10, 100, 1000, 5000])
    ax_d.set_xticklabels(['1', '10', '100', '1k', '5k'])
    ax_d.set_xlabel(r'Q-K Distance $\Delta$', fontsize=FONT_SIZE)
    ax_d.set_ylabel('Attention Logit', fontsize=FONT_SIZE, labelpad=2)
    ax_d.set_box_aspect(1)
    ax_d.set_yticks([0, 25, 50, 75, 100, 125])
    ax_d.legend(frameon=False, fontsize=FONT_SIZE - 2, loc='lower left')
    ax_d.text(-0.08, 1.05, '(D)', transform=ax_d.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom', fontname=LABEL_FONT)
    ax_d.set_title('Attention Reconstruction', fontsize=FONT_SIZE, pad=8)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
