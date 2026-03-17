"""Reconstruct relative position curve with GT error band.

Formula: ⟨q, k⟩_Δ = Σ_f |q_f| |k_f| cos(ω_f Δ + φ_f)

Shows GT mean ± std as shaded error band.
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
import matplotlib.font_manager as fm
import seaborn as sns
import torch
import numpy as np
from scipy import stats
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct position curve with GT error band"
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--head", type=int, default=0, help="Head index")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=10000,
        help="Maximum token distance for reconstruction",
    )
    parser.add_argument("--dpi", type=int, default=300, help="Figure DPI")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for the figure",
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=200,
        help="Number of distance samples for error band (log-spaced)",
    )
    return parser.parse_args()


def select_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    x1 = x[..., :d]
    x2 = x[..., d:]
    return torch.cat((-x2, x1), dim=-1)


def invert_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    z = rotated / scale
    return z * cos - rotate_half(z) * sin


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    seq_len, head_dim = tensor.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    num_freq = head_dim // 2
    real = tensor[:, :num_freq]
    imag = tensor[:, num_freq:]
    return torch.complex(real.float(), imag.float())


def compute_gt_curve(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    distances: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute GT mean using FFT (fast, for all distances)."""
    token_count = q_block.shape[0]
    max_dist = int(distances.max().item())

    fft_len = 1 << (2 * token_count - 1).bit_length()
    q_fft = torch.fft.rfft(q_block.float(), n=fft_len, dim=0)
    k_fft = torch.fft.rfft(k_block.float(), n=fft_len, dim=0)

    prod = torch.conj(k_fft) * q_fft
    corr = torch.fft.irfft(prod.sum(dim=1), n=fft_len, dim=0)

    all_counts = (token_count - torch.arange(0, max_dist + 1, device=device, dtype=torch.float32)).clamp_min(1.0)

    dist_indices = distances.long()
    gt_scores = corr[dist_indices] / all_counts[dist_indices]

    return gt_scores


def compute_gt_with_std(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    distances: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute GT mean and std directly for sampled distances.

    For each distance Δ, compute all q_i · k_{i-Δ} scores and get mean/std.
    This is O(num_distances * avg_pairs) but we use sampled distances.
    """
    token_count = q_block.shape[0]
    q_float = q_block.float()
    k_float = k_block.float()

    means = []
    stds = []

    for delta in distances.long().tolist():
        if delta >= token_count:
            means.append(0.0)
            stds.append(0.0)
            continue

        # q_i · k_{i-Δ} for i in [Δ, token_count)
        # q[delta:] · k[:token_count-delta]
        q_slice = q_float[delta:]  # [T-delta, D]
        k_slice = k_float[:token_count - delta]  # [T-delta, D]

        # Dot product per position
        scores = (q_slice * k_slice).sum(dim=1)  # [T-delta]

        means.append(scores.mean().item())
        stds.append(scores.std().item())

    return torch.tensor(means, device=device), torch.tensor(stds, device=device)


def compute_individual_correlation(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    reconstructed: torch.Tensor,
    distances: torch.Tensor,
    device: torch.device,
    max_pairs: int = 500000,
) -> dict[str, float]:
    """Compute individual-level correlation between actual scores and reconstruction.

    For each (i, Δ) pair:
      - actual: s_{i,Δ} = q_i · k_{i-Δ}
      - predicted: r_Δ (reconstruction value at distance Δ)

    Returns Pearson correlation, Spearman correlation, and R².

    Args:
        max_pairs: Maximum number of pairs to use (subsample if exceeded for memory)
    """
    token_count = q_block.shape[0]
    q_float = q_block.float()
    k_float = k_block.float()

    all_actual = []
    all_predicted = []

    # Build mapping from distance to reconstruction value
    dist_to_recon = {int(d): reconstructed[i].item()
                     for i, d in enumerate(distances.long().tolist())}

    # Collect all (actual, predicted) pairs
    for delta in distances.long().tolist():
        if delta >= token_count:
            continue

        q_slice = q_float[delta:]
        k_slice = k_float[:token_count - delta]
        scores = (q_slice * k_slice).sum(dim=1)  # [T-delta]

        recon_val = dist_to_recon[delta]

        all_actual.extend(scores.cpu().tolist())
        all_predicted.extend([recon_val] * len(scores))

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)

    # Subsample if too many pairs
    if len(all_actual) > max_pairs:
        indices = np.random.choice(len(all_actual), max_pairs, replace=False)
        all_actual = all_actual[indices]
        all_predicted = all_predicted[indices]

    # Pearson correlation
    pearson_r, _ = stats.pearsonr(all_actual, all_predicted)

    # Spearman correlation
    spearman_r, _ = stats.spearmanr(all_actual, all_predicted)

    # R² (coefficient of determination)
    ss_res = np.sum((all_actual - all_predicted) ** 2)
    ss_tot = np.sum((all_actual - np.mean(all_actual)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'pearson': pearson_r,
        'spearman': spearman_r,
        'r_squared': r_squared,
        'n_pairs': len(all_actual),
    }


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_reconstruct")

    # Load data
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {args.trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])

    device = torch.device(args.device)
    dtype = select_dtype(args.dtype)

    # Load model config for RoPE
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

    layer, head = args.layer, args.head
    head_dim = q_tensor.shape[-1]
    num_freq = head_dim // 2

    q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=dtype)
    k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=dtype)

    # Build RoPE tables
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=dtype)
    sin_table = sin_table[0].to(dtype=dtype)

    # ========== Core computation ==========

    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    q_complex = to_complex_pairs(q_orig)
    k_complex = to_complex_pairs(k_orig)

    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    q_f_mag = torch.abs(q_mean)
    k_f_mag = torch.abs(k_mean)
    q_f_phase = torch.angle(q_mean)
    k_f_phase = torch.angle(k_mean)

    phi_f = q_f_phase - k_f_phase
    amplitude = q_f_mag * k_f_mag
    omega = inv_freq[:num_freq].to(device=device, dtype=torch.float32)

    max_dist = min(args.max_distance, token_count - 1)

    # Full distances for main curves (using FFT)
    distances = torch.arange(1, max_dist + 1, device=device, dtype=torch.float32)

    # Log-spaced samples for error band computation (direct method)
    sample_distances = torch.unique(torch.logspace(0, math.log10(max_dist), args.num_samples, device=device).long())
    sample_distances = sample_distances[sample_distances >= 1]
    sample_distances = sample_distances[sample_distances <= max_dist]

    # Reconstructed curve (full)
    phase_matrix = distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)
    cos_matrix = torch.cos(phase_matrix)
    reconstructed = (cos_matrix * amplitude.unsqueeze(0)).sum(dim=1)

    # GT curve (full, using FFT)
    gt_scores = compute_gt_curve(q_block, k_block, distances, device)

    # GT with std (sampled, using direct computation)
    gt_means_sampled, gt_stds_sampled = compute_gt_with_std(
        q_block, k_block, sample_distances.float(), device
    )

    # Compute individual-level correlation
    individual_corr = compute_individual_correlation(
        q_block, k_block, reconstructed, distances, device
    )
    print(f"\nIndividual-level metrics (n={individual_corr['n_pairs']} pairs):")
    print(f"  Pearson r:  {individual_corr['pearson']:.4f}")
    print(f"  Spearman ρ: {individual_corr['spearman']:.4f}")
    print(f"  R²:         {individual_corr['r_squared']:.4f}")

    # ========== Plotting with paper style ==========

    dist_np = distances.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    gt_np = gt_scores.cpu().numpy()

    sample_dist_np = sample_distances.cpu().numpy()
    gt_means_np = gt_means_sampled.cpu().numpy()
    gt_stds_np = gt_stds_sampled.cpu().numpy()

    # Compute mean-level Spearman correlation (for reference)
    mean_spearman, _ = stats.spearmanr(gt_np, recon_np)

    # Custom color palette
    color_gt = (187 / 250, 130 / 250, 90 / 250)      # warm brown/orange
    color_recon = (85 / 250, 104 / 250, 154 / 250)   # blue

    # Background color
    face_color = (231 / 250, 231 / 250, 240 / 250)   # light gray-purple

    # Set font sizes for paper
    plt.rcParams.update({
        'font.size': 18,
        'axes.labelsize': 20,
        'axes.titlesize': 20,
        'xtick.labelsize': 16,
        'ytick.labelsize': 16,
        'legend.fontsize': 16,
    })

    fig, ax = plt.subplots(figsize=(7, 5.5))

    # Set background color
    ax.set_facecolor(face_color)

    # Hide all spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(False)
    ax.spines["bottom"].set_visible(False)

    # Hide tick marks but keep labels
    ax.tick_params(top=False, bottom=False, left=False, right=False)

    # White grid
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    # Plot GT error band (shaded region: mean ± std)
    ax.fill_between(sample_dist_np,
                    gt_means_np - gt_stds_np,
                    gt_means_np + gt_stds_np,
                    color=color_gt, alpha=0.25, linewidth=0,
                    label='GT ± std')

    # Plot curves
    ax.plot(dist_np, gt_np,
            color=color_gt, linestyle='--', linewidth=2.5, alpha=0.9,
            label='Ground Truth')
    ax.plot(dist_np, recon_np,
            color=color_recon, linestyle=':', linewidth=2.5, alpha=0.9,
            label='Reconstructed')

    # Log scale x-axis
    ax.set_xscale('log')

    # Labels
    ax.set_xlabel('Relative Position $\\Delta$')
    ax.set_ylabel('$\\langle q, k \\rangle_\\Delta$')

    # Add correlation annotations (bottom-left)
    # Show individual-level Pearson (more meaningful for actual attention prediction)
    annotation_text = (
        f"Individual Pearson $r$ = {individual_corr['pearson']:.4f}\n"
        f"Mean Spearman $\\rho$ = {mean_spearman:.4f}"
    )
    ax.text(0.03, 0.03, annotation_text,
            transform=ax.transAxes, fontsize=14,
            verticalalignment='bottom', horizontalalignment='left',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))

    # Legend
    ax.legend(frameon=False, loc='upper right')

    plt.tight_layout()

    # Save
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = Path(f"paper_visualizations/outputs/freq_magnitude_v2/reconstruct_l{layer}_h{head}_with_band.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main()
