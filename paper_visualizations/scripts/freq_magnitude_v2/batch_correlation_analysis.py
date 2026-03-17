"""Batch analysis of individual-level correlation across 100 random heads."""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
        description="Batch correlation analysis across random heads"
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--num-heads", type=int, default=100, help="Number of heads to sample")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--max-distance",
        type=int,
        default=5000,
        help="Maximum token distance",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for results",
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


def compute_metrics_for_head(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    inv_freq: torch.Tensor,
    max_dist: int,
    device: torch.device,
    max_pairs: int = 200000,
) -> dict:
    """Compute all metrics for a single head."""
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

    # Distances (log-spaced for consistent weighting with visualization)
    actual_max_dist = min(max_dist, token_count - 1)
    # Use ~500 log-spaced points
    distances = torch.unique(torch.logspace(0, math.log10(actual_max_dist), 500, device=device).long())
    distances = distances[distances >= 1].float()

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

    # Mean-level Spearman
    gt_np = gt_scores.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    mean_spearman, _ = stats.spearmanr(gt_np, recon_np)

    # Individual-level metrics
    q_float = q_block.float()
    k_float = k_block.float()

    all_actual = []
    all_predicted = []

    dist_to_recon = {int(d): reconstructed[i].item() for i, d in enumerate(distances.long().tolist())}

    for delta in distances.long().tolist():
        if delta >= token_count:
            continue
        q_slice = q_float[delta:]
        k_slice = k_float[:token_count - delta]
        scores = (q_slice * k_slice).sum(dim=1)
        recon_val = dist_to_recon[delta]
        all_actual.extend(scores.cpu().tolist())
        all_predicted.extend([recon_val] * len(scores))

    all_actual = np.array(all_actual)
    all_predicted = np.array(all_predicted)

    # Subsample if needed
    if len(all_actual) > max_pairs:
        indices = np.random.choice(len(all_actual), max_pairs, replace=False)
        all_actual = all_actual[indices]
        all_predicted = all_predicted[indices]

    # Individual correlations
    ind_pearson, _ = stats.pearsonr(all_actual, all_predicted)
    ind_spearman, _ = stats.spearmanr(all_actual, all_predicted)

    # R²
    ss_res = np.sum((all_actual - all_predicted) ** 2)
    ss_tot = np.sum((all_actual - np.mean(all_actual)) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    return {
        'mean_spearman': mean_spearman,
        'ind_pearson': ind_pearson,
        'ind_spearman': ind_spearman,
        'r_squared': r_squared,
        'n_pairs': len(all_actual),
    }


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_batch_corr")

    random.seed(args.seed)
    np.random.seed(args.seed)

    # Load data
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {args.trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
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
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]  # [L, H, T, D]
    k_tensor = data["k"]

    num_layers, num_heads, _, head_dim = q_tensor.shape
    print(f"Model: {num_layers} layers, {num_heads} heads/layer, head_dim={head_dim}")
    print(f"Token count: {token_count}")

    # Build RoPE tables
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=dtype)
    sin_table = sin_table[0].to(dtype=dtype)

    # Sample 100 heads from different layers
    # Strategy: sample roughly uniformly across layers
    all_layer_head_pairs = [(l, h) for l in range(num_layers) for h in range(num_heads)]
    random.shuffle(all_layer_head_pairs)

    # Try to get diverse layers
    selected = []
    layers_used = set()
    for l, h in all_layer_head_pairs:
        if len(selected) >= args.num_heads:
            break
        # Prefer heads from layers we haven't used much
        layer_count = sum(1 for ll, _ in selected if ll == l)
        max_per_layer = (args.num_heads // num_layers) + 2
        if layer_count < max_per_layer:
            selected.append((l, h))
            layers_used.add(l)

    # Fill remaining if needed
    for l, h in all_layer_head_pairs:
        if len(selected) >= args.num_heads:
            break
        if (l, h) not in selected:
            selected.append((l, h))

    selected = selected[:args.num_heads]
    print(f"\nSelected {len(selected)} heads from {len(set(l for l, _ in selected))} different layers")

    # Compute metrics for each head
    results = []
    for i, (layer, head) in enumerate(selected):
        q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=dtype)
        k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=dtype)

        metrics = compute_metrics_for_head(
            q_block, k_block, cos_table, sin_table,
            attention_scale, inv_freq, args.max_distance, device
        )
        metrics['layer'] = layer
        metrics['head'] = head
        results.append(metrics)

        if (i + 1) % 10 == 0:
            print(f"Processed {i + 1}/{len(selected)} heads...")

    # Aggregate statistics
    mean_spearman_arr = np.array([r['mean_spearman'] for r in results])
    ind_pearson_arr = np.array([r['ind_pearson'] for r in results])
    ind_spearman_arr = np.array([r['ind_spearman'] for r in results])
    r_squared_arr = np.array([r['r_squared'] for r in results])

    print("\n" + "=" * 60)
    print("AGGREGATE STATISTICS (100 heads)")
    print("=" * 60)
    print(f"\nMean-level Spearman (GT mean vs Reconstruction):")
    print(f"  Mean:   {mean_spearman_arr.mean():.4f}")
    print(f"  Std:    {mean_spearman_arr.std():.4f}")
    print(f"  Min:    {mean_spearman_arr.min():.4f}")
    print(f"  Max:    {mean_spearman_arr.max():.4f}")

    print(f"\nIndividual-level Pearson (actual scores vs reconstruction):")
    print(f"  Mean:   {ind_pearson_arr.mean():.4f}")
    print(f"  Std:    {ind_pearson_arr.std():.4f}")
    print(f"  Min:    {ind_pearson_arr.min():.4f}")
    print(f"  Max:    {ind_pearson_arr.max():.4f}")

    print(f"\nIndividual-level Spearman:")
    print(f"  Mean:   {ind_spearman_arr.mean():.4f}")
    print(f"  Std:    {ind_spearman_arr.std():.4f}")
    print(f"  Min:    {ind_spearman_arr.min():.4f}")
    print(f"  Max:    {ind_spearman_arr.max():.4f}")

    print(f"\nR² (variance explained):")
    print(f"  Mean:   {r_squared_arr.mean():.4f}")
    print(f"  Std:    {r_squared_arr.std():.4f}")
    print(f"  Min:    {r_squared_arr.min():.4f}")
    print(f"  Max:    {r_squared_arr.max():.4f}")

    # Save results
    output_dir = Path("paper_visualizations/outputs/freq_magnitude_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save raw results as JSON
    results_path = output_dir / "batch_correlation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {results_path}")

    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Colors
    color = (85 / 250, 104 / 250, 154 / 250)
    face_color = (231 / 250, 231 / 250, 240 / 250)

    for ax in axes.flat:
        ax.set_facecolor(face_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False)
        ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1)

    # Histogram of mean-level Spearman
    ax = axes[0, 0]
    ax.hist(mean_spearman_arr, bins=20, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(mean_spearman_arr.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={mean_spearman_arr.mean():.3f}')
    ax.set_xlabel('Mean-level Spearman')
    ax.set_ylabel('Count')
    ax.set_title('Mean-level Spearman Distribution')
    ax.legend()

    # Histogram of individual-level Pearson
    ax = axes[0, 1]
    ax.hist(ind_pearson_arr, bins=20, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(ind_pearson_arr.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={ind_pearson_arr.mean():.3f}')
    ax.set_xlabel('Individual-level Pearson')
    ax.set_ylabel('Count')
    ax.set_title('Individual-level Pearson Distribution')
    ax.legend()

    # Histogram of R²
    ax = axes[1, 0]
    ax.hist(r_squared_arr, bins=20, color=color, alpha=0.8, edgecolor='white')
    ax.axvline(r_squared_arr.mean(), color='red', linestyle='--', linewidth=2,
               label=f'Mean={r_squared_arr.mean():.3f}')
    ax.set_xlabel('R²')
    ax.set_ylabel('Count')
    ax.set_title('R² Distribution')
    ax.legend()

    # Scatter: mean-level vs individual-level
    ax = axes[1, 1]
    ax.scatter(mean_spearman_arr, ind_pearson_arr, color=color, alpha=0.6, s=30)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='y=x')
    ax.set_xlabel('Mean-level Spearman')
    ax.set_ylabel('Individual-level Pearson')
    ax.set_title('Mean-level vs Individual-level Correlation')
    ax.legend()

    plt.tight_layout()

    fig_path = output_dir / "batch_correlation_analysis.png"
    fig.savefig(fig_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f"Visualization saved to {fig_path}")


if __name__ == "__main__":
    main()
