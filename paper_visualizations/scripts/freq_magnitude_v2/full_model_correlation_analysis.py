"""Full model correlation analysis - all heads."""
from __future__ import annotations

import argparse
import json
import math
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
        description="Full model correlation analysis"
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
    parser.add_argument(
        "--max-distance",
        type=int,
        default=5000,
        help="Maximum token distance",
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


def compute_per_query_pearson(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
    inv_freq: torch.Tensor,
    max_dist: int,
    device: torch.device,
) -> float:
    """Compute per-query Pearson correlation with log-spaced distance sampling."""
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
    # Sample query positions to speed up computation
    q_float = q_block.float()
    k_float = k_block.float()
    min_history = 50  # Need enough history for meaningful log-spaced sampling
    num_log_samples = 50  # Number of log-spaced distance samples per query
    num_query_samples = 500  # Sample this many query positions instead of all

    # Sample query positions (log-spaced to cover both early and late positions)
    query_positions = torch.unique(torch.logspace(
        math.log10(min_history), math.log10(token_count - 1), num_query_samples, device=device
    ).long())
    query_positions = query_positions[(query_positions >= min_history) & (query_positions < token_count)]

    per_query_pearsons = []

    for query_pos in query_positions.tolist():
        # Log-spaced distance sampling for this query
        log_distances = torch.unique(torch.logspace(0, math.log10(query_pos), num_log_samples, device=device).long())
        log_distances = log_distances[(log_distances >= 1) & (log_distances <= query_pos)]

        if len(log_distances) < 3:
            continue

        # Vectorized computation of attention scores
        key_positions = query_pos - log_distances  # [num_distances]
        q_vec = q_float[query_pos]  # [head_dim]
        k_vecs = k_float[key_positions]  # [num_distances, head_dim]
        actual_scores = (q_vec.unsqueeze(0) * k_vecs).sum(dim=1)  # [num_distances]

        # Get predicted scores
        predicted_scores = torch.tensor([recon_dict[int(d)] for d in log_distances.tolist()], device=device)

        # Compute Pearson for this query
        r, _ = stats.pearsonr(actual_scores.cpu().numpy(), predicted_scores.cpu().numpy())
        if not np.isnan(r):
            per_query_pearsons.append(r)

    return np.mean(per_query_pearsons) if per_query_pearsons else 0.0


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_full_corr")

    np.random.seed(42)

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
    total_heads = num_layers * num_heads
    print(f"Model: {num_layers} layers, {num_heads} heads/layer = {total_heads} total heads")
    print(f"Token count: {token_count}")

    # Build RoPE tables
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=dtype)
    sin_table = sin_table[0].to(dtype=dtype)

    # Compute for all heads
    results = []  # List of (layer, head, pearson)

    for layer in range(num_layers):
        for head in range(num_heads):
            q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=dtype)
            k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=dtype)

            pearson = compute_per_query_pearson(
                q_block, k_block, cos_table, sin_table,
                attention_scale, inv_freq, args.max_distance, device
            )
            results.append({
                'layer': layer,
                'head': head,
                'ind_pearson': float(pearson),  # Keep key name for compatibility
            })

        print(f"Layer {layer + 1}/{num_layers} done")

    # Save raw results
    output_dir = Path("paper_visualizations/outputs/freq_magnitude_v2")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "full_model_correlation_results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nRaw results saved to {results_path}")

    # Aggregate
    all_pearson = np.array([r['ind_pearson'] for r in results])
    global_mean = all_pearson.mean()

    print("\n" + "=" * 60)
    print(f"FULL MODEL STATISTICS ({total_heads} heads)")
    print("=" * 60)
    print(f"Attn Reconstruction Pearson r (log-spaced):")
    print(f"  Mean:   {global_mean:.2f}")
    print(f"  Std:    {all_pearson.std():.2f}")
    print(f"  Min:    {all_pearson.min():.2f}")
    print(f"  Max:    {all_pearson.max():.2f}")

    # Per-layer statistics
    layer_above_mean_pct = []
    for layer in range(num_layers):
        layer_pearson = [r['ind_pearson'] for r in results if r['layer'] == layer]
        above_mean = sum(1 for p in layer_pearson if p > global_mean)
        pct = above_mean / len(layer_pearson) * 100
        layer_above_mean_pct.append(pct)
        print(f"  Layer {layer:2d}: {above_mean:2d}/{len(layer_pearson)} above mean ({pct:.1f}%)")

    # ========== Plotting ==========

    color = (85 / 250, 104 / 250, 154 / 250)
    color_bar = (187 / 250, 130 / 250, 90 / 250)
    face_color = (231 / 250, 231 / 250, 240 / 250)

    plt.rcParams.update({
        'font.size': 16,
        'axes.labelsize': 18,
        'xtick.labelsize': 14,
        'ytick.labelsize': 14,
    })

    # Figure 1: Histogram of all heads
    fig1, ax1 = plt.subplots(figsize=(7, 5))
    ax1.set_facecolor(face_color)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.tick_params(top=False, bottom=False, left=False, right=False)
    ax1.grid(True, axis='both', alpha=0.7, color='white', linewidth=1.5)

    ax1.hist(all_pearson, bins=25, color=color, alpha=0.85, edgecolor='white', linewidth=1.2)
    ax1.axvline(global_mean, color='#E24A33', linestyle='--', linewidth=2.5,
                label=f'Mean = {global_mean:.3f}')

    ax1.set_xlabel('Attn Reconstruction Pearson $\\bar{r}$')
    ax1.set_ylabel('Count')
    ax1.legend(frameon=False, fontsize=14)

    plt.tight_layout()
    fig1_path = output_dir / "full_model_pearson_histogram.png"
    fig1.savefig(fig1_path, dpi=200, bbox_inches='tight')
    plt.close(fig1)
    print(f"\nHistogram saved to {fig1_path}")

    # Figure 2: Per-layer percentage above mean
    fig2, ax2 = plt.subplots(figsize=(10, 5))
    ax2.set_facecolor(face_color)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.tick_params(top=False, bottom=False, left=False, right=False)
    ax2.grid(True, axis='y', alpha=0.7, color='white', linewidth=1.5)

    layers = np.arange(num_layers)
    ax2.bar(layers, layer_above_mean_pct, color=color_bar, alpha=0.85, edgecolor='white', linewidth=0.5)
    ax2.axhline(50, color='#333333', linestyle='--', linewidth=1.5, alpha=0.7, label='50%')

    ax2.set_xlabel('Layer Index')
    ax2.set_ylabel('% Heads Above Global Mean')
    ax2.set_xticks(layers[::2])  # Every other layer for readability
    ax2.set_ylim(0, 100)
    ax2.legend(frameon=False, fontsize=14)

    plt.tight_layout()
    fig2_path = output_dir / "per_layer_above_mean_pct.png"
    fig2.savefig(fig2_path, dpi=200, bbox_inches='tight')
    plt.close(fig2)
    print(f"Per-layer plot saved to {fig2_path}")


if __name__ == "__main__":
    main()
