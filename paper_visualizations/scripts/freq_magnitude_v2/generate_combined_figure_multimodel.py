"""Generate combined figure for multiple model architectures (Qwen2, Qwen3, Llama).

Adapted from generate_combined_figure.py and full_model_correlation_analysis.py.
Computes correlation for all heads and generates the combined visualization.

Usage:
    conda activate rkv
    python paper_visualizations/scripts/freq_magnitude_v2/generate_combined_figure_multimodel.py \
        paper_visualizations/outputs/qk_traces/deepseek_r1_llama_8b/trace_aime24_q70 \
        --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Llama-8B \
        --output-path paper_visualizations/outputs/freq_magnitude_v2/fig_freq_reconstruction_llama.png \
        --gpu 1
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def get_model_family(config) -> str:
    """Detect model family from config."""
    model_type = getattr(config, "model_type", "").lower()
    architectures = getattr(config, "architectures", [])

    if "qwen3" in model_type or any("Qwen3" in a for a in architectures):
        return "qwen3"
    elif "qwen2" in model_type or any("Qwen2" in a for a in architectures):
        return "qwen2"
    elif "llama" in model_type or any("Llama" in a for a in architectures):
        return "llama"
    else:
        raise ValueError(f"Unsupported model type: {model_type}, architectures: {architectures}")


def get_rotary_embedding(config, model_family: str, device):
    """Get the appropriate RotaryEmbedding class for the model family."""
    if model_family == "qwen3":
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
        return Qwen3RotaryEmbedding(config=config, device=device)
    elif model_family == "qwen2":
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding
        return Qwen2RotaryEmbedding(config=config, device=device)
    elif model_family == "llama":
        from transformers.models.llama.modeling_llama import LlamaRotaryEmbedding
        return LlamaRotaryEmbedding(config=config, device=device)
    else:
        raise ValueError(f"Unknown model family: {model_family}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate combined figure for frequency-magnitude reconstruction (multi-model)"
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU device index")
    parser.add_argument("--layer", type=int, default=0, help="Layer index for panel A")
    parser.add_argument("--head", type=int, default=0, help="Head index for panel A")
    parser.add_argument("--threshold", type=float, default=0.55, help="Correlation threshold for panel C")
    parser.add_argument("--max-distance", type=int, default=5000, help="Maximum token distance")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
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

    per_query_pearsons = []

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

    return np.mean(per_query_pearsons) if per_query_pearsons else 0.0


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

    # Per-query correlation
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


def main() -> None:
    args = parse_args()
    mask_process_command(f"PD-L1_binder_fig{args.gpu}")

    np.random.seed(42)

    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    dtype = torch.float32

    # Load trace metadata
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {args.trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])

    # Load model config and detect family
    print(f"Loading config from {args.model_path}...")
    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    model_family = get_model_family(config)
    print(f"Detected model family: {model_family}")

    # Handle rope_scaling
    rope_scaling = config.rope_scaling
    if rope_scaling is not None:
        rope_scaling = dict(rope_scaling)
        if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
            rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
        rope_scaling.pop("attn_factor", None)
        # If rope_scaling becomes empty, set to None
        if not rope_scaling:
            rope_scaling = None
    config.rope_scaling = rope_scaling

    # Get rotary embedding
    rotary = get_rotary_embedding(config, model_family, device)
    inv_freq = rotary.inv_freq.to(torch.float64)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    # Load Q/K tensors
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
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

    # ========== Compute correlation for all heads (Panels B & C) ==========
    print("\nComputing correlation for all heads...")
    results = []
    threshold = args.threshold

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
                'ind_pearson': float(pearson),
            })

        print(f"  Layer {layer + 1}/{num_layers} done")

    all_pearson = np.array([r['ind_pearson'] for r in results])

    # Per-layer stats
    layer_above_thr_pct = []
    for layer in range(num_layers):
        layer_pearson = [r['ind_pearson'] for r in results if r['layer'] == layer]
        above_thr = sum(1 for p in layer_pearson if p > threshold)
        pct = above_thr / len(layer_pearson) * 100
        layer_above_thr_pct.append(pct)

    # ========== Compute panel A data ==========
    print(f"\nComputing panel A data (Layer {args.layer}, Head {args.head})...")
    q_block = q_tensor[args.layer, args.head, :token_count].to(device=device, dtype=dtype)
    k_block = k_tensor[args.layer, args.head, :token_count].to(device=device, dtype=dtype)

    panel_a = compute_panel_a_data(
        q_block, k_block, cos_table, sin_table,
        attention_scale, inv_freq, args.max_distance, device
    )

    # ========== Plotting ==========
    print("\nGenerating figure...")
    color_gt = (187 / 250, 130 / 250, 90 / 250)
    color_recon = (85 / 250, 104 / 250, 154 / 250)
    color_bar = (187 / 250, 130 / 250, 90 / 250)
    face_color = (231 / 250, 231 / 250, 240 / 250)

    FONT_SIZE = 14
    LABEL_FONT_SIZE = 18
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
                      color=color_gt, alpha=0.25, linewidth=0, label='Ground Truth')
    ax_a.plot(panel_a['distances'], panel_a['gt_scores'],
              color=color_gt, linestyle='--', linewidth=2.5, alpha=0.9, label='GT Mean')
    ax_a.plot(panel_a['distances'], panel_a['reconstructed'],
              color=color_recon, linestyle=':', linewidth=2.5, alpha=0.9, label='Reconstruction')
    ax_a.set_xscale('log')
    ax_a.set_xlabel(r'QK Relative Position $\Delta$', fontsize=FONT_SIZE)
    ax_a.set_ylabel(r'$\langle q, k \rangle_\Delta$', fontsize=FONT_SIZE)
    ax_a.legend(frameon=False, fontsize=FONT_SIZE, loc='upper right', bbox_to_anchor=(1.02, 1.0))
    ax_a.text(0.03, 0.03,
              f"Attn Reconstruction Pearson $\\bar{{r}}$ = {panel_a['per_query_pearson']:.2f}",
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
                 label=f'Mean = {all_pearson.mean():.2f}')
    ax_b.set_xlabel('Attn Reconstruction Pearson $\\bar{r}$', fontsize=FONT_SIZE)
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
    ax_c.set_ylabel(f'% Heads with $\\bar{{r}}$ > {threshold:.2f}', fontsize=FONT_SIZE)
    ax_c.set_xticks(layers_arr[::2])
    ax_c.set_ylim(0, 100)

    handles, labels = ax_c.get_legend_handles_labels()
    handles.insert(0, Patch(facecolor=color_bar, alpha=0.85, edgecolor='white', label='Per-layer percentage'))
    labels.insert(0, 'Per-layer percentage')
    ax_c.legend(handles, labels, frameon=False, fontsize=FONT_SIZE, loc='upper right', bbox_to_anchor=(1.0, 1.05))
    ax_c.text(-0.04, 1.08, '(C)', transform=ax_c.transAxes, fontsize=LABEL_FONT_SIZE, fontweight='bold', va='bottom', fontname=LABEL_FONT)

    plt.tight_layout()

    # Save
    args.output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"\nFigure saved to {args.output_path}")
    print(f"\nPanel (A): Layer {args.layer}, Head {args.head}")
    print(f"  Attn Reconstruction Pearson r = {panel_a['per_query_pearson']:.4f}")
    print(f"\nPanel (B): {len(all_pearson)} heads")
    print(f"  Mean = {all_pearson.mean():.4f}, Std = {all_pearson.std():.4f}")
    print(f"\nPanel (C): Threshold = {threshold}")


if __name__ == "__main__":
    main()
