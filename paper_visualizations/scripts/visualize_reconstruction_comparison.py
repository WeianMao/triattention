"""Visualize attention map reconstruction comparison.

Creates two panels:
- (A) Reconstruction curve for L1H18 (same method as freq_magnitude_v2)
- (B) Original vs Reconstructed attention map
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
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.gridspec import GridSpec
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding


def pearsonr(x, y):
    """Compute Pearson correlation coefficient."""
    x = np.asarray(x)
    y = np.asarray(y)
    mx = x.mean()
    my = y.mean()
    xm = x - mx
    ym = y - my
    r_num = np.sum(xm * ym)
    r_den = np.sqrt(np.sum(xm**2) * np.sum(ym**2))
    if r_den == 0:
        return 0.0
    return r_num / r_den

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 12
color_gt = (187 / 250, 130 / 250, 90 / 250)
color_recon = (85 / 250, 104 / 250, 154 / 250)
face_color = (231 / 250, 231 / 250, 240 / 250)

# Darker blue hotspot (selected colormap)
hotspot_darker_blue = (30 / 255, 50 / 255, 120 / 255)
attn_cmap = LinearSegmentedColormap.from_list(
    "attn_darker_blue", [face_color, hotspot_darker_blue]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize reconstruction comparison")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--layer", type=int, default=1, help="Layer index")
    parser.add_argument("--head", type=int, default=18, help="Head index")
    parser.add_argument("--patch-size", type=int, default=32, help="Pooling patch size")
    parser.add_argument("--max-distance", type=int, default=5000, help="Maximum token distance")
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--output-path", type=Path, default=None, help="Output path")
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


def compute_reconstruction_curve(
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
    """Compute reconstruction curve data (same as panel A in freq_magnitude_v2)."""
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

    # Compute Pearson correlation
    mean_pearson = pearsonr(gt_scores.cpu().numpy(), reconstructed.cpu().numpy())

    return {
        'distances': distances.cpu().numpy(),
        'reconstructed': reconstructed.cpu().numpy(),
        'gt_scores': gt_scores.cpu().numpy(),
        'sample_distances': sample_distances.cpu().numpy(),
        'gt_means': np.array(gt_means_list),
        'gt_stds': np.array(gt_stds_list),
        'mean_pearson': mean_pearson,
        'omega': omega,
        'phi_f': phi_f,
        'amplitude': amplitude,
    }


def compute_attention_heatmap_original(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute original attention heatmap with pooling."""
    head_dim = q_block.shape[-1]
    scale = head_dim ** -0.5

    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    # Padding
    q_pad = num_q_groups * patch_size - seq_len
    k_pad = num_k_groups * patch_size - seq_len
    if q_pad > 0:
        q_block = torch.cat([q_block, torch.zeros(q_pad, head_dim, device=device, dtype=q_block.dtype)], dim=0)
    if k_pad > 0:
        k_block = torch.cat([k_block, torch.zeros(k_pad, head_dim, device=device, dtype=k_block.dtype)], dim=0)

    seq_k_padded = k_block.shape[0]
    key_positions = torch.arange(seq_k_padded, device=device)
    key_valid = key_positions < seq_len

    pooled_groups = torch.zeros((num_q_groups, num_k_groups), device=device, dtype=torch.float32)
    k_t = k_block.t().contiguous()

    q_tile = 512
    for q_start in range(0, seq_len, q_tile):
        q_end = min(q_start + q_tile, seq_len)
        indices = torch.arange(q_start, q_end, device=device)
        q_slice = q_block[q_start:q_end]

        scores = torch.matmul(q_slice, k_t) * scale
        scores = scores.to(torch.float32)

        # Causal mask
        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        scores = scores.masked_fill(future_mask, float("-inf"))
        scores = scores.masked_fill(~key_valid.view(1, -1), float("-inf"))

        # Softmax
        scores_flat = scores.view(-1, num_k_groups * patch_size)
        row_max = scores_flat.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        stable = torch.exp(scores_flat - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (stable / row_sum).view(-1, num_k_groups, patch_size)

        key_mask = key_valid.view(1, num_k_groups, patch_size)
        weights = weights * key_mask

        # Max pool
        pooled_k = weights.max(dim=-1).values

        query_groups = indices // patch_size
        for local_q, global_q_group in enumerate(query_groups.tolist()):
            pooled_groups[global_q_group] = torch.maximum(
                pooled_groups[global_q_group], pooled_k[local_q]
            )

    # Normalize
    pooled_groups = pooled_groups[:num_q_groups, :num_k_groups]
    row_min = pooled_groups.amin(dim=1, keepdim=True)
    row_max = pooled_groups.amax(dim=1, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_groups - row_min) / denom
    return norm.detach().cpu()


def compute_attention_heatmap_reconstructed(
    omega: torch.Tensor,
    phi_f: torch.Tensor,
    amplitude: torch.Tensor,
    seq_len: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute reconstructed attention heatmap using the frequency-based model."""
    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    pooled_groups = torch.zeros((num_q_groups, num_k_groups), device=device, dtype=torch.float32)

    q_tile = 512
    for q_start in range(0, seq_len, q_tile):
        q_end = min(q_start + q_tile, seq_len)
        tile_size = q_end - q_start

        # Create distance matrix for this tile
        query_positions = torch.arange(q_start, q_end, device=device, dtype=torch.float32)
        key_positions = torch.arange(seq_len, device=device, dtype=torch.float32)

        # delta = query_pos - key_pos (for causal: only positive deltas are valid)
        delta_matrix = query_positions.unsqueeze(1) - key_positions.unsqueeze(0)  # [tile_size, seq_len]

        # Reconstruct scores: sum over frequencies of cos(delta * omega + phi) * amplitude
        # For delta <= 0 (future keys), we'll mask later
        delta_flat = delta_matrix.reshape(-1, 1)  # [tile_size * seq_len, 1]
        phase = delta_flat * omega.unsqueeze(0) + phi_f.unsqueeze(0)  # [tile_size * seq_len, num_freq]
        scores_flat = (torch.cos(phase) * amplitude.unsqueeze(0)).sum(dim=1)  # [tile_size * seq_len]
        scores = scores_flat.reshape(tile_size, seq_len)

        # Causal mask (key_pos > query_pos means future)
        causal_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Pad to match pooling grid
        if seq_len < num_k_groups * patch_size:
            pad_size = num_k_groups * patch_size - seq_len
            scores = torch.cat([scores, torch.full((tile_size, pad_size), float("-inf"), device=device)], dim=1)

        # Softmax
        scores_flat = scores.view(tile_size, num_k_groups * patch_size)
        row_max = scores_flat.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        stable = torch.exp(scores_flat - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (stable / row_sum).view(tile_size, num_k_groups, patch_size)

        # Max pool
        pooled_k = weights.max(dim=-1).values

        query_groups = torch.arange(q_start, q_end, device=device) // patch_size
        for local_q, global_q_group in enumerate(query_groups.tolist()):
            pooled_groups[global_q_group] = torch.maximum(
                pooled_groups[global_q_group], pooled_k[local_q]
            )

    # Normalize
    pooled_groups = pooled_groups[:num_q_groups, :num_k_groups]
    row_min = pooled_groups.amin(dim=1, keepdim=True)
    row_max = pooled_groups.amax(dim=1, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_groups - row_min) / denom
    return norm.detach().cpu()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_recon_compare")

    device = torch.device(args.device)
    dtype = torch.float32

    # Load data
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {args.trace_dir}")

    meta = json.loads(meta_path.read_text())
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")

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
    print("Loading Q/K tensors...")
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

    # Compute reconstruction curve
    print(f"Computing reconstruction curve for Layer {args.layer}, Head {args.head}...")
    recon_data = compute_reconstruction_curve(
        q_block, k_block, cos_table, sin_table,
        attention_scale, inv_freq, args.max_distance, device
    )

    # Compute attention heatmaps
    print("Computing original attention heatmap...")
    heatmap_orig = compute_attention_heatmap_original(
        q_block, k_block, token_count, args.patch_size, device
    )

    print("Computing reconstructed attention heatmap...")
    heatmap_recon = compute_attention_heatmap_reconstructed(
        recon_data['omega'], recon_data['phi_f'], recon_data['amplitude'],
        token_count, args.patch_size, device
    )

    # ========== Plotting ==========
    fig = plt.figure(figsize=(14, 5), dpi=args.dpi)
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.2, 1, 1], wspace=0.25)

    def style_ax(ax):
        ax.set_facecolor(face_color)
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)

    # (A) Reconstruction curve
    ax_a = fig.add_subplot(gs[0])
    style_ax(ax_a)
    ax_a.set_axisbelow(True)
    ax_a.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)

    ax_a.fill_between(recon_data['sample_distances'],
                      recon_data['gt_means'] - recon_data['gt_stds'],
                      recon_data['gt_means'] + recon_data['gt_stds'],
                      color=color_gt, alpha=0.25, linewidth=0, label='Ground Truth')
    ax_a.plot(recon_data['distances'], recon_data['gt_scores'],
              color=color_gt, linestyle='--', linewidth=2.5, alpha=0.9, label='GT Mean')
    ax_a.plot(recon_data['distances'], recon_data['reconstructed'],
              color=color_recon, linestyle=':', linewidth=2.5, alpha=0.9, label='Reconstruction')
    ax_a.set_xscale('log')
    ax_a.set_xlabel(r'QK Relative Position $\Delta$', fontsize=FONT_SIZE)
    ax_a.set_ylabel(r'$\langle q, k \rangle_\Delta$', fontsize=FONT_SIZE)
    ax_a.legend(frameon=False, fontsize=FONT_SIZE - 1, loc='upper right')
    ax_a.text(0.03, 0.03,
              f"Pearson $r$ = {recon_data['mean_pearson']:.2f}",
              transform=ax_a.transAxes, fontsize=FONT_SIZE,
              verticalalignment='bottom', horizontalalignment='left',
              bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8, edgecolor='none'))
    ax_a.set_title(f'(A) L{args.layer}H{args.head} Reconstruction Curve', fontsize=FONT_SIZE + 1, fontweight='bold')

    # (B) Original attention map
    ax_b = fig.add_subplot(gs[1])
    ax_b.imshow(heatmap_orig.numpy(), cmap=attn_cmap, aspect="equal", origin="upper")
    ax_b.set_title(f'(B) Original Attention', fontsize=FONT_SIZE + 1, fontweight='bold')
    ax_b.set_xticks([])
    ax_b.set_yticks([])
    for spine in ax_b.spines.values():
        spine.set_visible(False)

    # (C) Reconstructed attention map
    ax_c = fig.add_subplot(gs[2])
    ax_c.imshow(heatmap_recon.numpy(), cmap=attn_cmap, aspect="equal", origin="upper")
    ax_c.set_title(f'(C) Reconstructed Attention', fontsize=FONT_SIZE + 1, fontweight='bold')
    ax_c.set_xticks([])
    ax_c.set_yticks([])
    for spine in ax_c.spines.values():
        spine.set_visible(False)

    plt.tight_layout()

    # Save
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = Path("paper_visualizations/outputs/early_layer_heads/fig_reconstruction_comparison.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)

    print(f"\nFigure saved to {output_path}")
    print(f"Pearson correlation: {recon_data['mean_pearson']:.4f}")


if __name__ == "__main__":
    main()
