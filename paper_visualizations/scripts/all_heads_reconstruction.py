"""Generate reconstruction vs ground truth visualization for ALL heads.

This script generates attention reconstruction comparison plots for every
layer and head in the model, saving them to a folder for inspection.

Based on: generate_intro_figure.py (subplot c/d logic)
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
import numpy as np
import torch
from scipy import stats
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 12
LABEL_FONT_SIZE = 14
color_q = (85 / 250, 104 / 250, 154 / 250)    # blue for reconstruction
color_k = (187 / 250, 130 / 250, 90 / 250)    # warm brown for ground truth
face_color = (231 / 250, 231 / 250, 240 / 250)


def style_ax(ax):
    """Apply consistent styling to axes."""
    ax.set_facecolor(face_color)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(top=False, bottom=False, left=False, right=False, labelsize=FONT_SIZE)
    ax.set_axisbelow(True)
    ax.grid(True, axis="both", alpha=0.7, color="white", linewidth=1.5)


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

    # Use log-spaced sampling for GT curve (median is more robust than mean)
    sample_distances = torch.unique(torch.logspace(0, math.log10(actual_max_dist), num_samples, device=device).long())
    sample_distances = sample_distances[(sample_distances >= 1) & (sample_distances <= actual_max_dist)]

    # Use FFT for efficient cross-correlation (mean)
    fft_len = 1 << (2 * token_count - 1).bit_length()
    q_fft = torch.fft.rfft(q_block.float(), n=fft_len, dim=0)
    k_fft = torch.fft.rfft(k_block.float(), n=fft_len, dim=0)
    corr = torch.fft.irfft((torch.conj(k_fft) * q_fft).sum(dim=1), n=fft_len, dim=0)
    all_counts = (token_count - torch.arange(0, actual_max_dist + 1, device=device, dtype=torch.float32)).clamp_min(1.0)
    gt_scores = corr[distances.long()] / all_counts[distances.long()]

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

    # Compute per-query Pearson correlation
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


def plot_single_head(recon_data: dict, layer: int, head: int, output_path: Path, dpi: int) -> None:
    """Generate and save reconstruction plot for a single head."""
    fig, ax = plt.subplots(figsize=(5, 5), dpi=dpi)
    style_ax(ax)

    # Error band (centered on mean, width = std)
    ax.fill_between(recon_data['sample_distances'],
                    recon_data['gt_means'] - recon_data['gt_stds'],
                    recon_data['gt_means'] + recon_data['gt_stds'],
                    color=color_k, alpha=0.25, linewidth=0)

    # Ground truth line
    ax.plot(recon_data['distances'], recon_data['gt_scores'],
            color=color_k, linestyle='--', linewidth=2.5, alpha=0.9, label='Ground Truth')

    # Reconstruction line
    ax.plot(recon_data['distances'], recon_data['reconstructed'],
            color=color_q, linestyle=':', linewidth=2.5, alpha=0.9, label='Trig. Recon.')

    # Log scale for x-axis
    ax.set_xscale('log')
    ax.set_xlim(1, 10000)
    ax.set_xticks([1, 10, 100, 1000, 10000])
    ax.set_xticklabels(['1', '10', '100', '1k', '10k'])
    ax.set_xlabel(r'Query-Key Distance $\Delta$', fontsize=FONT_SIZE)
    ax.set_ylabel('Attention Logit', fontsize=FONT_SIZE)
    ax.set_box_aspect(1)

    # Title with layer/head and Pearson r
    pearson_r = recon_data['per_query_pearson']
    ax.set_title(f'L{layer}H{head}  (r={pearson_r:.3f})', fontsize=LABEL_FONT_SIZE, fontweight='bold')

    ax.legend(frameon=False, fontsize=FONT_SIZE - 1, loc='lower left')

    fig.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate reconstruction plots for ALL heads")
    parser.add_argument(
        "--trace-dir",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        help="Trace directory containing qk.pt and metadata.json",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for RoPE parameters",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper_visualizations/outputs/all_heads_reconstruction"),
        help="Output directory for all head plots",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--dpi", type=int, default=150, help="DPI for saved figures")
    parser.add_argument("--max-distance", type=int, default=5000, help="Maximum token distance for reconstruction")
    parser.add_argument("--layers", type=str, default=None, help="Specific layers to process (e.g., '0,1,2' or '0-5')")
    parser.add_argument("--heads", type=str, default=None, help="Specific heads to process (e.g., '0,1,2' or '0-5')")
    return parser.parse_args()


def parse_range(range_str: str, max_val: int) -> list:
    """Parse range string like '0,1,2' or '0-5' into list of indices."""
    if range_str is None:
        return list(range(max_val))

    indices = []
    for part in range_str.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            indices.extend(range(int(start), int(end) + 1))
        else:
            indices.append(int(part))
    return [i for i in indices if 0 <= i < max_val]


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_all_heads")

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

    # ========== Load Trace Data ==========
    print(f"Loading trace from {args.trace_dir}...")
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"

    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {args.trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"  Sequence length: {token_count}")

    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    num_layers = q_tensor.shape[0]
    num_heads = q_tensor.shape[1]
    head_dim = q_tensor.shape[-1]
    print(f"  Model: {num_layers} layers, {num_heads} heads, head_dim={head_dim}")

    # Build RoPE tables
    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=torch.float32)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=torch.float32)
    sin_table = sin_table[0].to(dtype=torch.float32)

    # Parse layer/head ranges
    layers_to_process = parse_range(args.layers, num_layers)
    heads_to_process = parse_range(args.heads, num_heads)
    total_plots = len(layers_to_process) * len(heads_to_process)
    print(f"\nProcessing {len(layers_to_process)} layers x {len(heads_to_process)} heads = {total_plots} plots")

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Also create per-layer subdirectories
    for layer in layers_to_process:
        (args.output_dir / f"layer_{layer:02d}").mkdir(exist_ok=True)

    # ========== Generate All Plots ==========
    completed = 0
    pearson_results = []

    for layer in layers_to_process:
        for head in heads_to_process:
            q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
            k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

            recon_data = compute_reconstruction_data(
                q_block, k_block, cos_table, sin_table,
                attention_scale, inv_freq, args.max_distance, device
            )

            output_path = args.output_dir / f"layer_{layer:02d}" / f"L{layer:02d}_H{head:02d}.png"
            plot_single_head(recon_data, layer, head, output_path, args.dpi)

            pearson_results.append({
                'layer': layer,
                'head': head,
                'pearson_r': recon_data['per_query_pearson']
            })

            completed += 1
            if completed % 32 == 0 or completed == total_plots:
                print(f"  Progress: {completed}/{total_plots} ({100*completed/total_plots:.1f}%)")

    # ========== Save Summary ==========
    summary_path = args.output_dir / "pearson_summary.json"
    with open(summary_path, 'w') as f:
        json.dump(pearson_results, f, indent=2)

    # Print statistics
    all_r = [r['pearson_r'] for r in pearson_results]
    print(f"\n=== Summary ===")
    print(f"Total heads: {len(all_r)}")
    print(f"Pearson r: min={min(all_r):.4f}, max={max(all_r):.4f}, mean={np.mean(all_r):.4f}, std={np.std(all_r):.4f}")

    # Find best and worst heads
    sorted_results = sorted(pearson_results, key=lambda x: x['pearson_r'], reverse=True)
    print(f"\nTop 5 best reconstruction:")
    for r in sorted_results[:5]:
        print(f"  L{r['layer']}H{r['head']}: r={r['pearson_r']:.4f}")

    print(f"\nTop 5 worst reconstruction:")
    for r in sorted_results[-5:]:
        print(f"  L{r['layer']}H{r['head']}: r={r['pearson_r']:.4f}")

    print(f"\nAll plots saved to {args.output_dir}")
    print(f"Summary saved to {summary_path}")


if __name__ == "__main__":
    main()
