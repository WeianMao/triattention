"""Visualize post-RoPE key distribution with Top-K Frequency NMS pruning.

This script creates scatter plots showing keys in RoPE-rotated space,
color-coded by Top-K frequency NMS retention decision (blue=retained, red=dropped).
Uses only the top K highest-energy frequency bands for NMS computation.
Default: top_k=24 for ~99.5% retention with p10/p90 percentile weights.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command
from weian_development.hf_offline_runner_sparse.round_pruning_utils import (
    invert_rope,
    to_complex_pairs,
    build_rotary,
    compute_rotary_tables,
)

mask_process_command("PD-L1_binder_post_rope_viz_top6")


def compute_q_magnitude_percentile_weights(
    q_complex: torch.Tensor,
    low_percentile: float = 20.0,
    high_percentile: float = 80.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute Q-magnitude percentile weights for variance-aware NMS.

    Args:
        q_complex: [num_samples, freq_count] Q vectors in complex form (unrotated)
        low_percentile: percentile for positive-score frequencies (default 20)
        high_percentile: percentile for negative-score frequencies (default 80)

    Returns:
        w_low: [freq_count] low percentile Q-magnitude weights
        w_high: [freq_count] high percentile Q-magnitude weights

    Raises:
        ValueError: if low_percentile > high_percentile
    """
    if low_percentile > high_percentile:
        raise ValueError(
            f"low_percentile ({low_percentile}) must be <= high_percentile ({high_percentile}). "
            f"Violating this breaks the conservative weight selection principle."
        )
    q_magnitudes = torch.abs(q_complex)
    w_low = torch.quantile(q_magnitudes, low_percentile / 100.0, dim=0)
    w_high = torch.quantile(q_magnitudes, high_percentile / 100.0, dim=0)
    return w_low, w_high


def get_top_k_frequency_indices(k_complex: torch.Tensor, top_k: int = 6) -> torch.Tensor:
    """Select top-K frequency bands by mean magnitude energy.

    Args:
        k_complex: [N, F] tensor of K vectors in complex form
        top_k: Number of top frequency bands to select (default: 6)

    Returns:
        top_indices: [top_k] tensor of selected frequency indices
    """
    k_mean = k_complex.mean(dim=0)  # [freq_count]
    k_magnitude = torch.abs(k_mean)  # [freq_count]
    top_k = min(top_k, k_magnitude.shape[0])
    _, top_indices = torch.topk(k_magnitude, k=top_k)
    return top_indices


def variance_aware_fast_nms(
    k_complex: torch.Tensor,
    w_low: torch.Tensor,
    w_high: torch.Tensor,
    top_k: int = 24,
) -> torch.Tensor:
    """Variance-aware Fast Parallel NMS with conservative weight selection.

    A suppresses B when conservative_coverage_score(A, B) > 0

    Args:
        k_complex: [N, F] tensor of RoPE-rotated K converted to complex pairs
        w_low: [F] tensor of low percentile Q-magnitude weights
        w_high: [F] tensor of high percentile Q-magnitude weights

    Returns:
        keep_mask: [N] bool tensor, True = keep
    """
    N, F = k_complex.shape

    # Top-K Frequency Band Selection - only use highest energy frequencies
    top_indices = get_top_k_frequency_indices(k_complex, top_k=top_k)
    k_complex = k_complex[:, top_indices]  # Filter K to Top-K frequencies
    w_low = w_low[top_indices]  # Filter weights to Top-K frequencies
    w_high = w_high[top_indices]  # Filter weights to Top-K frequencies

    # 1. Compute per-K per-frequency magnitude
    k_abs = torch.abs(k_complex)
    k_abs_safe = k_abs.clamp(min=1e-5)

    # 2. Compute A's projection onto B direction (per-frequency)
    real_dot = torch.einsum("af,bf->abf", k_complex, k_complex.conj()).real
    proj_on_b = real_dot / k_abs_safe.unsqueeze(0)

    # 3. Per-frequency coverage score: proj - |B|
    per_freq_score = proj_on_b - k_abs.unsqueeze(0)

    # 4. Conservative weight selection based on score sign
    weights = torch.where(
        per_freq_score > 0,
        w_low.view(1, 1, -1),
        w_high.view(1, 1, -1),
    )

    # 5. Weighted sum for conservative coverage score
    conservative_score = (per_freq_score * weights).sum(dim=2)

    # 6. A suppresses B when conservative_score > 0 (epsilon=0)
    suppresses = conservative_score > 0
    suppresses.fill_diagonal_(False)

    # 7. B is suppressed if any A suppresses it
    is_suppressed = suppresses.any(dim=0)
    keep_mask = ~is_suppressed

    return keep_mask


def load_qk_data(
    trace_path: Path, layer_idx: int, head_idx: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Load Q/K tensors from qk.pt and select (layer, head) indices.

    Args:
        trace_path: Path to trace directory containing qk.pt
        layer_idx: Layer index to extract
        head_idx: Head index to extract

    Returns:
        q_head: [seq, head_dim] Q tensor for specified (layer, head)
        k_head: [seq, head_dim] K tensor for specified (layer, head) (already rotated)

    Raises:
        ValueError: if layer_idx or head_idx are out of bounds
    """
    qk_path = trace_path / "qk.pt"
    if not qk_path.exists():
        raise FileNotFoundError(f"qk.pt not found at {qk_path}")

    # Load on CPU to avoid GPU OOM, then extract only the needed head
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    num_layers, num_heads = q_tensor.shape[:2]
    if not (0 <= layer_idx < num_layers):
        raise ValueError(f"layer_idx {layer_idx} out of bounds [0, {num_layers})")
    if not (0 <= head_idx < num_heads):
        raise ValueError(f"head_idx {head_idx} out of bounds [0, {num_heads})")

    # Extract single head and move to target device (much smaller memory footprint)
    q_head = q_tensor[layer_idx, head_idx].clone()
    k_head = k_tensor[layer_idx, head_idx].clone()

    return q_head, k_head


def setup_rope_env(
    model_path: Path, seq_len: int, head_dim: int, dtype: torch.dtype, device: torch.device
) -> tuple[object, float, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Build RoPE embedding and compute rotary tables for sequence.

    Args:
        model_path: Path to model config directory
        seq_len: Sequence length
        head_dim: Head dimension
        dtype: Tensor dtype
        device: Device to use

    Returns:
        rotary: RoPE embedding object
        attention_scale: Attention scaling factor
        cos_table: [seq_len, head_dim] cosine table
        sin_table: [seq_len, head_dim] sine table
        inv_freq: Inverse frequency tensor
        freq_scale: Frequency scaling tensor
    """
    rotary = build_rotary(device, model_path, dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))
    cos_table, sin_table, inv_freq, freq_scale = compute_rotary_tables(
        rotary, seq_len, head_dim, dtype, device
    )
    return rotary, attention_scale, cos_table, sin_table, inv_freq, freq_scale


def compute_nms_mask(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    low_percentile: float,
    high_percentile: float,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    inv_freq: torch.Tensor,
    attention_scale: float,
    use_pre_rope_viz: bool = False,
    top_k: int = 24,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert keys to complex, compute weights, apply NMS to get keep_mask.

    Args:
        q_head: [seq, head_dim] Q tensor (rotated)
        k_head: [seq, head_dim] K tensor (rotated)
        low_percentile: Low percentile for Q-magnitude weights
        high_percentile: High percentile for Q-magnitude weights
        cos_table: [seq, head_dim] cosine table
        sin_table: [seq, head_dim] sine table
        inv_freq: Inverse frequency tensor
        attention_scale: Attention scaling factor
        use_pre_rope_viz: If True, return pre-RoPE K for visualization (like reference scripts)

    Returns:
        keep_mask: [N] boolean array, True=retained, False=dropped
        k_complex_for_viz: [seq, freq_count] complex tensor for visualization
    """
    # NMS operates on POST-RoPE K (k_head is already rotated from qk.pt)
    k_rotated_complex = to_complex_pairs(k_head)

    # Invert RoPE on q_head to get unrotated Q for statistics
    q_unrotated = invert_rope(q_head, cos_table, sin_table, attention_scale)
    q_complex = to_complex_pairs(q_unrotated)

    # Compute Q magnitude percentile weights
    # q_complex is [seq, freq_count], compute_q_magnitude_percentile_weights expects [num_samples, freq_count]
    w_low, w_high = compute_q_magnitude_percentile_weights(
        q_complex, low_percentile, high_percentile
    )

    # Apply NMS on CPU to avoid OOM (N×N×F tensor is too large for GPU)
    k_rotated_complex_cpu = k_rotated_complex.cpu().to(torch.complex64)
    w_low_cpu = w_low.cpu().float()
    w_high_cpu = w_high.cpu().float()
    keep_mask = variance_aware_fast_nms(k_rotated_complex_cpu, w_low_cpu, w_high_cpu, top_k)

    # For visualization: choose pre-RoPE or post-RoPE representation
    if use_pre_rope_viz:
        # Pre-RoPE (like reference visualization scripts)
        k_unrotated = invert_rope(k_head, cos_table, sin_table, attention_scale)
        k_complex_for_viz = to_complex_pairs(k_unrotated)
    else:
        # Post-RoPE (same as what NMS operates on)
        k_complex_for_viz = k_rotated_complex

    return keep_mask, k_complex_for_viz


def plot_post_rope_scatter(
    k_complex: torch.Tensor,
    keep_mask: torch.Tensor,
    output_path: Path,
    layer_idx: int,
    head_idx: int,
    top_k: int = 3,
    is_pre_rope: bool = False,
) -> None:
    """Create scatter plot showing retained vs dropped keys in complex plane.

    Visualizes top-k frequencies by magnitude, each in a separate subplot.

    Args:
        k_complex: [seq, freq_count] complex tensor of keys (pre or post RoPE)
        keep_mask: [seq] boolean mask, True=retained, False=dropped
        output_path: Path to save the plot
        layer_idx: Layer index for title
        head_idx: Head index for title
        top_k: Number of top frequencies to visualize (default 3)
        is_pre_rope: If True, title indicates pre-RoPE space
    """
    seq_len, freq_count = k_complex.shape

    # Compute mean magnitude per frequency for ranking
    k_mean = k_complex.mean(dim=0)  # [freq_count]
    k_magnitude = torch.abs(k_mean)  # [freq_count]

    # Get top-k frequency indices by magnitude
    top_k = min(top_k, freq_count)
    _, top_indices = torch.topk(k_magnitude, k=top_k)
    top_indices = top_indices.cpu().tolist()

    # Create figure with subplots (1 row, top_k columns)
    fig, axes = plt.subplots(1, top_k, figsize=(6 * top_k, 6), dpi=100)
    if top_k == 1:
        axes = [axes]

    num_retained = keep_mask.sum().item()
    num_total = len(keep_mask)

    for ax_idx, freq_idx in enumerate(top_indices):
        ax = axes[ax_idx]

        # Extract this frequency's data: [seq]
        k_freq = k_complex[:, freq_idx]

        # Separate retained vs dropped
        k_retained = k_freq[keep_mask]
        k_dropped = k_freq[~keep_mask]

        # Convert to numpy
        k_retained_real = k_retained.real.detach().cpu().numpy()
        k_retained_imag = k_retained.imag.detach().cpu().numpy()
        k_dropped_real = k_dropped.real.detach().cpu().numpy()
        k_dropped_imag = k_dropped.imag.detach().cpu().numpy()

        # Plot retained keys in blue
        ax.scatter(
            k_retained_real,
            k_retained_imag,
            c="blue",
            s=12,
            alpha=0.6,
            label="Retained",
            edgecolors="none",
        )

        # Plot dropped keys in red
        if len(k_dropped_real) > 0:
            ax.scatter(
                k_dropped_real,
                k_dropped_imag,
                c="red",
                s=20,
                alpha=0.8,
                label="Dropped",
                edgecolors="none",
            )

        # Add axes at zero
        ax.axhline(0, color="k", linewidth=0.5)
        ax.axvline(0, color="k", linewidth=0.5)

        # Set equal aspect
        ax.set_aspect("equal", adjustable="box")

        # Add labels
        ax.set_xlabel("Re(k)")
        ax.set_ylabel("Im(k)")

        # Add title
        freq_mag = k_magnitude[freq_idx].item()
        ax.set_title(
            f"Freq {freq_idx} (mag={freq_mag:.2f})\n"
            f"Rank {ax_idx + 1}/{freq_count}"
        )

        # Add legend only to first subplot
        if ax_idx == 0:
            ax.legend(loc="upper right")

        # Add grid
        ax.grid(alpha=0.2, linestyle="--")

    # Add overall title
    rope_label = "Pre-RoPE" if is_pre_rope else "Post-RoPE"
    fig.suptitle(
        f"{rope_label} Key Distribution (Layer {layer_idx}, Head {head_idx}) - Top {top_k} Frequencies\n"
        f"Retained: {num_retained}/{num_total} keys ({100*num_retained/num_total:.1f}%)",
        fontsize=12,
    )

    plt.tight_layout()

    # Save figure
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize post-RoPE key distribution with NMS pruning tracking."
    )
    parser.add_argument(
        "--trace",
        type=str,
        required=True,
        help="Trace folder name (e.g., qid0003_trace34)",
    )
    parser.add_argument(
        "--head-sample-file",
        type=Path,
        default=Path("weian_development/online_k_pruning_viz/hybrid_sample_heads_lowret_top10.json"),
        help="Path to JSON file storing sampled (layer, head) indices",
    )
    parser.add_argument(
        "--low-percentile",
        type=float,
        default=20.0,
        help="Low percentile for Q-magnitude weights (default: 20.0)",
    )
    parser.add_argument(
        "--high-percentile",
        type=float,
        default=80.0,
        help="High percentile for Q-magnitude weights (default: 80.0)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("./post_rope_nms_viz"),
        help="Output directory for plots (default: ./post_rope_nms_viz)",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Path to model config directory",
    )
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/outputs/deepseek_r1_qwen3_8b/qk_bf16_traces"),
        help="Root directory containing trace folders",
    )
    parser.add_argument(
        "--pre-rope",
        action="store_true",
        help="Visualize in pre-RoPE space (like reference scripts). Default is post-RoPE.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=24,
        help="Number of top frequency bands to use for NMS (default: 24).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Setup device and dtype
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16

    # Load head sample file
    if not args.head_sample_file.exists():
        raise FileNotFoundError(f"Head sample file not found: {args.head_sample_file}")

    with open(args.head_sample_file) as f:
        sample_data = json.load(f)
        # Support both list format [[layer, head], ...] and dict format {"sampled_heads": [...]}
        if isinstance(sample_data, list):
            sampled_heads = sample_data
        else:
            sampled_heads = sample_data["sampled_heads"]

    # Create output directory
    args.output_dir.mkdir(parents=True, exist_ok=True)

    # Construct trace path
    trace_path = args.input_root / args.trace

    print(f"Processing trace: {args.trace}")
    print(f"Device: {device}, dtype: {dtype}")
    print(f"Output directory: {args.output_dir}")
    print(f"Processing {len(sampled_heads)} heads...")

    # Process each sampled head
    for idx, (layer, head) in enumerate(sampled_heads):
        try:
            # Load Q/K data
            q_head, k_head = load_qk_data(trace_path, layer, head)
            seq_len, head_dim = k_head.shape

            # Move to device
            q_head = q_head.to(device=device, dtype=dtype)
            k_head = k_head.to(device=device, dtype=dtype)

            # Setup RoPE environment
            rotary, attention_scale, cos_table, sin_table, inv_freq, freq_scale = setup_rope_env(
                args.model_path, seq_len, head_dim, dtype, device
            )

            # Compute NMS mask
            keep_mask, k_complex_viz = compute_nms_mask(
                q_head,
                k_head,
                args.low_percentile,
                args.high_percentile,
                cos_table,
                sin_table,
                inv_freq,
                attention_scale,
                use_pre_rope_viz=args.pre_rope,
                top_k=args.top_k,
            )

            # Plot scatter
            rope_prefix = "pre_rope" if args.pre_rope else "post_rope"
            output_path = args.output_dir / f"{rope_prefix}_nms_L{layer}H{head}.png"
            plot_post_rope_scatter(
                k_complex_viz, keep_mask, output_path, layer, head,
                is_pre_rope=args.pre_rope,
            )

            num_retained = keep_mask.sum().item()
            num_total = len(keep_mask)
            print(
                f"  [{idx+1}/{len(sampled_heads)}] L{layer}H{head}: "
                f"retained {num_retained}/{num_total} ({100*num_retained/num_total:.1f}%)"
            )

        except Exception as e:
            print(f"  Error processing L{layer}H{head}: {e}")
            continue

    # Cleanup
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print(f"\nCompleted! Plots saved to {args.output_dir}")


if __name__ == "__main__":
    main()
