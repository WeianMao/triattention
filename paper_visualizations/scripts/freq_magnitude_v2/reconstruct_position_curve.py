"""Reconstruct relative position curve using mean vector statistics.

Formula: ⟨q, k⟩_Δ = Σ_f |q_f| |k_f| cos(ω_f Δ + φ_f)

Where:
- q_f = E[q_f] (mean of pre-RoPE q vectors)
- k_f = E[k_f] (mean of pre-RoPE k vectors)
- φ_f = arg(q_f) - arg(k_f) (phase difference of mean vectors)
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
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reconstruct position curve using mean vector formula"
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
    parser.add_argument("--dpi", type=int, default=150, help="Figure DPI")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for the figure",
    )
    return parser.parse_args()


def select_dtype(name: str) -> torch.dtype:
    return {"float32": torch.float32, "float16": torch.float16, "bfloat16": torch.bfloat16}[name]


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Rotate half for RoPE."""
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
    """Invert YaRN-scaled RoPE to recover pre-rotation vectors."""
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    z = rotated / scale
    return z * cos - rotate_half(z) * sin


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [seq_len, head_dim] to [seq_len, num_freq] complex tensor.

    Assumes head_dim layout is [real_0, real_1, ..., imag_0, imag_1, ...].
    """
    seq_len, head_dim = tensor.shape
    if head_dim % 2 != 0:
        raise ValueError("head_dim must be even")
    num_freq = head_dim // 2
    # Standard RoPE layout: first half is real, second half is imaginary
    real = tensor[:, :num_freq]
    imag = tensor[:, num_freq:]
    return torch.complex(real.float(), imag.float())


def compute_gt_curve(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    distances: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    """Compute ground-truth Q·K average per distance using FFT-based correlation.

    Returns gt_scores aligned with the input distances tensor.
    """
    token_count = q_block.shape[0]
    max_dist = int(distances.max().item())

    # FFT-based cross-correlation
    fft_len = 1 << (2 * token_count - 1).bit_length()
    q_fft = torch.fft.rfft(q_block.float(), n=fft_len, dim=0)
    k_fft = torch.fft.rfft(k_block.float(), n=fft_len, dim=0)

    # Cross-correlation: sum over head_dim
    prod = torch.conj(k_fft) * q_fft
    corr = torch.fft.irfft(prod.sum(dim=1), n=fft_len, dim=0)

    # Normalize by count of pairs at each distance
    all_counts = (token_count - torch.arange(0, max_dist + 1, device=device, dtype=torch.float32)).clamp_min(1.0)

    # Extract values at requested distances
    dist_indices = distances.long()
    gt_scores = corr[dist_indices] / all_counts[dist_indices]

    return gt_scores


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
    q_tensor = data["q"]  # [L, H, T, D]
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

    # 1. Invert RoPE to get pre-rotation Q/K
    q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)

    # 2. Convert to complex representation
    q_complex = to_complex_pairs(q_orig)  # [seq_len, num_freq]
    k_complex = to_complex_pairs(k_orig)

    # 3. Compute mean vectors: E[q_f], E[k_f]
    q_mean = q_complex.mean(dim=0)  # [num_freq] complex
    k_mean = k_complex.mean(dim=0)

    # 4. Extract magnitudes and phases from mean vectors
    q_f_mag = torch.abs(q_mean)  # |E[q_f]|
    k_f_mag = torch.abs(k_mean)  # |E[k_f]|
    q_f_phase = torch.angle(q_mean)  # arg(E[q_f])
    k_f_phase = torch.angle(k_mean)  # arg(E[k_f])

    # 5. Phase difference: φ_f = arg(q_f) - arg(k_f)
    phi_f = q_f_phase - k_f_phase

    # 6. Amplitude product: |q_f| * |k_f|
    amplitude = q_f_mag * k_f_mag

    # 7. Compute ω_f from RoPE frequencies
    # period = 2π / ω, so ω = 2π / period = inv_freq
    omega = inv_freq[:num_freq].to(device=device, dtype=torch.float32)

    # 8. Generate distance values
    max_dist = min(args.max_distance, token_count - 1)
    # Use linear spacing for cleaner curve, but also show log scale
    distances = torch.arange(1, max_dist + 1, device=device, dtype=torch.float32)

    # 9. Reconstruct: ⟨q, k⟩_Δ = Σ_f |q_f| |k_f| cos(ω_f Δ + φ_f)
    # Shape: [num_distances, num_freq]
    phase_matrix = distances.unsqueeze(1) * omega.unsqueeze(0) + phi_f.unsqueeze(0)
    cos_matrix = torch.cos(phase_matrix)
    reconstructed = (cos_matrix * amplitude.unsqueeze(0)).sum(dim=1)

    # 10. Compute ground-truth (using same distances for alignment)
    gt_scores = compute_gt_curve(q_block, k_block, distances, device)

    # ========== Plotting ==========

    dist_np = distances.cpu().numpy()
    recon_np = reconstructed.cpu().numpy()
    gt_np = gt_scores.cpu().numpy()

    # Use a clean style
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, ax = plt.subplots(figsize=(8, 5), dpi=args.dpi)

    # Plot GT first (background), then reconstructed (foreground)
    ax.plot(dist_np, gt_np,
            color="#E24A33", linestyle="-", linewidth=2.2, alpha=0.85,
            label="Ground Truth")
    ax.plot(dist_np, recon_np,
            color="#348ABD", linestyle="--", linewidth=2.0, alpha=0.95,
            label="Reconstructed")

    ax.set_xscale("log")
    ax.set_xlabel("Relative Position $\\Delta$", fontsize=12)
    ax.set_ylabel("$\\langle q, k \\rangle_\\Delta$", fontsize=12)

    # Clean legend
    ax.legend(frameon=True, fancybox=True, framealpha=0.9, fontsize=10, loc="upper right")

    # Refined grid
    ax.grid(True, alpha=0.4, linestyle="-", linewidth=0.5)
    ax.set_axisbelow(True)

    # Tick styling
    ax.tick_params(axis='both', which='major', labelsize=10)

    # Spine styling
    for spine in ax.spines.values():
        spine.set_linewidth(0.8)
        spine.set_color('#333333')

    fig.tight_layout()

    # Save
    if args.output_path:
        output_path = args.output_path
    else:
        output_path = Path(f"paper_visualizations/outputs/freq_magnitude_v2/reconstruct_l{layer}_h{head}.png")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path)
    plt.close(fig)

    print(f"Saved to {output_path}")

    # Print some statistics
    print(f"\nStatistics:")
    print(f"  Token count: {token_count}")
    print(f"  Num frequencies: {num_freq}")
    print(f"  Top 5 amplitudes: {amplitude.topk(5).values.cpu().numpy()}")
    print(f"  Top 5 amplitude freq indices: {amplitude.topk(5).indices.cpu().numpy()}")


if __name__ == "__main__":
    main()
