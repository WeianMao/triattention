"""Scatter visualization comparing K distribution before/after NMS drop."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command
from weian_development.attention_qk_analysis.visualization_archive.freq_magnitude_plots import (
    invert_rope,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scatter plot comparing K before/after NMS drop"
    )
    parser.add_argument(
        "trace_dir",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        nargs="?",
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--layer", type=int, default=3, help="Layer index to visualize")
    parser.add_argument("--head", type=int, default=5, help="Head index to visualize")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Computation dtype",
    )
    parser.add_argument("--top-k", type=int, default=3, help="Number of frequencies to scatter")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for the saved plot")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=None,
        help="Optional matplotlib figsize override (width height)",
    )
    parser.add_argument(
        "--energy-method",
        choices=["amplitude", "causal", "meanvec"],
        default="meanvec",
        help="Method for computing frequency band energy weights",
    )
    parser.add_argument(
        "--sample-keys",
        type=int,
        default=2048,
        help="Number of keys to sample for NMS visualization (to avoid OOM)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for key sampling",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/vis/nms_comparison.png"),
        help="Destination for the generated plot",
    )
    return parser.parse_args()


def select_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [N, head_dim] tensor to [N, freq_count] complex tensor."""
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    freq_count = tensor.shape[-1] // 2
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    real = tensor_real[..., :freq_count].contiguous()
    imag = tensor_real[..., freq_count:].contiguous()
    return torch.complex(real, imag)


def mean_vector_product(q_unrot: torch.Tensor, k_unrot: torch.Tensor) -> torch.Tensor:
    """Compute |E[q]| * |E[k]| per frequency."""
    q_complex = to_complex_pairs(q_unrot)
    k_complex = to_complex_pairs(k_unrot)
    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)
    return torch.abs(q_mean) * torch.abs(k_mean)


def compute_frequency_energy_weights(
    q_complex: torch.Tensor,
    k_complex: torch.Tensor,
    method: str,
    device: torch.device,
) -> torch.Tensor:
    """Compute frequency band energy weights for spectrum-aware NMS."""
    q_complex = q_complex.to(device=device)
    k_complex = k_complex.to(device=device)

    if method == "amplitude":
        q_abs = torch.abs(q_complex)
        k_abs = torch.abs(k_complex)
        energy_per_freq = (q_abs * k_abs).mean(dim=0)
    elif method == "meanvec":
        q_mean = q_complex.mean(dim=0)
        k_mean = k_complex.mean(dim=0)
        energy_per_freq = torch.abs(q_mean) * torch.abs(k_mean)
    elif method == "causal":
        seq_len = q_complex.shape[0]
        freq_count = q_complex.shape[1]
        k_conj = k_complex.conj()
        energy_per_freq = torch.zeros(freq_count, device=device, dtype=torch.float32)
        chunk_size = min(256, seq_len)
        for i_start in range(0, seq_len, chunk_size):
            i_end = min(i_start + chunk_size, seq_len)
            q_chunk = q_complex[i_start:i_end]
            for i_local, i_global in enumerate(range(i_start, i_end)):
                q_i = q_chunk[i_local]
                k_causal = k_conj[:i_global + 1]
                attn_contrib = (q_i.unsqueeze(0) * k_causal).real.sum(dim=0)
                energy_per_freq += attn_contrib
        num_valid_pairs = seq_len * (seq_len + 1) / 2
        energy_per_freq = energy_per_freq / num_valid_pairs
    else:
        raise ValueError(f"Unknown energy method: {method}")

    energy_per_freq = energy_per_freq.to(dtype=torch.float32)
    total = energy_per_freq.sum()
    if torch.abs(total) > 1e-8:
        freq_weights = energy_per_freq / total
    else:
        freq_weights = torch.ones_like(energy_per_freq) / energy_per_freq.numel()
    return freq_weights


def fast_parallel_nms(
    k_complex: torch.Tensor,
    freq_weights: torch.Tensor,
) -> torch.Tensor:
    """Fast Parallel NMS based on projection coverage.

    Returns:
        keep_mask: [N] bool tensor, True = keep
    """
    N, F = k_complex.shape
    device = k_complex.device

    k_abs = torch.abs(k_complex)
    k_abs_safe = k_abs.clamp(min=1e-5)

    real_dot = torch.einsum("af,bf->abf", k_complex, k_complex.conj()).real
    proj_on_b = real_dot / k_abs_safe.unsqueeze(0)
    per_freq_score = proj_on_b - k_abs.unsqueeze(0)
    coverage_score = (per_freq_score * freq_weights.view(1, 1, -1)).sum(dim=2)

    suppresses = coverage_score > 0
    suppresses.fill_diagonal_(False)
    is_suppressed = suppresses.any(dim=0)
    keep_mask = ~is_suppressed

    return keep_mask


def plot_nms_comparison(
    k_complex_rot: torch.Tensor,
    keep_mask: torch.Tensor,
    top_indices: torch.Tensor,
    amp_product: torch.Tensor,
    freq_weights: torch.Tensor,
    energy_method: str,
    out_path: Path,
    dpi: int,
    figsize: Optional[Tuple[float, float]],
) -> dict:
    """Plot K scatter in rotated space: all points, with dropped (red) and kept (blue) markers."""
    freq_count = top_indices.numel()
    cols = 3  # Before NMS | After NMS (kept only) | Overlay
    rows = max(1, freq_count)
    if figsize is None:
        base = 3.5
        figsize = (cols * base, rows * base)

    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        subplot_kw={"box_aspect": 1},
    )

    keep_count = keep_mask.sum().item()
    drop_count = (~keep_mask).sum().item()
    total_count = keep_mask.numel()

    stats = {
        "total_keys": total_count,
        "kept_keys": keep_count,
        "dropped_keys": drop_count,
        "drop_rate": drop_count / total_count,
        "energy_method": energy_method,
        "per_freq": [],
    }

    for row, f_idx in enumerate(top_indices.tolist()):
        # Use rotated K for visualization (same space as NMS)
        k_vals = k_complex_rot[:, f_idx]
        k_real = k_vals.real.detach().cpu().numpy()
        k_imag = k_vals.imag.detach().cpu().numpy()

        keep_np = keep_mask.cpu().numpy()
        drop_np = ~keep_mask.cpu().numpy()

        # Column 0: Before NMS (all K)
        ax0 = axes[row, 0]
        ax0.scatter(k_real, k_imag, s=8, alpha=0.5, c="gray", edgecolors="none", label="All K")

        # Column 1: After NMS (kept only)
        ax1 = axes[row, 1]
        ax1.scatter(k_real[keep_np], k_imag[keep_np], s=10, alpha=0.6, c="tab:blue", edgecolors="none", label="Kept")

        # Column 2: Overlay - dropped (red) vs kept (blue)
        ax2 = axes[row, 2]
        ax2.scatter(k_real[drop_np], k_imag[drop_np], s=8, alpha=0.4, c="tab:red", edgecolors="none", label="Dropped", zorder=1)
        ax2.scatter(k_real[keep_np], k_imag[keep_np], s=10, alpha=0.6, c="tab:blue", edgecolors="none", label="Kept", zorder=2)

        # Common formatting
        amp_val = amp_product[f_idx].item()
        weight_val = freq_weights[f_idx].item()

        for col, (ax, title_suffix) in enumerate([
            (ax0, "Before NMS"),
            (ax1, f"After NMS (kept={keep_count})"),
            (ax2, "Overlay"),
        ]):
            ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
            ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.7)
            ax.set_xlabel(r"Re($k^{rot}$)")
            ax.set_ylabel(r"Im($k^{rot}$)")
            ax.grid(alpha=0.2, linestyle="--")
            ax.set_aspect("equal", adjustable="box")

            limit = max(np.max(np.abs(k_real)), np.max(np.abs(k_imag)))
            if not np.isfinite(limit) or limit == 0.0:
                limit = 1e-6
            limit *= 1.05
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)

            ax.set_title(f"f={f_idx} w={weight_val:.3f} | {title_suffix}", fontsize=9)
            if col == 2:
                ax.legend(fontsize=7, loc="upper right")

        stats["per_freq"].append({
            "freq_idx": f_idx,
            "amp_product": amp_val,
            "weight": weight_val,
        })

    fig.suptitle(
        f"NMS Comparison (Rotated Space) | Method: {energy_method} | "
        f"Drop: {drop_count}/{total_count} ({100*drop_count/total_count:.1f}%)",
        fontsize=11,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)

    return stats


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_nms_viz")

    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists():
        raise FileNotFoundError(f"Missing qk.pt under {args.trace_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json under {args.trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta.get("sequence_length", 0))

    device = torch.device(args.device)
    dtype = select_dtype(args.dtype)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling
    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]
    k_tensor: torch.Tensor = data["k"]

    if args.layer >= q_tensor.shape[0] or args.head >= q_tensor.shape[1]:
        raise IndexError("Requested layer/head exceeds captured tensor dimensions")

    q_block = q_tensor[args.layer, args.head].to(device=device, dtype=dtype)
    k_block = k_tensor[args.layer, args.head].to(device=device, dtype=dtype)
    if token_count:
        q_block = q_block[:token_count]
        k_block = k_block[:token_count]

    head_dim = q_block.shape[-1]
    seq_len = q_block.shape[0]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=q_block.dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=q_block.dtype)
    sin_table = sin_table[0].to(dtype=q_block.dtype)

    # Get unrotated Q/K for frequency ranking (meanvec uses unrotated)
    q_unrot = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_unrot = invert_rope(k_block, cos_table, sin_table, attention_scale)

    q_complex_unrot = to_complex_pairs(q_unrot)
    k_complex_unrot = to_complex_pairs(k_unrot)

    # Rotated K for NMS and visualization (original k_block)
    q_complex_rot = to_complex_pairs(q_block)
    k_complex_rot = to_complex_pairs(k_block)

    # Compute frequency weights based on energy method
    if args.energy_method in ("amplitude", "meanvec"):
        # amplitude/meanvec use unrotated Q/K for weight computation
        freq_weights = compute_frequency_energy_weights(
            q_complex_unrot, k_complex_unrot, args.energy_method, device
        )
    else:  # causal uses rotated Q/K
        freq_weights = compute_frequency_energy_weights(
            q_complex_rot, k_complex_rot, "causal", device
        )

    # Use meanvec for frequency ranking (consistent with original scatter script)
    amp_product = mean_vector_product(q_unrot, k_unrot)

    # Sample keys for NMS visualization to avoid OOM
    sample_size = min(args.sample_keys, seq_len)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    perm = torch.randperm(seq_len, generator=generator)[:sample_size]
    perm = perm.sort().values  # Keep temporal order

    k_complex_rot_sample = k_complex_rot[perm]

    print(f"Sampled {sample_size} keys from {seq_len} total for NMS visualization")

    # Apply NMS using rotated K (sampled)
    keep_mask = fast_parallel_nms(k_complex_rot_sample, freq_weights)
    top_k = min(args.top_k, amp_product.numel())
    _, top_indices = torch.topk(amp_product, k=top_k)

    # Plot comparison using sampled rotated K (same space as NMS)
    stats = plot_nms_comparison(
        k_complex_rot_sample,  # Use rotated K for both NMS and visualization
        keep_mask,
        top_indices,
        amp_product,
        freq_weights,
        args.energy_method,
        args.output_path,
        args.dpi,
        tuple(args.figsize) if args.figsize is not None else None,
    )

    print(f"Layer {args.layer}, Head {args.head}")
    print(f"Energy method: {args.energy_method}")
    print(f"Total keys: {stats['total_keys']}")
    print(f"Kept keys: {stats['kept_keys']}")
    print(f"Dropped keys: {stats['dropped_keys']} ({100*stats['drop_rate']:.2f}%)")
    print(f"Saved to: {args.output_path}")

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
