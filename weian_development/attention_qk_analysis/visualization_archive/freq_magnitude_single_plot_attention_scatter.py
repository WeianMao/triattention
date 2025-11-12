"""Scatter visualization of centered Q/K offsets for frequencies ranked by attention contribution."""
from __future__ import annotations

import argparse
import json
import math
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
from weian_development.attention_qk_analysis.freq_magnitude_plots import invert_rope


PAIR_CHUNK_SIZE = 65536


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scatter top-frequency centered Q/K offsets for a single head")
    parser.add_argument(
        "trace_dir",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        nargs="?",
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index to visualize")
    parser.add_argument("--head", type=int, default=0, help="Head index to visualize")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE parameters",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device (cuda:N or cpu)")
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
        "--no-center",
        action="store_true",
        help="Plot raw complex values without subtracting their mean",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/layer_00_head_00_freq_attn.png"),
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
    if tensor.size(-1) % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    freq_count = tensor.shape[1] // 2
    real_dtype = torch.float32 if tensor.dtype in (torch.bfloat16, torch.float16) else tensor.dtype
    tensor_real = tensor.to(dtype=real_dtype)
    real = tensor_real[:, :freq_count].contiguous()
    imag = tensor_real[:, freq_count:].contiguous()
    return torch.complex(real, imag)


def mean_vector_product(
    q_unrot: torch.Tensor,
    k_unrot: torch.Tensor,
) -> torch.Tensor:
    q_complex = to_complex_pairs(q_unrot)
    k_complex = to_complex_pairs(k_unrot)

    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    q_abs = torch.abs(q_mean)
    k_abs = torch.abs(k_mean)
    amp_product = q_abs * k_abs

    return amp_product


def compute_attention_frequency_scores(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    scale: float,
    chunk_size: int = PAIR_CHUNK_SIZE,
) -> torch.Tensor:
    seq_len, head_dim = q_block.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even to form RoPE frequency pairs")
    freq_count = head_dim // 2
    if seq_len == 0 or freq_count == 0:
        return torch.zeros(freq_count, device=q_block.device, dtype=torch.float32)

    q_pairs = q_block.view(seq_len, freq_count, 2).to(dtype=torch.float32)
    k_pairs = k_block.view(seq_len, freq_count, 2).to(dtype=torch.float32)

    row_idx, col_idx = torch.tril_indices(seq_len, seq_len, device=q_block.device)
    total_pairs = row_idx.numel()
    if total_pairs == 0:
        return torch.zeros(freq_count, device=q_block.device, dtype=torch.float32)

    scores = torch.zeros(freq_count, device=q_block.device, dtype=torch.float64)
    counts = torch.zeros(freq_count, device=q_block.device, dtype=torch.float64)

    for start in range(0, total_pairs, chunk_size):
        end = min(start + chunk_size, total_pairs)
        rows = row_idx[start:end]
        cols = col_idx[start:end]

        q_chunk = q_pairs[rows]
        k_chunk = k_pairs[cols]

        dots = (q_chunk * k_chunk).sum(dim=2) * scale
        finite_mask = torch.isfinite(dots)
        dots = torch.where(finite_mask, dots, torch.zeros_like(dots))

        abs_chunk = dots.abs().to(dtype=torch.float64)
        scores += abs_chunk.sum(dim=0)
        counts += finite_mask.to(dtype=torch.float64).sum(dim=0)

    counts = torch.clamp(counts, min=1.0)
    scores = scores / counts
    return scores.to(dtype=torch.float32)


def plot_scatter_grid(
    q_complex: torch.Tensor,
    k_complex: torch.Tensor,
    q_mean: torch.Tensor,
    k_mean: torch.Tensor,
    top_indices: torch.Tensor,
    amp_product: torch.Tensor,
    out_path: Path,
    dpi: int,
    figsize: Optional[Tuple[float, float]],
    centered: bool,
) -> None:
    freq_count = top_indices.numel()
    cols = 2  # left: Q, right: K
    rows = max(1, freq_count)
    if figsize is None:
        base = 3.2
        figsize = (cols * base, rows * base)
    fig, axes = plt.subplots(
        rows,
        cols,
        figsize=figsize,
        dpi=dpi,
        squeeze=False,
        subplot_kw={"box_aspect": 1},
    )

    for row, f_idx in enumerate(top_indices.tolist()):
        q_vals = q_complex[:, f_idx] - q_mean[f_idx] if centered else q_complex[:, f_idx]
        k_vals = k_complex[:, f_idx] - k_mean[f_idx] if centered else k_complex[:, f_idx]

        q_ax = axes[row, 0]
        k_ax = axes[row, 1]

        q_real = q_vals.real.detach().cpu().numpy()
        q_imag = q_vals.imag.detach().cpu().numpy()
        k_real = k_vals.real.detach().cpu().numpy()
        k_imag = k_vals.imag.detach().cpu().numpy()

        q_ax.scatter(q_real, q_imag, s=12, alpha=0.6, edgecolors="none")
        k_ax.scatter(k_real, k_imag, s=12, alpha=0.6, edgecolors="none")

        prefix = "Δ" if centered else ""
        for ax, label, real_vals, imag_vals in (
            (q_ax, "q", q_real, q_imag),
            (k_ax, "k", k_real, k_imag),
        ):
            ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
            ax.axvline(0.0, color="gray", linewidth=0.8, alpha=0.7)
            ax.set_xlabel(f"Re({prefix}{label})")
            ax.set_ylabel(f"Im({prefix}{label})")
            ax.grid(alpha=0.2, linestyle="--")
            ax.set_aspect("equal", adjustable="box")
            limit = max(np.max(np.abs(real_vals)), np.max(np.abs(imag_vals)))
            if not np.isfinite(limit) or limit == 0.0:
                limit = 1e-6
            limit *= 1.05
            ax.set_xlim(-limit, limit)
            ax.set_ylim(-limit, limit)

        amp_val = amp_product[f_idx].item()
        suffix = "centered" if centered else "raw"
        q_ax.set_title(f"f={f_idx}  Q {suffix} |E[q]E[k]|≈{amp_val:.3f}")
        k_ax.set_title(f"f={f_idx}  K {suffix} |E[q]E[k]|≈{amp_val:.3f}")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_freq_single")

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
    inv_freq = rotary.inv_freq.to(torch.float64)
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

    q_unrot = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_unrot = invert_rope(k_block, cos_table, sin_table, attention_scale)

    q_complex = to_complex_pairs(q_unrot)
    k_complex = to_complex_pairs(k_unrot)
    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    scale = (head_dim ** -0.5) * attention_scale
    attn_scores = compute_attention_frequency_scores(q_block, k_block, scale)

    amp_product = mean_vector_product(q_unrot, k_unrot)

    top_k = min(args.top_k, attn_scores.numel())
    if top_k < 1:
        raise ValueError("No frequencies available for scatter plot")
    _, top_indices = torch.topk(attn_scores, k=top_k)

    plot_scatter_grid(
        q_complex,
        k_complex,
        q_mean,
        k_mean,
        top_indices,
        amp_product,
        args.output_path,
        args.dpi,
        tuple(args.figsize) if args.figsize is not None else None,
        centered=not args.no_center,
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
