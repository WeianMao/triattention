"""Focused visualization for a single layer/head frequency reconstruction."""
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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command
from weian_development.attention_qk_analysis.freq_magnitude_plots import invert_rope


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot frequency distance diagnostics for a single head")
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
    parser.add_argument("--max-distance", type=int, default=10000, help="Maximum distance Δ to consider")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for the saved plot")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(12.0, 4.5),
        help="Matplotlib figsize for the distance plots",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/layer_00_head_00_freq_coarse.png"),
        help="Destination for the generated plot",
    )
    parser.add_argument(
        "--flip-phi",
        action="store_true",
        help="Negate the estimated phase shift for diagnostic comparisons",
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


def mean_vector_product_and_phase(
    q_unrot: torch.Tensor,
    k_unrot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_complex = to_complex_pairs(q_unrot)
    k_complex = to_complex_pairs(k_unrot)

    q_mean = q_complex.mean(dim=0)
    k_mean = k_complex.mean(dim=0)

    q_abs = torch.abs(q_mean)
    k_abs = torch.abs(k_mean)
    amp_product = q_abs * k_abs

    relative = q_mean * torch.conj(k_mean)
    phi_hat = torch.atan2(relative.imag, relative.real)

    return amp_product, phi_hat


def compute_ground_truth(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    max_distance: int,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    token_count = q_block.shape[0]
    limit = min(max_distance, token_count - 1)
    if limit < 1:
        return (
            torch.arange(1, 2, device=device, dtype=dtype),
            torch.zeros(1, device=device, dtype=dtype),
        )

    fft_len = 1 << (2 * token_count - 1).bit_length()
    q_fft = torch.fft.rfft(q_block, n=fft_len, dim=0)
    k_fft = torch.fft.rfft(k_block, n=fft_len, dim=0)
    prod = torch.conj(k_fft) * q_fft
    corr = torch.fft.irfft(prod.sum(dim=1), n=fft_len, dim=0)
    corr = corr[: limit + 1]

    distances = torch.arange(1, limit + 1, device=device, dtype=dtype)
    counts = (token_count - torch.arange(0, limit + 1, device=device, dtype=dtype)).clamp_min(1.0)
    scores = corr[1 : limit + 1] / counts[1 : limit + 1]
    return distances, scores


def plot_distance_curves(
    distance_vals: torch.Tensor,
    reconstructed_plain: torch.Tensor,
    reconstructed_weighted: torch.Tensor,
    gt_distances: torch.Tensor,
    gt_scores: torch.Tensor,
    out_path: Path,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharex=False)
    axes = axes.flatten().tolist() if hasattr(axes, "flatten") else list(axes)

    ax_plain, ax_weighted, ax_gt = axes

    x_plain = distance_vals.detach().cpu().numpy()
    plain_vals = reconstructed_plain.detach().cpu().numpy()
    ax_plain.plot(x_plain, plain_vals, color="#1f77b4", linewidth=1.4)
    ax_plain.set_title("Σ |Q||K| cos(ωΔ)")
    ax_plain.set_xlabel("Δ (tokens)")
    ax_plain.set_ylabel("Value")
    ax_plain.set_xscale("log")
    ax_plain.grid(alpha=0.3, linestyle="--")

    weighted_vals = reconstructed_weighted.detach().cpu().numpy()
    ax_weighted.plot(x_plain, weighted_vals, color="#ff7f0e", linewidth=1.4)
    ax_weighted.set_title("Σ |Q||K| cos(ωΔ + φ̂)")
    ax_weighted.set_xlabel("Δ (tokens)")
    ax_weighted.set_xscale("log")
    ax_weighted.grid(alpha=0.3, linestyle="--")

    x_gt = gt_distances.detach().cpu().numpy()
    gt_vals = gt_scores.detach().cpu().numpy()
    ax_gt.plot(x_gt, gt_vals, color="#2ca02c", linewidth=1.4)
    ax_gt.set_title("Ground-truth avg Q·K")
    ax_gt.set_xlabel("Δ (tokens)")
    ax_gt.set_xscale("log")
    ax_gt.grid(alpha=0.3, linestyle="--")

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
    periods = ((2 * math.pi) / inv_freq).to(device=device, dtype=dtype)
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
    freq_count = head_dim // 2
    omega = (periods[:freq_count] ** -1) * (2 * math.pi)

    seq_len = q_block.shape[0]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=q_block.dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=q_block.dtype)
    sin_table = sin_table[0].to(dtype=q_block.dtype)

    q_unrot = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_unrot = invert_rope(k_block, cos_table, sin_table, attention_scale)

    amp_product, phi_hat = mean_vector_product_and_phase(q_unrot, k_unrot)
    amp_product = amp_product.to(device=device, dtype=dtype)
    phi_hat = phi_hat.to(device=device, dtype=dtype)
    if args.flip_phi:
        phi_hat = -phi_hat

    distance_limit = min(args.max_distance, q_block.shape[0])
    if distance_limit < 1:
        distance_vals = torch.ones(1, device=device, dtype=dtype)
    else:
        # Sample sparsely in log-space (base 2) to reduce curve resolution, e.g., 1, 2, 4, 8, …
        log_steps: list[int] = []
        value = 1
        while value <= distance_limit:
            log_steps.append(value)
            value *= 2
        if log_steps[-1] != distance_limit:
            log_steps.append(distance_limit)
        distance_vals = torch.tensor(log_steps, device=device, dtype=dtype)

    cos_terms = torch.cos(distance_vals.unsqueeze(1) * omega.unsqueeze(0))
    reconstructed_plain = cos_terms @ amp_product[:freq_count]

    cos_terms_phi = torch.cos(distance_vals.unsqueeze(1) * omega.unsqueeze(0) + phi_hat[:freq_count].unsqueeze(0))
    reconstructed_phi = cos_terms_phi @ amp_product[:freq_count]

    gt_distances, gt_scores = compute_ground_truth(q_block, k_block, args.max_distance, device, dtype)

    plot_distance_curves(
        distance_vals,
        reconstructed_plain,
        reconstructed_phi,
        gt_distances,
        gt_scores,
        args.output_path,
        args.dpi,
        tuple(args.figsize),
    )

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
