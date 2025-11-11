"""Frequency aggregation with per-frequency DFT computed from complex Q/K pairs."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot frequency distance diagnostics via normalized series aggregation")
    parser.add_argument(
        "trace_dir",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        nargs="?",
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index to visualize")
    parser.add_argument("--head", type=int, default=0, help="Head index to visualize")
    parser.add_argument("--device", default="cuda:0", help="Computation device (cuda:N or cpu)")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE frequencies",
    )
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
        default=Path("outputs/deepseek_r1_qwen3_8b/layer_00_head_00_freq_series_freqwise.png"),
        help="Destination for the generated plot",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=100,
        help="Ignore queries with fewer historical keys",
    )
    parser.add_argument(
        "--fft-window",
        type=int,
        default=None,
        help="Maximum historical positions used in the RoPE transform (default: full length)",
    )
    parser.add_argument(
        "--energy-eps",
        type=float,
        default=1e-6,
        help="Energy threshold used when normalizing per-query FFT coefficients",
    )
    return parser.parse_args()


def select_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def collect_query_scores(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    window: int,
) -> torch.Tensor:
    if q_block.size(0) <= window:
        raise ValueError("Sequence too short for the requested FFT window")

    q_cpu = q_block.to("cpu", non_blocking=True)
    k_cpu = k_block.to("cpu", non_blocking=True)

    k_windows = k_cpu.unfold(0, window, 1)[:-1]
    history = k_windows.permute(0, 2, 1).flip(dims=(1,))
    q_slice = q_cpu[window:]

    scores = torch.einsum("nwd,nd->nw", history, q_slice)
    return scores


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
    distances: torch.Tensor,
    direct_curve: torch.Tensor,
    mean_scores: torch.Tensor,
    gt_distances: torch.Tensor,
    gt_scores: torch.Tensor,
    out_path: Path,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=figsize, dpi=dpi, sharex=False)
    axes = axes.flatten().tolist() if hasattr(axes, "flatten") else list(axes)

    ax_direct, ax_mean, ax_gt = axes

    x_vals = distances.detach().cpu().numpy()
    direct_vals = direct_curve.detach().cpu().numpy()
    ax_direct.plot(x_vals, direct_vals, color="#1f77b4", linewidth=1.4)
    ax_direct.set_title("RoPE-frequency reconstruction")
    ax_direct.set_xlabel("Δ (tokens)")
    ax_direct.set_ylabel("Value")
    ax_direct.set_xscale("log")
    ax_direct.grid(alpha=0.3, linestyle="--")

    mean_vals = mean_scores.detach().cpu().numpy()
    ax_mean.plot(x_vals, mean_vals, color="#ff7f0e", linewidth=1.4)
    ax_mean.set_title("Average raw Q·K (per Δ)")
    ax_mean.set_xlabel("Δ (tokens)")
    ax_mean.set_xscale("log")
    ax_mean.grid(alpha=0.3, linestyle="--")

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
    mask_process_command("PD-L1_binder_freq_series_freqwise")

    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists():
        raise FileNotFoundError(f"Missing qk.pt under {args.trace_dir}")
    if not meta_path.exists():
        raise FileNotFoundError(f"Missing metadata.json under {args.trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta.get("sequence_length", 0))

    compute_device = torch.device(args.device)
    dtype = select_dtype(args.dtype)
    work_dtype = torch.float32

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]
    k_tensor: torch.Tensor = data["k"]

    if args.layer >= q_tensor.shape[0] or args.head >= q_tensor.shape[1]:
        raise IndexError("Requested layer/head exceeds captured tensor dimensions")

    q_cpu = q_tensor[args.layer, args.head].to(dtype=work_dtype, device="cpu")
    k_cpu = k_tensor[args.layer, args.head].to(dtype=work_dtype, device="cpu")
    if token_count:
        q_cpu = q_cpu[:token_count]
        k_cpu = k_cpu[:token_count]

    seq_len = q_cpu.size(0)
    max_window = seq_len - 1
    window = args.fft_window or max_window
    window = max(args.min_history, min(window, max_window))

    if seq_len <= window:
        raise ValueError("Sequence too short for the requested FFT window")

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rotary = Qwen3RotaryEmbedding(config=config, device="cpu")
    inv_freq = rotary.inv_freq.to(device=compute_device, dtype=work_dtype)

    q_device = q_cpu.to(device=compute_device)
    k_device = k_cpu.to(device=compute_device)

    freq_count = min(inv_freq.numel(), q_device.shape[-1] // 2)
    omega = inv_freq[:freq_count]

    q_complex = torch.complex(q_device[:, :freq_count], q_device[:, freq_count:])
    k_complex = torch.complex(k_device[:, :freq_count], k_device[:, freq_count:])

    deltas = torch.arange(1, window + 1, device=compute_device, dtype=work_dtype)
    phase = deltas.unsqueeze(1) * omega.unsqueeze(0)
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)
    exp_pos = torch.complex(cos_phase, sin_phase)

    cos_kernel = cos_phase.transpose(0, 1).flip(dims=(1,)).unsqueeze(1)
    sin_kernel = sin_phase.transpose(0, 1).flip(dims=(1,)).unsqueeze(1)

    k_real = k_complex.real.transpose(0, 1).unsqueeze(0)
    k_imag = k_complex.imag.transpose(0, 1).unsqueeze(0)

    conv_real_cos = F.conv1d(k_real, cos_kernel, groups=freq_count)
    conv_imag_sin = F.conv1d(k_imag, sin_kernel, groups=freq_count)
    conv_real_sin = F.conv1d(k_real, sin_kernel, groups=freq_count)
    conv_imag_cos = F.conv1d(k_imag, cos_kernel, groups=freq_count)

    s_real = conv_real_cos - conv_imag_sin
    s_imag = conv_real_sin + conv_imag_cos

    s_real = s_real[0, :, :-1].transpose(0, 1)
    s_imag = s_imag[0, :, :-1].transpose(0, 1)
    weighted_keys = torch.complex(s_real, s_imag)

    sum_exp = torch.sum(exp_pos, dim=0)

    prefix_real = torch.zeros((seq_len + 1, freq_count), device=compute_device, dtype=work_dtype)
    prefix_imag = torch.zeros_like(prefix_real)
    prefix_real[1:] = torch.cumsum(k_complex.real, dim=0)
    prefix_imag[1:] = torch.cumsum(k_complex.imag, dim=0)

    sum_real = prefix_real[window:seq_len] - prefix_real[: seq_len - window]
    sum_imag = prefix_imag[window:seq_len] - prefix_imag[: seq_len - window]
    mean_keys = torch.complex(sum_real, sum_imag) / window

    fft_centered = weighted_keys - mean_keys * sum_exp.unsqueeze(0)

    q_slice = q_complex[window:]
    coeffs = q_slice * torch.conj(fft_centered) / window

    scores = collect_query_scores(q_cpu, k_cpu, window)
    if scores.numel() == 0:
        raise ValueError("No queries satisfy the history requirements")

    scores_device = scores.to(device=compute_device, dtype=work_dtype)
    query_means = scores_device.mean(dim=1, keepdim=True)
    scores_centered = scores_device - query_means

    energies = scores_centered.abs().sum(dim=1)
    keep_mask = energies > args.energy_eps
    if not keep_mask.any():
        raise ValueError("All queries filtered out by energy threshold; consider lowering --energy-eps")

    coeffs = coeffs[keep_mask]
    energies_kept = energies[keep_mask]
    query_means_kept = query_means[keep_mask]

    coeffs_norm = coeffs / energies_kept.unsqueeze(1).to(coeffs.dtype)
    avg_coeffs = coeffs_norm.mean(dim=0)
    avg_energy = energies_kept.mean().to(coeffs.dtype)
    overall_mean = query_means_kept.mean()

    direct_centered = torch.matmul(exp_pos, avg_coeffs * avg_energy).real
    direct_curve = direct_centered + overall_mean

    mean_scores = scores.mean(dim=0)

    distances = torch.arange(1, window + 1, device=compute_device, dtype=work_dtype)

    q_gt = q_cpu.to(dtype=work_dtype)
    k_gt = k_cpu.to(dtype=work_dtype)
    gt_distances, gt_scores = compute_ground_truth(q_gt, k_gt, args.max_distance, torch.device("cpu"), work_dtype)
    if gt_distances.numel() > window:
        gt_distances = gt_distances[:window]
        gt_scores = gt_scores[:window]

    plot_distance_curves(
        distances.to("cpu", dtype=dtype),
        direct_curve.to("cpu", dtype=dtype),
        mean_scores.to(dtype=dtype),
        gt_distances.to(dtype=dtype),
        gt_scores.to(dtype=dtype),
        args.output_path,
        args.dpi,
        tuple(args.figsize),
    )

    if compute_device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
