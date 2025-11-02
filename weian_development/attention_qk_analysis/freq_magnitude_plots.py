"""Generate per-frequency magnitude diagnostics (including angle statistics) for captured Q/K tensors."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import torch
import torch.nn.functional as F
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize frequency-wise diagnostics for Q/K")
    parser.add_argument("input_root", type=Path, help="Directory containing qid*_trace*/qk.pt")
    parser.add_argument(
        "--output-root",
        type=Path,
        required=True,
        help="Root directory to store frequency diagnostic plots",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory used to recover RoPE frequency periods",
    )
    parser.add_argument("--device", default="cuda:0", help="Device used for computation (e.g., cuda:0 or cpu)")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Computation dtype",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(24.0, 10.0),
        help="Matplotlib figsize in inches",
    )
    parser.add_argument("--verbose", action="store_true", help="Print progress for each layer/head")
    parser.add_argument("--max-layers", type=int, default=None, help="Optional limit on number of layers processed")
    parser.add_argument("--max-heads", type=int, default=None, help="Optional limit on number of heads per layer")
    parser.add_argument(
        "--max-distance",
        type=int,
        default=10000,
        help="Maximum token distance (Δ) for reconstructed kernels",
    )
    return parser.parse_args()


def select_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def magnitude_pairs(tensor: torch.Tensor) -> torch.Tensor:
    seq_len, head_dim = tensor.shape
    if head_dim % 2 != 0:
        raise ValueError(f"Head dimension {head_dim} is not even; cannot form (x, y) pairs")
    num_freq = head_dim // 2
    reshaped = tensor.view(seq_len, num_freq, 2)
    return torch.linalg.norm(reshaped, dim=-1)  # [seq_len, num_freq]


def mean_magnitude(mags: torch.Tensor) -> torch.Tensor:
    return mags.mean(dim=0)


def causal_product_average(q_mag: torch.Tensor, k_mag: torch.Tensor) -> torch.Tensor:
    seq_len = q_mag.shape[0]
    if k_mag.shape[0] != seq_len or k_mag.shape[1] != q_mag.shape[1]:
        raise ValueError("q_mag and k_mag must share dimensions")

    q64 = q_mag.to(torch.float64)
    k64 = k_mag.to(torch.float64)
    prefix_k = torch.cumsum(k64, dim=0)
    cumulative = q64 * prefix_k
    numerator = cumulative.sum(dim=0)
    total_pairs = seq_len * (seq_len + 1) / 2.0
    result = numerator / total_pairs
    return result.to(q_mag.dtype)


def rotate_half(x: torch.Tensor) -> torch.Tensor:
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
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    scale_t = torch.tensor(scale, device=rotated.device, dtype=rotated.dtype)
    base = rotated / scale_t
    cos_unit = cos / scale_t
    sin_unit = sin / scale_t
    return base * cos_unit - rotate_half(base) * sin_unit


def angle_statistics(q_pairs: torch.Tensor, k_pairs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    eps = 1e-8
    q_norm = q_pairs.norm(dim=-1, keepdim=True).clamp_min(eps)
    k_norm = k_pairs.norm(dim=-1, keepdim=True).clamp_min(eps)
    q_unit = q_pairs / q_norm
    k_unit = k_pairs / k_norm

    qx = q_unit[..., 0]
    qy = q_unit[..., 1]
    kx = k_unit[..., 0]
    ky = k_unit[..., 1]

    prefix_kx = torch.cumsum(kx, dim=0)
    prefix_ky = torch.cumsum(ky, dim=0)

    cos_total = (qx * prefix_kx + qy * prefix_ky).sum(dim=0)
    sin_total = (qx * prefix_ky - qy * prefix_kx).sum(dim=0)

    total_pairs = q_pairs.size(0) * (q_pairs.size(0) + 1) / 2.0
    mean_cos = cos_total / total_pairs
    mean_sin = sin_total / total_pairs

    mean_angles = torch.atan2(mean_sin, mean_cos)
    R = torch.sqrt(mean_cos ** 2 + mean_sin ** 2).clamp_max(1.0)
    circular_variance = 1 - R

    return mean_angles, circular_variance


def plot_frequency_diagnostics(
    freq_values: torch.Tensor,
    titles: list[str],
    periods: torch.Tensor,
    distances: torch.Tensor,
    reconstructed_plain: torch.Tensor,
    reconstructed_phased: torch.Tensor,
    out_path: Path,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    freq_count = freq_values[0].numel()
    freq_axis = periods[:freq_count].cpu().numpy()

    total_plots = len(freq_values) + 2
    cols = 4
    rows = (total_plots + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=figsize, dpi=dpi, sharex=False)
    axes = axes.flatten().tolist() if hasattr(axes, "flatten") else [axes]
    axes = axes[:total_plots]

    for ax, values, title in zip(axes[: len(freq_values)], freq_values, titles):
        ax.plot(freq_axis, values.cpu().numpy(), marker="o", markersize=2, linewidth=1.2)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xlabel("RoPE period (tokens)")
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.3, linestyle="--")

    ax_plain = axes[-2]
    ax_plain.plot(distances.cpu().numpy(), reconstructed_plain.cpu().numpy(), linewidth=1.2)
    ax_plain.set_title("Σ_f |Q||K| cos(ω_f Δ)")
    ax_plain.set_xlabel("Token distance Δ")
    ax_plain.set_ylabel("Value")
    ax_plain.set_xscale("log")
    ax_plain.grid(alpha=0.3, linestyle="--")

    ax_phase = axes[-1]
    ax_phase.plot(distances.cpu().numpy(), reconstructed_phased.cpu().numpy(), linewidth=1.2)
    ax_phase.set_title("Σ_f |Q||K| cos(ω_f Δ + φ_f)")
    ax_phase.set_xlabel("Token distance Δ")
    ax_phase.set_ylabel("Value")
    ax_phase.set_xscale("log")
    ax_phase.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_trace_readme(trace_dir: Path) -> None:
    content = """# 频段幅值诊断

- 图 1：对所有 key token 的 |K| 取平均，展示各频段幅值变化（未取对数）。
- 图 2：对所有 query token 的 |Q| 取平均。
- 图 3：按自回归遮罩聚合 `|Q|_i * |K|_j` 的平均值，其中 i≥j。
- 图 4：每个频段的平均夹角 φ_f（仅考虑 i≥j）。
- 图 5：对应夹角的方差 σ²_f。
- 图 6：未考虑夹角偏移的 Σ_f |Q||K| cos(ω_f Δ) 曲线。
- 图 7：考虑平均夹角偏移的 Σ_f |Q||K| cos(ω_f Δ + φ_f) 曲线。
- 前五幅横轴为 RoPE 周期（token），后两幅横轴为 token 距离 Δ（log 刻度）。
"""
    (trace_dir / "README.md").write_text(content, encoding="utf-8")


def process_trace(
    trace_dir: Path,
    out_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    periods: torch.Tensor,
    rotary: Qwen3RotaryEmbedding,
    attention_scale: float,
    args: argparse.Namespace,
) -> None:
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        return

    trace_out_dir = out_root / trace_dir.name
    trace_out_dir.mkdir(parents=True, exist_ok=True)

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]  # [L, H, T, D]
    k_tensor: torch.Tensor = data["k"]
    num_layers, num_heads, _, head_dim = q_tensor.shape

    layer_limit = min(args.max_layers or num_layers, num_layers)
    head_limit = min(args.max_heads or num_heads, num_heads)

    write_trace_readme(trace_out_dir)

    position_ids = torch.arange(token_count, device=device).unsqueeze(0)
    base = torch.zeros(1, token_count, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0]
    sin_table = sin_table[0]

    periods = periods.to(device=device, dtype=dtype)

    for layer in range(layer_limit):
        for head in range(head_limit):
            q_block = q_tensor[layer, head].to(device=device, dtype=dtype)[:token_count]
            k_block = k_tensor[layer, head].to(device=device, dtype=dtype)[:token_count]

            q_mag = magnitude_pairs(q_block)
            k_mag = magnitude_pairs(k_block)

            mean_k = mean_magnitude(k_mag)
            mean_q = mean_magnitude(q_mag)
            causal_mean = causal_product_average(q_mag, k_mag)

            # Restore pre-RoPE Q/K to compute angles
            q_orig = invert_rope(q_block, cos_table, sin_table, attention_scale)
            k_orig = invert_rope(k_block, cos_table, sin_table, attention_scale)
            q_pairs = q_orig.view(token_count, head_dim // 2, 2)
            k_pairs = k_orig.view(token_count, head_dim // 2, 2)

            mean_angles, var_angles = angle_statistics(q_pairs, k_pairs)

            freq_count = mean_k.numel()
            omega = (periods[:freq_count] ** -1) * (2 * math.pi)
            distance_limit = min(args.max_distance, token_count)
            steps = min(distance_limit, 1024)
            distance_vals = torch.logspace(
                0.0,
                math.log10(float(distance_limit)),
                steps=max(2, steps),
                device=device,
                dtype=dtype,
            )
            omega = omega.to(device=device, dtype=dtype)
            cos_matrix = torch.cos(distance_vals.unsqueeze(1) * omega.unsqueeze(0))
            reconstructed_plain = cos_matrix @ causal_mean[:freq_count]
            phase_shift = mean_angles[:freq_count]
            cos_matrix_phase = torch.cos(distance_vals.unsqueeze(1) * omega.unsqueeze(0) + phase_shift.unsqueeze(0))
            reconstructed_phase = cos_matrix_phase @ causal_mean[:freq_count]

            freq_values = [mean_k, mean_q, causal_mean, mean_angles, var_angles]
            titles = [
                "Avg |K| per frequency",
                "Avg |Q| per frequency",
                "Avg |Q||K| over causal pairs",
                "Mean angle φ_f",
                "Angle variance σ²_f",
            ]

            out_path = trace_out_dir / f"layer_{layer:02d}_head_{head:02d}_freq.png"
            plot_frequency_diagnostics(
                freq_values,
                titles,
                periods,
                distance_vals,
                reconstructed_plain,
                reconstructed_phase,
                out_path,
                args.dpi,
                tuple(args.figsize),
            )

            if args.verbose:
                rel = out_path.relative_to(out_root)
                print(f"Saved {rel}")

    del q_tensor, k_tensor, data
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_freq")

    trace_dirs = sorted(p for p in args.input_root.iterdir() if p.is_dir() and p.name.startswith("qid"))
    if not trace_dirs:
        raise SystemExit(f"No trace directories found under {args.input_root}")

    device = torch.device(args.device)
    dtype = select_dtype(args.dtype)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    inv_freq = rotary.inv_freq.to(torch.float64)
    periods = ((2 * math.pi) / inv_freq).to(device=device, dtype=dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    args.output_root.mkdir(parents=True, exist_ok=True)

    for trace_dir in trace_dirs:
        if args.verbose:
            print(f"Processing {trace_dir}")
        process_trace(
            trace_dir,
            args.output_root,
            device,
            dtype,
            periods,
            rotary,
            attention_scale,
            args,
        )


if __name__ == "__main__":
    main()
