"""Generate per-frequency magnitude diagnostics for captured Q/K tensors."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import math
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from transformers import AutoConfig
from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize frequency-wise magnitudes for Q/K")
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
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for computation (e.g., cuda:0 or cpu)",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Computation dtype",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Figure DPI",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(18.0, 5.0),
        help="Matplotlib figsize in inches",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print progress for each layer/head",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="Optional limit on number of layers processed",
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=None,
        help="Optional limit on number of heads per layer",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=10000,
        help="Maximum token distance for reconstructed curve",
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
    # q_mag, k_mag: [seq_len, num_freq]
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


def plot_frequency_diagnostics(
    freq_values: torch.Tensor,
    titles: list[str],
    periods: torch.Tensor,
    distances: torch.Tensor,
    reconstructed: torch.Tensor,
    out_path: Path,
    dpi: int,
    figsize: tuple[float, float],
) -> None:
    freq_count = freq_values[0].numel()
    freq_axis = periods[:freq_count].cpu().numpy()

    fig, axes = plt.subplots(1, len(freq_values) + 1, figsize=figsize, dpi=dpi, sharex=False)
    axes = list(axes.flatten())

    for ax, values, title in zip(axes[:-1], freq_values, titles):
        ax.plot(freq_axis, values.cpu().numpy(), marker="o", markersize=2, linewidth=1.2)
        ax.set_title(title)
        ax.set_xscale("log")
        ax.set_xlabel("RoPE period (tokens)")
        ax.set_ylabel("Magnitude")
        ax.grid(alpha=0.3, linestyle="--")

    ax_recon = axes[-1]
    ax_recon.plot(distances.cpu().numpy(), reconstructed.cpu().numpy(), linewidth=1.2)
    ax_recon.set_title("Reconstructed avg cos kernel")
    ax_recon.set_xlabel("Token distance Δ")
    ax_recon.set_ylabel("Σ_f |Q||K| cos(ω_f Δ)")
    ax_recon.set_xscale("log")
    ax_recon.grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def write_trace_readme(trace_dir: Path) -> None:
    content = """# 频段幅值诊断

- 图 1：对所有 key token 的 |K| 取平均，展示各频段幅值变化（未取对数）。
- 图 2：对所有 query token 的 |Q| 取平均。
- 图 3：按自回归遮罩聚合 `|Q|_i * |K|_j` 的平均值，其中 i≥j。
- 横轴为 RoPE 频段对应的周期（单位：token），并采用对数坐标便于观察；纵轴为直接幅值大小。
"""
    (trace_dir / "README.md").write_text(content, encoding="utf-8")


def process_trace(
    trace_dir: Path,
    out_root: Path,
    device: torch.device,
    dtype: torch.dtype,
    periods: torch.Tensor,
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

    for layer in range(layer_limit):
        head_idx = 0
        while head_idx < head_limit:
            q_block = q_tensor[layer, head_idx : head_idx + 1].to(device=device, dtype=dtype)[0]
            k_block = k_tensor[layer, head_idx : head_idx + 1].to(device=device, dtype=dtype)[0]

            q_block = q_block[:token_count]
            k_block = k_block[:token_count]

            q_mag = magnitude_pairs(q_block)
            k_mag = magnitude_pairs(k_block)

            mean_k = mean_magnitude(k_mag)
            mean_q = mean_magnitude(q_mag)
            causal_mean = causal_product_average(q_mag, k_mag)

            freq_count = mean_k.numel()
            omega = (periods[:freq_count] ** -1) * (2 * math.pi)
            distance_limit = min(args.max_distance, token_count)
            step_count = min(distance_limit, 1024)
            distance_axis = torch.logspace(
                start=0.0,
                end=math.log10(distance_limit),
                steps=step_count,
                device=device,
                dtype=mean_k.dtype,
            )
            omega = omega.to(device=distance_axis.device, dtype=distance_axis.dtype)
            cos_matrix = torch.cos(distance_axis.unsqueeze(1) * omega.unsqueeze(0))
            reconstructed = cos_matrix @ causal_mean[:freq_count]

            values = [mean_k, mean_q, causal_mean]
            titles = [
                "Avg |K| per frequency",
                "Avg |Q| per frequency",
                "Avg |Q||K| over causal pairs",
            ]

            out_path = trace_out_dir / f"layer_{layer:02d}_head_{head_idx:02d}_freq.png"
            plot_frequency_diagnostics(
                values,
                titles,
                periods,
                distance_axis,
                reconstructed,
                out_path,
                args.dpi,
                tuple(args.figsize),
            )

            if args.verbose:
                rel = out_path.relative_to(out_root)
                print(f"Saved {rel}")

            head_idx += 1

    del q_tensor, k_tensor, data
    if device.type == "cuda":
        torch.cuda.empty_cache()


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_freq")

    trace_dirs = sorted(
        p for p in args.input_root.iterdir() if p.is_dir() and p.name.startswith("qid")
    )
    if not trace_dirs:
        raise SystemExit(f"No trace directories found under {args.input_root}")

    device = torch.device(args.device)
    dtype = select_dtype(args.dtype)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rotary = Qwen3RotaryEmbedding(config=config, device="cpu")
    inv_freq = rotary.inv_freq.to(torch.float64)
    periods = (2 * math.pi) / inv_freq

    args.output_root.mkdir(parents=True, exist_ok=True)

    for trace_dir in trace_dirs:
        if args.verbose:
            print(f"Processing {trace_dir}")
        process_trace(trace_dir, args.output_root, device, dtype, periods, args)


if __name__ == "__main__":
    main()
