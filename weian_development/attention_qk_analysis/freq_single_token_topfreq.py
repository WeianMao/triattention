"""Visualize raw vs reconstructed contributions for top-magnitude RoPE frequencies on a single query."""
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
from transformers.modeling_rope_utils import _compute_yarn_parameters

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect top-magnitude frequency reconstructions for a single query")
    parser.add_argument(
        "trace_dir",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        nargs="?",
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--head", type=int, default=0, help="Head index")
    parser.add_argument("--token-index", type=int, default=None, help="Query token index; default uses the midpoint")
    parser.add_argument("--min-history", type=int, default=100, help="Minimum history length required")
    parser.add_argument(
        "--fft-window",
        type=int,
        default=None,
        help="Historical window size (default: full history up to token index)",
    )
    parser.add_argument("--top-k", type=int, default=6, help="Number of frequencies to visualize")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE frequencies",
    )
    parser.add_argument("--device", default="cpu", help="Device for minor tensor ops (default cpu)")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Export dtype for plots (kept cpu float32 internally)",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/topfreq_debug.png"),
        help="Destination for the generated figure",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--figsize", type=float, nargs=2, default=(12.0, 8.0), help="Figure size in inches")
    return parser.parse_args()


def select_dtype(name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[name]


def choose_token(token_count: int, min_history: int, requested: int | None) -> int:
    if requested is not None:
        if requested < min_history or requested >= token_count:
            raise ValueError("Requested token_index must lie in [min_history, token_count-1]")
        return requested
    candidate = max(token_count // 2, min_history)
    if candidate >= token_count:
        candidate = token_count - 1
    if candidate < min_history:
        raise ValueError("Sequence too short to satisfy min_history")
    return candidate


def view_as_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    seq_len, head_dim = tensor.shape
    if head_dim % 2 != 0:
        raise ValueError("Head dimension must be even to form complex pairs")
    freq_count = head_dim // 2
    # Qwen stores RoPE-rotated vectors as [real_half | imag_half]; split halves explicitly.
    real = tensor[:, :freq_count].contiguous()
    imag = tensor[:, freq_count:].contiguous()
    return torch.complex(real, imag)


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_freq_topfreq")

    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError("trace_dir must contain qk.pt and metadata.json")

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

    inv_freq, _ = _compute_yarn_parameters(config, torch.device("cpu"))

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]
    k_tensor: torch.Tensor = data["k"]

    if args.layer >= q_tensor.shape[0] or args.head >= q_tensor.shape[1]:
        raise IndexError("Layer/head index out of bounds for captured tensors")

    q_block = q_tensor[args.layer, args.head].float()
    k_block = k_tensor[args.layer, args.head].float()
    if token_count:
        q_block = q_block[:token_count]
        k_block = k_block[:token_count]

    token_idx = choose_token(q_block.shape[0], args.min_history, args.token_index)

    freq_count = q_block.shape[-1] // 2
    omega = inv_freq[:freq_count]
    periods = (2 * torch.pi) / omega

    q_complex = view_as_complex_pairs(q_block)
    k_complex = view_as_complex_pairs(k_block)

    mean_mag = (q_complex.abs() * k_complex.abs()).mean(dim=0)
    top_k = min(args.top_k, freq_count)
    top_vals, top_indices = torch.topk(mean_mag, k=top_k)

    window_max = token_idx if args.fft_window is None else min(args.fft_window, token_idx)
    window = max(args.min_history, window_max)
    if window < 1:
        raise ValueError("FFT window must be >= 1")

    hist_slice = k_complex[token_idx - window : token_idx]
    hist_reversed = hist_slice.flip(0)
    q_token = q_complex[token_idx]

    deltas = torch.arange(1, window + 1, dtype=torch.float32)
    phase_full = deltas.unsqueeze(1) * omega.unsqueeze(0)
    exp_neg_full = torch.exp(-1j * phase_full)
    exp_pos_full = torch.exp(1j * phase_full)

    cols = 3
    rows = (top_k + cols - 1) // cols
    rows = max(rows, 1)
    fig, axes = plt.subplots(rows, cols, figsize=tuple(args.figsize), dpi=args.dpi)
    axes = axes.flatten()
    for ax in axes[top_k:]:
        ax.axis("off")

    x_vals = deltas.numpy()
    for idx_plot, (f_idx, m_val) in enumerate(zip(top_indices.tolist(), top_vals.tolist())):
        ax = axes[idx_plot]
        contrib = q_token[f_idx] * torch.conj(hist_reversed[:, f_idx])
        raw_series = contrib.real

        exp_neg = exp_neg_full[:, f_idx]
        coeff = (contrib * exp_neg).sum() / window
        exp_pos = exp_pos_full[:, f_idx]
        recon = (exp_pos * coeff).real

        ax.plot(x_vals, raw_series.numpy(), label="Raw", linewidth=1.2)
        ax.plot(x_vals, recon.numpy(), label="Recon", linewidth=1.2)
        period_val = periods[f_idx].item()
        ax.set_title(f"f={f_idx} (period≈{period_val:.1f}), m_f={m_val:.3f}")
        ax.set_xlabel("Δ (tokens)")
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3, linestyle="--")
        ax.legend(fontsize="small")

    out_path = args.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
