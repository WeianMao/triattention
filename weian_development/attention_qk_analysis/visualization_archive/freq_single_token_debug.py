"""Debug visualization comparing a single query's raw scores and Fourier reconstruction."""
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Inspect Fourier reconstruction for a single query token")
    parser.add_argument(
        "trace_dir",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        nargs="?",
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--layer", type=int, default=0, help="Layer index")
    parser.add_argument("--head", type=int, default=0, help="Head index")
    parser.add_argument(
        "--token-index",
        type=int,
        default=None,
        help="Query token index to inspect; default picks the middle token with sufficient history",
    )
    parser.add_argument(
        "--min-history",
        type=int,
        default=100,
        help="Minimum history (Δ) required for the chosen token",
    )
    parser.add_argument(
        "--max-distance",
        type=int,
        default=1024,
        help="Maximum distance Δ to visualize",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE frequencies",
    )
    parser.add_argument(
        "--fft-window",
        type=int,
        default=1024,
        help="Number of historical positions used for the RoPE-frequency transform",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Computation dtype",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=Path("outputs/deepseek_r1_qwen3_8b/single_token_debug.png"),
        help="Where to save the comparison plot",
    )
    parser.add_argument("--dpi", type=int, default=200, help="Figure DPI")
    parser.add_argument("--figsize", type=float, nargs=2, default=(10.0, 4.0), help="Figure size")
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
            raise ValueError("Requested token_index must be within [min_history, token_count-1]")
        return requested
    candidate = max(token_count // 2, min_history)
    if candidate >= token_count:
        candidate = token_count - 1
    if candidate < min_history:
        raise ValueError("Sequence too short to satisfy min_history for any token")
    return candidate


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_freq_debug")

    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError("trace_dir must contain qk.pt and metadata.json")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta.get("sequence_length", 0))

    device = torch.device(args.device)
    dtype = select_dtype(args.dtype)

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rotary = Qwen3RotaryEmbedding(config=config, device="cpu")
    inv_freq = rotary.inv_freq.to(dtype=dtype)

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]
    k_tensor: torch.Tensor = data["k"]

    if args.layer >= q_tensor.shape[0] or args.head >= q_tensor.shape[1]:
        raise IndexError("Layer/head index out of bounds for captured tensors")

    q_block = q_tensor[args.layer, args.head].to(device=device, dtype=dtype)
    k_block = k_tensor[args.layer, args.head].to(device=device, dtype=dtype)
    if token_count:
        q_block = q_block[:token_count]
        k_block = k_block[:token_count]

    token_idx = choose_token(q_block.shape[0], args.min_history, args.token_index)

    window = max(args.min_history, min(args.fft_window, token_idx))
    if window < 1:
        raise ValueError("FFT window must be >= 1")

    deltas = torch.arange(1, window + 1, device=device, dtype=dtype)

    hist_start = token_idx - window
    hist_slice = k_block[hist_start:token_idx]
    hist_reversed = hist_slice.flip(0)
    raw_scores = torch.mv(hist_reversed, q_block[token_idx])

    omega = inv_freq[: q_block.shape[-1] // 2].to(device=device, dtype=dtype)
    phase = deltas.unsqueeze(1) * omega.unsqueeze(0)
    cos_phase = torch.cos(phase)
    sin_phase = torch.sin(phase)

    scores_centered = raw_scores - raw_scores.mean()
    exp_neg = torch.complex(cos_phase, -sin_phase)
    coeffs = torch.matmul(scores_centered.unsqueeze(0).to(exp_neg.dtype), exp_neg).squeeze(0) / window

    exp_pos = torch.complex(cos_phase, sin_phase)
    reconstructed_centered = torch.matmul(exp_pos, coeffs).real
    reconstructed = reconstructed_centered + raw_scores.mean()

    fig, ax = plt.subplots(1, 1, figsize=tuple(args.figsize), dpi=args.dpi)
    x_vals = deltas.detach().cpu().numpy()
    ax.plot(x_vals, raw_scores.detach().cpu().numpy(), label="Raw Q·K", linewidth=1.2)
    ax.plot(x_vals, reconstructed.detach().cpu().numpy(), label="Inverse FFT", linewidth=1.2)
    ax.set_title(f"Layer {args.layer}, Head {args.head}, Token {token_idx} (window={window})")
    ax.set_xlabel("Δ (tokens)")
    ax.set_ylabel("Value")
    ax.grid(alpha=0.3, linestyle="--")
    ax.legend()

    out_path = args.output_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)

    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
