"""Case study attention visualization with keywise max pooling overlays."""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

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
from weian_development.attention_qk_analysis.freq_magnitude_single_plot_meanvec import (
    to_complex_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate attention map case studies with additional keywise max pooling visualizations."
        )
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="Directory containing qid*_trace*/qk.pt (e.g., outputs/.../qk_bf16_traces).",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="Trace folder name to process (e.g., qid0003_trace34).",
    )
    parser.add_argument(
        "--layer-head",
        dest="layer_heads",
        action="append",
        required=True,
        help="Layer/head spec in the form L:H (zero-indexed, e.g., 3:05 or 3:5).",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help=(
            "Base directory for outputs (defaults to <input_root>/../attention_case_studies)."
        ),
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=4096,
        help="Target pixel count (query/key) when inferring pooling window.",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Pooling window along query/key axes (overrides --target-size inference).",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=512,
        help="Number of queries to score per tile (to control memory).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for attention computation (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="Computation dtype for Q/K tensors (float32 suggested).",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="DPI when saving images.",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(20.48, 20.48),
        help="Figure size (inches) for the attention heatmap baseline output.",
    )
    parser.add_argument(
        "--colormap",
        default="inferno",
        help="Matplotlib colormap name.",
    )
    parser.add_argument(
        "--skip-baseline",
        action="store_true",
        help="Skip saving the baseline heatmap that mirrors visualize_attention_maps.py output.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE parameters.",
    )
    parser.add_argument(
        "--topk",
        type=int,
        default=30,
        help="Number of key tokens to highlight per scoring mechanism.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed used when sampling negative key examples.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log detailed progress information.",
    )
    return parser.parse_args()


def select_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def resolve_patch_size(seq_len: int, target_size: int, patch_arg: int | None) -> int:
    if patch_arg and patch_arg > 0:
        return patch_arg
    if seq_len <= target_size:
        return 1
    return math.ceil(seq_len / target_size)


def parse_layer_head_spec(specs: Sequence[str]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = defaultdict(list)
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid layer/head spec: {spec}")
        layer_str, head_str = spec.split(":", 1)
        layer = int(layer_str)
        head = int(head_str)
        grouped[layer].append(head)
    return grouped


def compute_attention_and_keymax(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
    topk: int,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    List[torch.Tensor],
    List[torch.Tensor],
    torch.Tensor,
]:
    """Return pooled heatmaps, key-group statistics, and token-level top-k selections."""

    head_count, seq_q, head_dim = q_block.shape
    _, seq_k, _ = k_block.shape
    scale = head_dim ** -0.5

    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    q_pad = num_q_groups * patch_size - seq_len
    k_pad = num_k_groups * patch_size - seq_len
    if q_pad > 0:
        pad = torch.zeros(head_count, q_pad, head_dim, device=device, dtype=q_block.dtype)
        q_block = torch.cat([q_block, pad], dim=1)
    if k_pad > 0:
        pad = torch.zeros(head_count, k_pad, head_dim, device=device, dtype=k_block.dtype)
        k_block = torch.cat([k_block, pad], dim=1)

    seq_q_real = seq_len
    seq_k_padded = k_block.shape[1]

    key_positions = torch.arange(seq_k_padded, device=device)
    key_valid = key_positions < seq_len

    pooled_groups = torch.zeros(
        (head_count, num_q_groups, num_k_groups),
        device=device,
        dtype=torch.float32,
    )
    key_max_tokens = torch.zeros(
        (head_count, seq_k_padded),
        device=device,
        dtype=torch.float32,
    )
    key_sum_tokens = torch.zeros(
        (head_count, seq_k_padded),
        device=device,
        dtype=torch.float32,
    )
    key_count_tokens = torch.zeros(
        (head_count, seq_k_padded),
        device=device,
        dtype=torch.float32,
    )

    k_t = k_block.transpose(1, 2).contiguous()

    for q_start in range(0, seq_q_real, q_tile):
        q_end = min(q_start + q_tile, seq_q_real)
        indices = torch.arange(q_start, q_end, device=device)
        q_slice = q_block[:, q_start:q_end, :]

        scores = torch.matmul(q_slice, k_t) * scale
        scores = scores.to(torch.float32)

        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        valid_mask = (~future_mask).unsqueeze(0) & key_valid.view(1, 1, -1)

        mask_float = valid_mask.to(scores.dtype)
        mask_head = mask_float.expand(head_count, -1, -1)

        scores = scores.masked_fill(~valid_mask, float("-inf"))

        scores_flat = scores.view(head_count, -1, num_k_groups * patch_size)
        row_max = scores_flat.max(dim=-1, keepdim=True).values
        row_max = torch.where(
            torch.isfinite(row_max),
            row_max,
            torch.zeros_like(row_max),
        )

        stable = torch.exp(scores_flat - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (stable / row_sum).view(head_count, -1, num_k_groups, patch_size)

        key_mask = key_valid.view(1, 1, num_k_groups, patch_size)
        weights = weights * key_mask

        reshaped = weights.view(head_count, -1, seq_k_padded)
        key_sum_tokens += reshaped.sum(dim=1)
        key_count_tokens += mask_head.sum(dim=1)
        tile_token_max = reshaped.max(dim=1).values
        key_max_tokens = torch.maximum(key_max_tokens, tile_token_max)

        pooled_k = weights.max(dim=-1).values

        query_groups = indices // patch_size
        base_group = int(query_groups.min().item())
        local_groups = (query_groups - base_group).to(torch.int64)
        groups_in_tile = int(local_groups.max().item()) + 1

        expanded_index = local_groups.view(1, -1, 1).expand(head_count, -1, num_k_groups)
        tile_max = torch.zeros(
            (head_count, groups_in_tile, num_k_groups),
            device=device,
            dtype=torch.float32,
        )
        tile_max.scatter_reduce_(
            dim=1,
            index=expanded_index,
            src=pooled_k,
            reduce="amax",
        )

        end_group = base_group + groups_in_tile
        pooled_groups[:, base_group:end_group] = torch.maximum(
            pooled_groups[:, base_group:end_group], tile_max
        )

    valid_query_groups = math.ceil(seq_len / patch_size)
    valid_key_groups = math.ceil(seq_len / patch_size)
    if valid_query_groups < num_q_groups:
        pooled_groups[:, valid_query_groups:, :] = 0.0
    if valid_key_groups < num_k_groups:
        pooled_groups[:, :, valid_key_groups:] = 0.0

    key_max_tokens = key_max_tokens[:, :seq_len]
    key_sum_tokens = key_sum_tokens[:, :seq_len]
    key_count_trimmed = key_count_tokens[:, :seq_len]

    if seq_len > 500:
        drop_start = seq_len - 500
        key_sum_tokens[:, drop_start:] = 0.0
        key_count_trimmed[:, drop_start:] = 0.0

    key_avg_tokens = key_sum_tokens / key_count_trimmed.clamp_min(1e-12)

    total_keys = num_k_groups * patch_size

    def pad_to_total(tensor: torch.Tensor) -> torch.Tensor:
        if tensor.shape[1] < total_keys:
            pad = torch.zeros(
                head_count,
                total_keys - tensor.shape[1],
                device=device,
                dtype=tensor.dtype,
            )
            tensor = torch.cat([tensor, pad], dim=1)
        return tensor

    key_max_tokens = pad_to_total(key_max_tokens)
    key_avg_tokens = pad_to_total(key_avg_tokens)
    key_count_tokens = pad_to_total(key_count_trimmed.clone())

    token_positions = torch.arange(total_keys, device=device)
    valid_max_mask = token_positions < seq_len
    mean_cutoff = seq_len - 500 if seq_len > 500 else seq_len
    valid_mean_base = token_positions < mean_cutoff

    key_max_top_tokens = torch.zeros(
        (head_count, total_keys), device=device, dtype=torch.bool
    )
    key_avg_top_tokens = torch.zeros(
        (head_count, total_keys), device=device, dtype=torch.bool
    )
    max_index_list: List[torch.Tensor] = []
    mean_index_list: List[torch.Tensor] = []

    for head_idx in range(head_count):
        head_max_scores = key_max_tokens[head_idx].clone()
        head_valid_max = valid_max_mask.clone()
        head_max_scores[~head_valid_max] = float("-inf")
        valid_max_count = int(head_valid_max.sum().item())
        select_max = min(topk, valid_max_count)
        if select_max > 0 and head_valid_max.any():
            max_vals, max_idx = torch.topk(head_max_scores, k=select_max)
            finite_mask = torch.isfinite(max_vals)
            max_idx = max_idx[finite_mask]
            if max_idx.numel() > 0:
                key_max_top_tokens[head_idx, max_idx] = True
        else:
            max_idx = head_max_scores.new_empty(0, dtype=torch.long)
        max_index_list.append(max_idx.detach().cpu())

        head_mean_counts = key_count_tokens[head_idx]
        head_mean_valid = (head_mean_counts > 0.0) & valid_mean_base
        head_mean_scores = key_avg_tokens[head_idx].clone()
        head_mean_scores[~head_mean_valid] = float("-inf")
        valid_mean_count = int(head_mean_valid.sum().item())
        select_mean = min(topk, valid_mean_count)
        if select_mean > 0 and head_mean_valid.any():
            mean_vals, mean_idx = torch.topk(head_mean_scores, k=select_mean)
            finite_mask = torch.isfinite(mean_vals)
            mean_idx = mean_idx[finite_mask]
            if mean_idx.numel() > 0:
                key_avg_top_tokens[head_idx, mean_idx] = True
        else:
            mean_idx = head_mean_scores.new_empty(0, dtype=torch.long)
        mean_index_list.append(mean_idx.detach().cpu())

    key_max_groups = key_max_tokens.view(head_count, num_k_groups, patch_size).amax(dim=2)
    key_avg_groups = key_avg_tokens.view(head_count, num_k_groups, patch_size).amax(dim=2)
    key_max_top_groups = (
        key_max_top_tokens.view(head_count, num_k_groups, patch_size)
        .any(dim=2)
        .to(torch.float32)
    )
    key_avg_top_groups = (
        key_avg_top_tokens.view(head_count, num_k_groups, patch_size)
        .any(dim=2)
        .to(torch.float32)
    )

    key_min = key_max_groups.amin(dim=1, keepdim=True)
    key_max = key_max_groups.amax(dim=1, keepdim=True)
    key_denom = (key_max - key_min).clamp_min(1e-12)
    key_max_norm = torch.clamp((key_max_groups - key_min) / key_denom, 0.0, 1.0)

    key_avg_min = key_avg_groups.amin(dim=1, keepdim=True)
    key_avg_max = key_avg_groups.amax(dim=1, keepdim=True)
    key_avg_denom = (key_avg_max - key_avg_min).clamp_min(1e-12)
    key_avg_norm = torch.clamp((key_avg_groups - key_avg_min) / key_avg_denom, 0.0, 1.0)

    row_min = pooled_groups.amin(dim=2, keepdim=True)
    row_max = pooled_groups.amax(dim=2, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_groups - row_min) / denom
    norm = torch.clamp(norm, 0.0, 1.0)

    return (
        norm.detach().cpu(),
        key_max_norm.detach().cpu(),
        key_avg_norm.detach().cpu(),
        key_max_top_groups.detach().cpu(),
        key_avg_top_groups.detach().cpu(),
        [idx.clone() for idx in max_index_list],
        [idx.clone() for idx in mean_index_list],
        key_count_trimmed.detach().cpu(),
    )


def save_baseline_heatmap(
    heatmap: torch.Tensor,
    out_path: Path,
    cmap: str,
    figsize: Tuple[float, float],
    dpi: int,
    title: str,
) -> None:
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    im = ax.imshow(heatmap.numpy(), cmap=cmap, aspect="auto", origin="upper")
    ax.set_title(title)
    ax.set_xlabel("Key group index")
    ax.set_ylabel("Query group index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_case_study_figure(
    heatmap: torch.Tensor,
    key_max: torch.Tensor,
    key_avg: torch.Tensor,
    key_max_mask: torch.Tensor,
    key_avg_mask: torch.Tensor,
    out_path: Path,
    cmap: str,
    figsize: Tuple[float, float],
    dpi: int,
    title: str,
    patch_size: int,
) -> None:
    # Extend figure height to host the keywise max strip while preserving baseline width.
    width, height = figsize
    case_height = height * 1.6

    fig = plt.figure(figsize=(width, case_height), dpi=dpi)
    gs = fig.add_gridspec(
        nrows=5,
        ncols=1,
        height_ratios=[4, 0.7, 0.35, 0.7, 0.35],
        hspace=0.35,
    )

    ax_main = fig.add_subplot(gs[0, 0])
    im = ax_main.imshow(heatmap.numpy(), cmap=cmap, aspect="auto", origin="upper")
    ax_main.set_title(title)
    ax_main.set_xlabel("Key group index")
    ax_main.set_ylabel("Query group index")
    fig.colorbar(im, ax=ax_main, fraction=0.046, pad=0.02)

    ax_max = fig.add_subplot(gs[1, 0])
    max_data = key_max.unsqueeze(0).numpy()
    max_strip = ax_max.imshow(max_data, cmap=cmap, aspect="auto", origin="lower")
    ax_max.set_xlabel(f"Key group index (patch={patch_size})")
    ax_max.set_yticks([])
    ax_max.set_title("Max attention over queries (per key group)")
    max_strip.set_clim(0.0, 1.0)

    ax_max_mask = fig.add_subplot(gs[2, 0])
    max_mask_data = key_max_mask.unsqueeze(0).numpy()
    max_mask_strip = ax_max_mask.imshow(
        max_mask_data, cmap=cmap, aspect="auto", origin="lower"
    )
    ax_max_mask.set_xlabel(f"Top-30 mask (max attention, patch={patch_size})")
    ax_max_mask.set_yticks([])
    ax_max_mask.set_title("Highlighted keys (max attention)")
    max_mask_strip.set_clim(0.0, 1.0)

    ax_avg = fig.add_subplot(gs[3, 0])
    avg_data = key_avg.unsqueeze(0).numpy()
    avg_strip = ax_avg.imshow(avg_data, cmap=cmap, aspect="auto", origin="lower")
    ax_avg.set_xlabel(f"Key group index (patch={patch_size})")
    ax_avg.set_yticks([])
    ax_avg.set_title("Mean attention weight (per key group, drop last 500 keys)")
    avg_strip.set_clim(0.0, 1.0)

    ax_avg_mask = fig.add_subplot(gs[4, 0])
    avg_mask_data = key_avg_mask.unsqueeze(0).numpy()
    avg_mask_strip = ax_avg_mask.imshow(
        avg_mask_data, cmap=cmap, aspect="auto", origin="lower"
    )
    ax_avg_mask.set_xlabel(f"Top-30 mask (mean attention, patch={patch_size})")
    ax_avg_mask.set_yticks([])
    ax_avg_mask.set_title("Highlighted keys (mean attention)")
    avg_mask_strip.set_clim(0.0, 1.0)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def sample_negative_indices(
    valid_indices: torch.Tensor,
    positives: torch.Tensor,
    sample_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if valid_indices.numel() == 0:
        return torch.empty(0, dtype=torch.long)

    positive_set = set(positives.tolist()) if positives.numel() > 0 else set()
    remaining = [idx for idx in valid_indices.tolist() if idx not in positive_set]
    if not remaining:
        return torch.empty(0, dtype=torch.long)

    remaining_tensor = torch.tensor(remaining, dtype=torch.long)
    if remaining_tensor.numel() <= sample_size:
        return remaining_tensor

    perm = torch.randperm(remaining_tensor.numel(), generator=generator)[:sample_size]
    return remaining_tensor[perm]


def compute_weighted_series(
    q_mean_abs: torch.Tensor,
    k_abs: torch.Tensor,
    k_mean_abs: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    if token_indices.numel() == 0:
        return torch.zeros_like(q_mean_abs)

    subset = k_abs.index_select(0, token_indices)
    delta = subset - k_mean_abs.unsqueeze(0)
    values = q_mean_abs.unsqueeze(0) * delta
    return values.mean(dim=0)


def compute_weighted_series_matrix(
    q_mean_abs: torch.Tensor,
    k_abs: torch.Tensor,
    k_mean_abs: torch.Tensor,
    token_indices: torch.Tensor,
) -> torch.Tensor:
    if token_indices.numel() == 0:
        return torch.zeros(0, q_mean_abs.shape[0], dtype=q_mean_abs.dtype)
    subset = k_abs.index_select(0, token_indices)
    delta = subset - k_mean_abs.unsqueeze(0)
    return q_mean_abs.unsqueeze(0) * delta


def sample_subset(
    indices: torch.Tensor,
    sample_size: int,
    generator: torch.Generator,
) -> torch.Tensor:
    if indices.numel() <= sample_size:
        return indices
    perm = torch.randperm(indices.numel(), generator=generator)[:sample_size]
    return indices[perm]


def save_frequency_overlay_figure(
    freq_axis: torch.Tensor,
    q_mean_abs: torch.Tensor,
    max_pos: torch.Tensor,
    max_neg: torch.Tensor,
    mean_pos: torch.Tensor,
    mean_neg: torch.Tensor,
    counts: Dict[str, int],
    out_path: Path,
    dpi: int,
) -> None:
    freq_np = freq_axis.cpu().numpy()
    fig, axes = plt.subplots(5, 1, figsize=(12.0, 14.0), dpi=dpi, sharex=True)

    axes[0].plot(freq_np, q_mean_abs.cpu().numpy(), color="tab:blue", linewidth=1.5)
    axes[0].set_title("|E[q_f]|")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(alpha=0.3, linestyle="--")

    axes[1].plot(freq_np, max_pos.cpu().numpy(), color="tab:orange", linewidth=1.5)
    axes[1].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    axes[1].set_title(f"Top-{counts['max_pos']} max-attention keys")
    axes[1].set_ylabel("Value")
    axes[1].grid(alpha=0.3, linestyle="--")

    axes[2].plot(freq_np, max_neg.cpu().numpy(), color="tab:green", linewidth=1.5)
    axes[2].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    axes[2].set_title(f"Random {counts['max_neg']} non-top keys (max-attention)")
    axes[2].set_ylabel("Value")
    axes[2].grid(alpha=0.3, linestyle="--")

    axes[3].plot(freq_np, mean_pos.cpu().numpy(), color="tab:red", linewidth=1.5)
    axes[3].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    axes[3].set_title(f"Top-{counts['mean_pos']} mean-attention keys")
    axes[3].set_ylabel("Value")
    axes[3].grid(alpha=0.3, linestyle="--")

    axes[4].plot(freq_np, mean_neg.cpu().numpy(), color="tab:purple", linewidth=1.5)
    axes[4].axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
    axes[4].set_title(f"Random {counts['mean_neg']} non-top keys (mean-attention)")
    axes[4].set_ylabel("Value")
    axes[4].set_xlabel("Frequency index f")
    axes[4].grid(alpha=0.3, linestyle="--")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def save_frequency_overlay_matrix(
    freq_axis: torch.Tensor,
    q_mean_abs: torch.Tensor,
    matrices: Dict[str, torch.Tensor],
    out_path: Path,
    dpi: int,
) -> None:
    freq_np = freq_axis.cpu().numpy()
    fig, axes = plt.subplots(5, 1, figsize=(12.0, 18.0), dpi=dpi, sharex=True)

    axes[0].plot(freq_np, q_mean_abs.cpu().numpy(), color="tab:blue", linewidth=1.5)
    axes[0].set_title("|E[q_f]|")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(alpha=0.3, linestyle="--")

    def plot_matrix(ax, matrix: torch.Tensor, title: str) -> None:
        if matrix.numel() == 0:
            ax.set_title(f"{title} (no samples)")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.3, linestyle="--")
            return
        arr = matrix.cpu().numpy()
        for row in arr:
            ax.plot(freq_np, row, linewidth=0.8, alpha=0.6)
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3, linestyle="--")

    plot_matrix(
        axes[1],
        matrices["max_pos"],
        f"Top-{matrices['max_pos'].shape[0]} max-attention keys",
    )
    plot_matrix(
        axes[2],
        matrices["max_neg"],
        f"Random {matrices['max_neg'].shape[0]} non-top keys (max-attention)",
    )
    plot_matrix(
        axes[3],
        matrices["mean_pos"],
        f"Top-{matrices['mean_pos'].shape[0]} mean-attention keys",
    )
    plot_matrix(
        axes[4],
        matrices["mean_neg"],
        f"Random {matrices['mean_neg'].shape[0]} non-top keys (mean-attention)",
    )
    axes[4].set_xlabel("Frequency index f")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_frequency_overlay_subset(
    freq_axis: torch.Tensor,
    q_mean_abs: torch.Tensor,
    subset: Dict[str, torch.Tensor],
    out_path: Path,
    dpi: int,
) -> None:
    freq_np = freq_axis.cpu().numpy()
    fig, axes = plt.subplots(5, 1, figsize=(12.0, 14.0), dpi=dpi, sharex=True)

    axes[0].plot(freq_np, q_mean_abs.cpu().numpy(), color="tab:blue", linewidth=1.5)
    axes[0].set_title("|E[q_f]|")
    axes[0].set_ylabel("Magnitude")
    axes[0].grid(alpha=0.3, linestyle="--")

    def plot_subset(ax, matrix: torch.Tensor, title: str) -> None:
        if matrix.numel() == 0:
            ax.set_title(f"{title} (no samples)")
            ax.set_ylabel("Value")
            ax.grid(alpha=0.3, linestyle="--")
            return
        arr = matrix.cpu().numpy()
        for row in arr:
            ax.plot(freq_np, row, linewidth=1.1, alpha=0.75)
        ax.axhline(0.0, color="gray", linewidth=0.8, alpha=0.7)
        ax.set_title(title)
        ax.set_ylabel("Value")
        ax.grid(alpha=0.3, linestyle="--")

    plot_subset(
        axes[1],
        subset["max_pos"],
        f"Random {subset['max_pos'].shape[0]} from max-top30",
    )
    plot_subset(
        axes[2],
        subset["max_neg"],
        f"Random {subset['max_neg'].shape[0]} non-top (max attention)",
    )
    plot_subset(
        axes[3],
        subset["mean_pos"],
        f"Random {subset['mean_pos'].shape[0]} from mean-top30",
    )
    plot_subset(
        axes[4],
        subset["mean_neg"],
        f"Random {subset['mean_neg'].shape[0]} non-top (mean attention)",
    )
    axes[4].set_xlabel("Frequency index f")

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_case")

    trace_dir = args.input_root / args.trace
    if not trace_dir.exists():
        raise SystemExit(f"Trace directory not found: {trace_dir}")

    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise SystemExit(f"Missing qk assets in {trace_dir}")

    output_root = args.output_root
    if output_root is None:
        output_root = args.input_root.parent / "attention_case_studies"
    output_root = output_root / trace_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    layer_to_heads = parse_layer_head_spec(args.layer_heads)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = int(meta["sequence_length"])

    patch_size = resolve_patch_size(seq_len, args.target_size, args.patch_size)
    if args.verbose:
        print(
            f"Using patch_size={patch_size} (target={args.target_size}, sequence_length={seq_len})"
        )

    device = torch.device(args.device)
    compute_dtype = select_dtype(args.dtype)

    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True

    load_start = time.perf_counter()
    data = torch.load(qk_path, map_location="cpu")
    load_dur = time.perf_counter() - load_start
    if args.verbose:
        print(f"Loaded qk.pt in {load_dur:.2f}s")
    q_tensor = data["q"]
    k_tensor = data["k"]

    config = AutoConfig.from_pretrained(args.model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling
    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    head_dim = q_tensor.shape[-1]
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=compute_dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0].to(dtype=compute_dtype)
    sin_table = sin_table[0].to(dtype=compute_dtype)


    for layer, heads in layer_to_heads.items():
        unique_heads = sorted(set(heads))
        if args.verbose:
            print(f"Processing layer {layer} heads {unique_heads}")
        q_block = q_tensor[layer, unique_heads].to(device=device, dtype=compute_dtype)
        k_block = k_tensor[layer, unique_heads].to(device=device, dtype=compute_dtype)

        with torch.no_grad():
            (
                heatmaps,
                key_max_groups,
                key_avg_groups,
                key_max_top_groups,
                key_avg_top_groups,
                max_token_lists,
                mean_token_lists,
                key_count_valid,
            ) = compute_attention_and_keymax(
                q_block,
                k_block,
                seq_len,
                patch_size,
                args.q_tile,
                device,
                args.topk,
            )

        for idx, head in enumerate(unique_heads):
            title = f"Layer {layer} Head {head} | patch={patch_size}"
            baseline_path = output_root / f"layer_{layer:02d}_head_{head:02d}.png"
            case_path = output_root / f"layer_{layer:02d}_head_{head:02d}_case.png"

            if not args.skip_baseline:
                save_baseline_heatmap(
                    heatmaps[idx],
                    baseline_path,
                    args.colormap,
                    tuple(args.figsize),
                    args.dpi,
                    title,
                )
                if args.verbose:
                    print(f"Saved baseline {baseline_path.relative_to(output_root)}")

            save_case_study_figure(
                heatmaps[idx],
                key_max_groups[idx],
                key_avg_groups[idx],
                key_max_top_groups[idx],
                key_avg_top_groups[idx],
                case_path,
                args.colormap,
                tuple(args.figsize),
                args.dpi,
                title,
                patch_size,
            )
            if args.verbose:
                print(f"Saved case study {case_path.relative_to(output_root)}")

            q_head = q_block[idx, :seq_len, :]
            k_head = k_block[idx, :seq_len, :]

            q_unrot = invert_rope(q_head, cos_table, sin_table, attention_scale)
            k_unrot = invert_rope(k_head, cos_table, sin_table, attention_scale)

            q_complex = to_complex_pairs(q_unrot.detach().cpu())
            k_complex = to_complex_pairs(k_unrot.detach().cpu())

            q_mean_abs = torch.abs(q_complex.mean(dim=0))
            k_mean_abs = torch.abs(k_complex.mean(dim=0))
            k_abs = torch.abs(k_complex)

            max_pos_tokens = max_token_lists[idx]
            max_pos_tokens = max_pos_tokens[max_pos_tokens < seq_len]
            if max_pos_tokens.numel() > 0:
                max_pos_tokens = torch.unique(max_pos_tokens.to(torch.long))
            else:
                max_pos_tokens = torch.empty(0, dtype=torch.long)

            valid_max_indices = torch.arange(seq_len, dtype=torch.long)
            gen_max = torch.Generator(device="cpu")
            gen_max.manual_seed(args.seed + layer * 1000 + head)
            max_neg_tokens = sample_negative_indices(
                valid_max_indices, max_pos_tokens, args.topk, gen_max
            )

            key_counts_head = key_count_valid[idx]
            mean_valid_mask = key_counts_head > 0
            mean_valid_indices = mean_valid_mask.nonzero(as_tuple=False).squeeze(-1)

            mean_pos_tokens = mean_token_lists[idx]
            mean_pos_tokens = mean_pos_tokens[mean_pos_tokens < seq_len]
            if mean_pos_tokens.numel() > 0:
                mean_pos_tokens = mean_pos_tokens.to(torch.long)
                valid_mask = mean_valid_mask[mean_pos_tokens]
                mean_pos_tokens = mean_pos_tokens[valid_mask]
                mean_pos_tokens = torch.unique(mean_pos_tokens)
            else:
                mean_pos_tokens = torch.empty(0, dtype=torch.long)

            gen_mean = torch.Generator(device="cpu")
            gen_mean.manual_seed(args.seed + layer * 1000 + head + 1)
            mean_neg_tokens = sample_negative_indices(
                mean_valid_indices, mean_pos_tokens, args.topk, gen_mean
            )

            freq_axis = torch.arange(q_mean_abs.shape[0], dtype=torch.float32)

            max_pos_series = compute_weighted_series(
                q_mean_abs, k_abs, k_mean_abs, max_pos_tokens
            )
            max_neg_series = compute_weighted_series(
                q_mean_abs, k_abs, k_mean_abs, max_neg_tokens
            )
            mean_pos_series = compute_weighted_series(
                q_mean_abs, k_abs, k_mean_abs, mean_pos_tokens
            )
            mean_neg_series = compute_weighted_series(
                q_mean_abs, k_abs, k_mean_abs, mean_neg_tokens
            )

            max_pos_matrix = compute_weighted_series_matrix(
                q_mean_abs, k_abs, k_mean_abs, max_pos_tokens
            )
            max_neg_matrix = compute_weighted_series_matrix(
                q_mean_abs, k_abs, k_mean_abs, max_neg_tokens
            )
            mean_pos_matrix = compute_weighted_series_matrix(
                q_mean_abs, k_abs, k_mean_abs, mean_pos_tokens
            )
            mean_neg_matrix = compute_weighted_series_matrix(
                q_mean_abs, k_abs, k_mean_abs, mean_neg_tokens
            )

            freq_path = output_root / f"layer_{layer:02d}_head_{head:02d}_freq_overlays.png"
            counts = {
                "max_pos": int(max_pos_tokens.numel()),
                "max_neg": int(max_neg_tokens.numel()),
                "mean_pos": int(mean_pos_tokens.numel()),
                "mean_neg": int(mean_neg_tokens.numel()),
            }
            save_frequency_overlay_figure(
                freq_axis,
                q_mean_abs,
                max_pos_series,
                max_neg_series,
                mean_pos_series,
                mean_neg_series,
                counts,
                freq_path,
                args.dpi,
            )
            if args.verbose:
                print(f"Saved frequency overlays {freq_path.relative_to(output_root)}")

            freq_matrix_path = (
                output_root
                / f"layer_{layer:02d}_head_{head:02d}_freq_overlays_multi.png"
            )
            matrix_dict = {
                "max_pos": max_pos_matrix,
                "max_neg": max_neg_matrix,
                "mean_pos": mean_pos_matrix,
                "mean_neg": mean_neg_matrix,
            }
            save_frequency_overlay_matrix(
                freq_axis,
                q_mean_abs,
                matrix_dict,
                freq_matrix_path,
                args.dpi,
            )
            if args.verbose:
                print(
                    f"Saved frequency overlay matrix {freq_matrix_path.relative_to(output_root)}"
                )

            subset_path = (
                output_root
                / f"layer_{layer:02d}_head_{head:02d}_freq_overlays_subset.png"
            )
            gen_subset = torch.Generator(device="cpu")
            gen_subset.manual_seed(args.seed + layer * 1000 + head + 2)

            subset_max_pos = sample_subset(
                max_pos_tokens, min(6, max_pos_tokens.numel()), gen_subset
            )
            subset_max_neg = sample_subset(
                max_neg_tokens, min(6, max_neg_tokens.numel()), gen_subset
            )
            subset_mean_pos = sample_subset(
                mean_pos_tokens, min(6, mean_pos_tokens.numel()), gen_subset
            )
            subset_mean_neg = sample_subset(
                mean_neg_tokens, min(6, mean_neg_tokens.numel()), gen_subset
            )

            subset_matrices = {
                "max_pos": compute_weighted_series_matrix(
                    q_mean_abs, k_abs, k_mean_abs, subset_max_pos
                ),
                "max_neg": compute_weighted_series_matrix(
                    q_mean_abs, k_abs, k_mean_abs, subset_max_neg
                ),
                "mean_pos": compute_weighted_series_matrix(
                    q_mean_abs, k_abs, k_mean_abs, subset_mean_pos
                ),
                "mean_neg": compute_weighted_series_matrix(
                    q_mean_abs, k_abs, k_mean_abs, subset_mean_neg
                ),
            }

            save_frequency_overlay_subset(
                freq_axis,
                q_mean_abs,
                subset_matrices,
                subset_path,
                args.dpi,
            )
            if args.verbose:
                print(
                    f"Saved frequency overlay subset {subset_path.relative_to(output_root)}"
                )

        del (
            q_block,
            k_block,
            heatmaps,
            key_max_groups,
            key_avg_groups,
            key_max_top_groups,
            key_avg_top_groups,
            max_token_lists,
            mean_token_lists,
            key_count_valid,
        )

    del q_tensor, k_tensor, data
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
