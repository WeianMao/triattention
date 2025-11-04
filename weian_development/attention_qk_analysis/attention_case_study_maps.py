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

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


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
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return pooled heatmaps, key-group maxima/means, and their top-30 highlight masks."""

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
    key_count_tokens = key_count_tokens[:, :seq_len]

    if seq_len > 500:
        drop_start = seq_len - 500
        key_sum_tokens[:, drop_start:] = 0.0
        key_count_tokens[:, drop_start:] = 0.0

    key_avg_tokens = key_sum_tokens / key_count_tokens.clamp_min(1e-12)

    total_keys = num_k_groups * patch_size
    if key_max_tokens.shape[1] < total_keys:
        pad = torch.zeros(
            head_count,
            total_keys - key_max_tokens.shape[1],
            device=device,
            dtype=key_max_tokens.dtype,
        )
        key_max_tokens = torch.cat([key_max_tokens, pad], dim=1)
    if key_avg_tokens.shape[1] < total_keys:
        pad = torch.zeros(
            head_count,
            total_keys - key_avg_tokens.shape[1],
            device=device,
            dtype=key_avg_tokens.dtype,
        )
        key_avg_tokens = torch.cat([key_avg_tokens, pad], dim=1)
    if key_count_tokens.shape[1] < total_keys:
        pad = torch.zeros(
            head_count,
            total_keys - key_count_tokens.shape[1],
            device=device,
            dtype=key_count_tokens.dtype,
        )
        key_count_tokens = torch.cat([key_count_tokens, pad], dim=1)

    token_positions = torch.arange(total_keys, device=device)

    # Top-30 keys by max attention weight
    valid_max_tokens = token_positions < seq_len
    topk_max = min(30, total_keys)
    key_max_top_tokens = torch.zeros(
        (head_count, total_keys), device=device, dtype=torch.bool
    )
    if valid_max_tokens.any():
        max_scores = key_max_tokens.clone()
        max_scores = max_scores.masked_fill(~valid_max_tokens.unsqueeze(0), float("-inf"))
        topk_indices = max_scores.topk(k=topk_max, dim=1).indices
        key_max_top_tokens.scatter_(1, topk_indices, True)
        key_max_top_tokens &= valid_max_tokens.unsqueeze(0)

    # Top-30 keys by mean attention weight (excluding last 500 tokens)
    mean_cutoff = seq_len - 500 if seq_len > 500 else seq_len
    valid_mean_tokens = token_positions < mean_cutoff
    attended_mean = key_count_tokens > 0.0
    mean_valid_mask = valid_mean_tokens.unsqueeze(0) & attended_mean
    topk_mean = min(30, total_keys)
    key_avg_top_tokens = torch.zeros(
        (head_count, total_keys), device=device, dtype=torch.bool
    )
    if mean_valid_mask.any():
        mean_scores = key_avg_tokens.clone()
        mean_scores = mean_scores.masked_fill(~mean_valid_mask, float("-inf"))
        topk_indices_mean = mean_scores.topk(k=topk_mean, dim=1).indices
        key_avg_top_tokens.scatter_(1, topk_indices_mean, True)
        key_avg_top_tokens &= mean_valid_mask

    key_max_groups = key_max_tokens.view(head_count, num_k_groups, patch_size).amax(dim=2)
    key_avg_groups = key_avg_tokens.view(head_count, num_k_groups, patch_size).amax(dim=2)
    key_max_top_groups = (
        key_max_top_tokens.view(head_count, num_k_groups, patch_size).amax(dim=2).to(torch.float32)
    )
    key_avg_top_groups = (
        key_avg_top_tokens.view(head_count, num_k_groups, patch_size).amax(dim=2).to(torch.float32)
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
            ) = compute_attention_and_keymax(
                q_block,
                k_block,
                seq_len,
                patch_size,
                args.q_tile,
                device,
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

        del (
            q_block,
            k_block,
            heatmaps,
            key_max_groups,
            key_avg_groups,
            key_max_top_groups,
            key_avg_top_groups,
        )

    del q_tensor, k_tensor, data
    if device.type == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
