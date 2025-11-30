#!/usr/bin/env python3
"""
离线复刻 hybrid rounds xtrace（命中率与可视化），适配 LLaMA 捕获的 Q/K。

- 输入：`input_root/trace/qk.pt` 与 `metadata.json`（结构与 capture_fullkv_qk.py 相同）。
- 功能：复用 upstream `attention_pruning_case_study_hybrid_rounds_xtrace.py` 的频域打分与 round-based pruning，
  输出命中率指标（retention_metrics.json）与对比图（baseline vs pruned attention heatmap、argmax 覆盖、层均值柱状图）。
- 约束：仅读取旁路捕获的张量，不修改 SpeckV 主逻辑；默认 stats_trace = trace。
"""
from __future__ import annotations

import argparse
import json
import math
import sys
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

from weian_development.process_utils import mask_process_command  # noqa: E402
from weian_development.hf_offline_runner_sparse.round_pruning_utils import (  # noqa: E402
    build_geometric_offsets,
    build_rotary,
    compute_frequency_statistics_from_means,
    compute_rotary_tables,
    invert_rope,
    load_or_create_sample,
    to_complex_pairs,
)

DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Offline hit-rate + visualization for LLaMA Q/K captures (hybrid rounds xtrace)."
    )
    parser.add_argument(
        "input_root",
        type=Path,
        help="目录，包含 <trace>/qk.pt 与 metadata.json（例如 outputs/qk_capture_fullkv）",
    )
    parser.add_argument(
        "--trace",
        required=True,
        help="trace 子目录名称（例如 shard00/run001_sample00061）",
    )
    parser.add_argument(
        "--stats-trace",
        type=Path,
        default=None,
        help="用于统计均值的 trace 目录（默认与 --trace 相同）",
    )
    parser.add_argument(
        "--head-sample-file",
        type=Path,
        default=Path("weian_development/rkv_debug/llama_sample_heads.json"),
        help="存放抽样 (layer, head) 的 JSON；不存在则按 sample-count/seed 生成",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=16,
        help="需要抽样的 head 数量（若 sample 文件已存在则忽略）",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="抽样用随机种子",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="输出基目录（默认 input_root/../llama_pruning_case_studies_xtrace/<trace>/agg_*）",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=4096,
        help="推断 pooling patch 的目标尺寸（较短序列自动设为 1），如未显式指定 patch-size",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="显式 pooling patch（默认 32，对齐 32x32 pooling）；若提供则覆盖 target-size 推断",
    )
    parser.add_argument(
        "--min-patch-size",
        type=int,
        default=32,
        help="最小 patch size，默认 32（保证不低于 32x32 pooling）",
    )
    parser.add_argument(
        "--max-seq-len",
        type=int,
        default=None,
        help="可选：仅使用前 N 个 token 做离线指标/可视化，便于 smoke（默认全长）",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=512,
        help="分块处理的 query 数",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="计算设备（如 cuda:0 或 cpu）",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        required=True,
        help="模型目录（用于构造 RoPE 参数，兼容 LLaMA/Qwen）",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="计算 dtype（建议 float32）",
    )
    parser.add_argument(
        "--max-keys",
        "--keep-keys",
        dest="max_keys",
        type=int,
        default=2048,
        help="每轮保留的最大 KV 数",
    )
    parser.add_argument(
        "--round-window",
        type=int,
        default=128,
        help="每轮维护的 token 数",
    )
    parser.add_argument(
        "--offset-max-length",
        type=int,
        default=65536,
        help="几何网格的最大 offset",
    )
    parser.add_argument(
        "--score-aggregation",
        choices=["mean", "max"],
        default="mean",
        help="分数聚合方式",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(20.48, 10.24),
        help="图像尺寸（英寸）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="图片 DPI",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印详细日志",
    )
    return parser.parse_args()


def resolve_patch_size(seq_len: int, target_size: int, patch_arg: int | None) -> int:
    if patch_arg and patch_arg > 0:
        return min(seq_len, patch_arg)
    base = 1 if seq_len <= target_size else math.ceil(seq_len / target_size)
    return min(seq_len, base)


def compute_pooled_attention(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
    prune_mask: torch.Tensor | None = None,
    return_argmax: bool = False,
    return_query_argmax: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
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

    argmax_groups = None
    if return_argmax:
        argmax_groups = torch.zeros_like(pooled_groups)

    query_argmax = None
    if return_query_argmax:
        query_argmax = torch.full(
            (head_count, seq_len),
            -1,
            device=device,
            dtype=torch.long,
        )

    if prune_mask is not None and prune_mask.shape[1] < seq_k_padded:
        pad = torch.zeros(
            prune_mask.shape[0],
            seq_k_padded - prune_mask.shape[1],
            device=device,
            dtype=prune_mask.dtype,
        )
        prune_mask = torch.cat([prune_mask, pad], dim=1)

    k_t = k_block.transpose(1, 2).contiguous()

    for q_start in range(0, seq_q_real, q_tile):
        q_end = min(q_start + q_tile, seq_q_real)
        indices = torch.arange(q_start, q_end, device=device)
        q_slice = q_block[:, q_start:q_end, :]

        scores = torch.matmul(q_slice, k_t) * scale
        scores = scores.to(torch.float32)

        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        valid_mask = (~future_mask).unsqueeze(0) & key_valid.view(1, 1, -1)

        scores = scores.masked_fill(~valid_mask, float("-inf"))

        if prune_mask is not None:
            prune_slice = prune_mask[q_start:q_end]
            prune_slice = prune_slice.unsqueeze(0)
            scores = scores.masked_fill(~prune_slice, float("-inf"))

        scores_flat = scores.view(head_count, -1, num_k_groups * patch_size)
        row_max = scores_flat.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))

        stable = torch.exp(scores_flat - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (stable / row_sum).view(head_count, -1, num_k_groups, patch_size)

        key_mask = key_valid.view(1, 1, num_k_groups, patch_size)
        weights = weights * key_mask

        pooled_k = weights.max(dim=-1).values

        if return_argmax:
            weights_flat = weights.view(head_count, -1, num_k_groups * patch_size)
            top_indices = weights_flat.argmax(dim=-1, keepdim=True)
            top_mask = torch.zeros_like(weights_flat)
            top_mask.scatter_(dim=-1, index=top_indices, value=1.0)
            top_mask = top_mask.view(head_count, -1, num_k_groups, patch_size)
            top_mask = top_mask * key_mask
            argmax_tile = top_mask.max(dim=-1).values

        if return_query_argmax:
            finite = torch.isfinite(scores)
            has_valid = finite.any(dim=-1)
            local_argmax = scores.argmax(dim=-1)
            local_argmax = torch.where(
                has_valid,
                local_argmax,
                torch.zeros_like(local_argmax),
            )
            tile_len = q_end - q_start
            query_argmax[:, q_start:q_end] = local_argmax[:, :tile_len]

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

        if return_argmax:
            argmax_tile_groups = torch.zeros(
                (head_count, groups_in_tile, num_k_groups),
                device=device,
                dtype=torch.float32,
            )
            argmax_tile_groups.scatter_reduce_(
                dim=1,
                index=expanded_index,
                src=argmax_tile,
                reduce="amax",
            )
            argmax_groups[:, base_group:end_group] = torch.maximum(
                argmax_groups[:, base_group:end_group], argmax_tile_groups
            )

    valid_query_groups = math.ceil(seq_len / patch_size)
    valid_key_groups = math.ceil(seq_len / patch_size)
    if valid_query_groups < num_q_groups:
        pooled_groups[:, valid_query_groups:, :] = 0.0
    if valid_key_groups < num_k_groups:
        pooled_groups[:, :, valid_key_groups:] = 0.0

    if return_argmax and argmax_groups is not None:
        if valid_query_groups < num_q_groups:
            argmax_groups[:, valid_query_groups:, :] = 0.0
        if valid_key_groups < num_k_groups:
            argmax_groups[:, :, valid_key_groups:] = 0.0

    row_min = pooled_groups.amin(dim=2, keepdim=True)
    row_max = pooled_groups.amax(dim=2, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = torch.clamp((pooled_groups - row_min) / denom, 0.0, 1.0)
    return norm, argmax_groups, query_argmax


def simulate_round_pruning(
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    seq_len: int,
    max_keys: int,
    round_window: int,
    offsets: torch.Tensor,
    aggregation: str,
    seed: int,
) -> torch.Tensor:
    device = amp.device
    prune_mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)
    current_cache: List[int] = []

    generator: torch.Generator | None = None
    if max_keys < 0:
        raise ValueError("max_keys must be >= 0")
    if round_window < 1:
        raise ValueError("round_window must be >= 1")
    if max_keys < round_window:
        raise ValueError("max_keys must be >= round_window to reserve new keys")

    if seed is not None:
        if device.type == "cuda":
            generator = torch.Generator(device=device)
        else:
            generator = torch.Generator()
        generator.manual_seed(seed)

    for q_idx in range(seq_len):
        if q_idx % round_window == 0:
            remaining = seq_len - q_idx
            upcoming = min(round_window, remaining)
            keep_capacity = max(0, max_keys - upcoming)
            if keep_capacity <= 0:
                current_cache = []
            elif len(current_cache) > keep_capacity:
                key_tensor = torch.tensor(
                    current_cache,
                    device=device,
                    dtype=torch.long,
                )
                aggregated = score_keys_for_round(
                    key_tensor,
                    q_idx,
                    amp,
                    phi,
                    omega,
                    extra,
                    offsets,
                    aggregation,
                )
                if generator is not None and aggregated.numel() > 0:
                    noise = torch.rand(
                        aggregated.shape,
                        device=device,
                        dtype=aggregated.dtype,
                        generator=generator,
                    ) * 1e-6
                    aggregated = aggregated + noise
                topk = torch.topk(aggregated, k=keep_capacity, largest=True)
                selected = key_tensor.index_select(0, topk.indices)
                current_cache = selected.tolist()
                current_cache.sort()
            else:
                current_cache.sort()

        allowed_keys = [idx for idx in current_cache if idx <= q_idx]
        if allowed_keys:
            idx_tensor = torch.tensor(
                allowed_keys,
                device=device,
                dtype=torch.long,
            )
            prune_mask[q_idx, idx_tensor] = True

        prune_mask[q_idx, q_idx] = True

        if not current_cache or current_cache[-1] != q_idx:
            current_cache.append(q_idx)

    return prune_mask


def score_keys_for_round(
    key_indices: torch.Tensor,
    round_start: int,
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
    offsets: torch.Tensor,
    aggregation: str,
) -> torch.Tensor:
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    base_delta = round_start - key_indices.to(device=amp.device, dtype=torch.float32)
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)

    amp_sel = amp.index_select(0, key_indices)
    phi_sel = phi.index_select(0, key_indices)
    extra_sel = extra.index_select(0, key_indices)

    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi_sel.unsqueeze(1)
    cos_phase = torch.cos(phase)
    base_scores = (amp_sel.unsqueeze(1) * cos_phase).sum(dim=2)
    additive = extra_sel.sum(dim=1, keepdim=True)
    combined = base_scores + additive

    if aggregation == "mean":
        return combined.mean(dim=1)
    return combined.max(dim=1).values


def save_comparison_figure(
    baseline: torch.Tensor,
    pruned: torch.Tensor,
    out_path: Path,
    cmap: str,
    figsize: Tuple[float, float],
    dpi: int,
    title: str,
) -> None:
    baseline_np = baseline.squeeze(0).cpu().numpy()
    pruned_np = pruned.squeeze(0).cpu().numpy()

    vmin = min(baseline_np.min(), pruned_np.min())
    vmax = max(baseline_np.max(), pruned_np.max())

    fig, axes = plt.subplots(
        1,
        2,
        figsize=figsize,
        dpi=dpi,
        sharex=False,
        sharey=True,
        constrained_layout=True,
    )
    ax_base, ax_prune = axes

    im0 = ax_base.imshow(baseline_np, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    ax_base.set_title("Baseline attention")
    ax_base.set_xlabel("Key group index")
    ax_base.set_ylabel("Query group index")

    im1 = ax_prune.imshow(pruned_np, cmap=cmap, aspect="auto", origin="upper", vmin=vmin, vmax=vmax)
    ax_prune.set_title("Pruned attention")
    ax_prune.set_xlabel("Key group index")

    cbar = fig.colorbar(im1, ax=axes.tolist(), fraction=0.046, pad=0.04)
    cbar.set_label("Normalized pooled weight")

    fig.suptitle(title)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path)
    plt.close(fig)


def save_layer_retention_plot(
    per_layer: Dict[int, float],
    out_path: Path,
    dpi: int,
) -> None:
    if not per_layer:
        return

    layers = sorted(per_layer.keys())
    values = [per_layer[layer] for layer in layers]

    fig, ax = plt.subplots(figsize=(12.0, 6.0), dpi=dpi)
    ax.bar(layers, values, color="tab:blue", alpha=0.8)
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean retention")
    ax.set_title("Per-layer baseline argmax retention")
    ax.set_ylim(0.0, 1.0)
    ax.grid(axis="y", linestyle="--", alpha=0.3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_pruning_case")

    trace_dir = args.input_root / args.trace
    if not trace_dir.exists():
        raise SystemExit(f"Trace directory not found: {trace_dir}")
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise SystemExit(f"Missing qk assets in {trace_dir}")

    stats_trace_dir = args.stats_trace or trace_dir
    if not stats_trace_dir.is_absolute():
        stats_trace_dir = (Path.cwd() / stats_trace_dir).resolve()
    if not stats_trace_dir.exists():
        raise SystemExit(f"Stats trace directory not found: {stats_trace_dir}")
    stats_qk_path = stats_trace_dir / "qk.pt"
    stats_meta_path = stats_trace_dir / "metadata.json"
    if not stats_qk_path.exists() or not stats_meta_path.exists():
        raise SystemExit(f"Missing qk assets in stats trace {stats_trace_dir}")

    output_root = args.output_root
    if output_root is None:
        output_root = (
            args.input_root.parent / "llama_pruning_case_studies_xtrace"
        )
    config_dir = f"agg_{args.score_aggregation}_max{args.max_keys}_w{args.round_window}"
    output_root = output_root / trace_dir.name / config_dir
    output_root.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = int(meta["sequence_length"])
    if args.max_seq_len and args.max_seq_len > 0:
        seq_len = min(seq_len, args.max_seq_len)

    with stats_meta_path.open("r", encoding="utf-8") as f:
        stats_meta = json.load(f)
    stats_seq_len = int(stats_meta["sequence_length"])
    if args.max_seq_len and args.max_seq_len > 0:
        stats_seq_len = min(stats_seq_len, args.max_seq_len)

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]
    k_tensor: torch.Tensor = data["k"]
    if args.max_seq_len and args.max_seq_len > 0:
        q_tensor = q_tensor[:, :, :seq_len, :]
        k_tensor = k_tensor[:, :, :seq_len, :]

    stats_data = torch.load(stats_qk_path, map_location="cpu")
    stats_q_tensor: torch.Tensor = stats_data["q"]
    if args.max_seq_len and args.max_seq_len > 0:
        stats_q_tensor = stats_q_tensor[:, :, :stats_seq_len, :]

    layer_count = q_tensor.shape[0]
    head_per_layer = q_tensor.shape[1]
    dtype = DTYPE_MAP[args.dtype]

    device = torch.device(args.device)
    rotary = build_rotary(device, args.model_path, dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    head_dim = q_tensor.shape[-1]
    cos_table, sin_table, inv_freq, _ = compute_rotary_tables(
        rotary, seq_len, head_dim, dtype, device
    )
    freq_count = head_dim // 2
    omega = inv_freq[:freq_count].to(device=device, dtype=torch.float32)
    offsets = build_geometric_offsets(args.offset_max_length, device)

    stats_cos_table, stats_sin_table, _, _ = compute_rotary_tables(
        rotary, stats_seq_len, head_dim, dtype, device
    )

    patch_size = resolve_patch_size(seq_len, args.target_size, args.patch_size)
    patch_size = min(seq_len, max(args.min_patch_size, patch_size))

    sampled_heads = load_or_create_sample(
        args.head_sample_file,
        args.sample_count,
        args.sample_seed,
        layer_count,
        head_per_layer,
    )
    sampled_heads = sorted(sampled_heads)
    if args.verbose:
        print(f"Using {len(sampled_heads)} sampled heads from {args.head_sample_file}")

    layer_to_heads_map: Dict[int, List[int]] = defaultdict(list)
    for layer, head in sampled_heads:
        layer_to_heads_map[layer].append(head)

    q_mean_cache: Dict[Tuple[int, int], torch.Tensor] = {}
    q_abs_mean_cache: Dict[Tuple[int, int], torch.Tensor] = {}

    for layer, heads in layer_to_heads_map.items():
        stats_layer = stats_q_tensor[layer].to(device=device, dtype=dtype)
        for head in heads:
            stats_head_tensor = stats_layer[head, :stats_seq_len, :].contiguous()
            stats_unrot = invert_rope(
                stats_head_tensor,
                stats_cos_table,
                stats_sin_table,
                attention_scale,
            )
            stats_complex = to_complex_pairs(stats_unrot)
            q_mean_complex = stats_complex.mean(dim=0)
            q_abs_mean = torch.abs(stats_complex).mean(dim=0)
            q_mean_cache[(layer, head)] = q_mean_complex.detach().cpu()
            q_abs_mean_cache[(layer, head)] = q_abs_mean.detach().cpu()
        del stats_layer
        if device.type == "cuda":
            torch.cuda.empty_cache()

    retention_records: List[Tuple[int, int, float]] = []
    per_layer_rates: Dict[int, List[float]] = defaultdict(list)

    current_layer = None
    q_layer = None
    k_layer = None

    for layer, head in sampled_heads:
        if layer >= layer_count or head >= head_per_layer:
            raise IndexError(f"Sampled head ({layer}, {head}) exceeds tensor dimensions")

        if layer != current_layer:
            q_layer = q_tensor[layer].to(device=device, dtype=dtype)
            k_layer = k_tensor[layer].to(device=device, dtype=dtype)
            current_layer = layer
            if args.verbose:
                print(f"Loaded layer {layer} to device")

        if args.verbose:
            print(f"Processing layer {layer}, head {head}")

        q_head = q_layer[head, :seq_len, :].contiguous()
        k_head = k_layer[head, :seq_len, :].contiguous()

        q_unrot = invert_rope(q_head, cos_table, sin_table, attention_scale)
        k_unrot = invert_rope(k_head, cos_table, sin_table, attention_scale)

        q_mean_complex = q_mean_cache[(layer, head)].to(device=device)
        q_abs_mean = q_abs_mean_cache[(layer, head)].to(device=device, dtype=torch.float32)
        amp, phi, extra = compute_frequency_statistics_from_means(
            q_mean_complex,
            q_abs_mean,
            k_unrot,
        )
        amp = amp.to(device=device, dtype=torch.float32)
        phi = phi.to(device=device, dtype=torch.float32)
        extra = extra.to(device=device, dtype=torch.float32)

        prune_mask = simulate_round_pruning(
            amp,
            phi,
            omega,
            extra,
            seq_len,
            args.max_keys,
            args.round_window,
            offsets,
            args.score_aggregation,
            args.sample_seed,
        )

        q_block = q_head.unsqueeze(0)
        k_block = k_head.unsqueeze(0)

        baseline_heatmap, baseline_argmax, baseline_query_argmax = compute_pooled_attention(
            q_block,
            k_block,
            seq_len,
            patch_size,
            args.q_tile,
            device,
            prune_mask=None,
            return_argmax=True,
            return_query_argmax=True,
        )
        baseline_heatmap = baseline_heatmap.detach().cpu()
        baseline_argmax = baseline_argmax.detach().cpu() if baseline_argmax is not None else None
        baseline_query_argmax = (
            baseline_query_argmax.detach().cpu()
            if baseline_query_argmax is not None
            else None
        )

        pruned_heatmap, pruned_argmax, _ = compute_pooled_attention(
            q_block,
            k_block,
            seq_len,
            patch_size,
            args.q_tile,
            device,
            prune_mask=prune_mask,
            return_argmax=True,
        )
        pruned_heatmap = pruned_heatmap.detach().cpu()
        pruned_argmax = pruned_argmax.detach().cpu() if pruned_argmax is not None else None

        hit_rate = None
        if baseline_query_argmax is not None:
            indices = baseline_query_argmax[0, :seq_len].clamp(0, seq_len - 1)
            prune_cpu = prune_mask[:seq_len, :seq_len].to(torch.bool).cpu()
            hits = prune_cpu[torch.arange(seq_len), indices]
            hit_rate = hits.to(torch.float32).mean().item()
            retention_records.append((layer, head, hit_rate))
            per_layer_rates[layer].append(hit_rate)

        title = (
            f"Layer {layer:02d} Head {head:02d} "
            f"(max {args.max_keys}, W {args.round_window}, agg {args.score_aggregation})"
        )
        out_path = (
            output_root
            / (
                f"layer_{layer:02d}_head_{head:02d}_"
                f"pruning_comparison_rounds_{args.score_aggregation}.png"
            )
        )
        save_comparison_figure(
            baseline_heatmap,
            pruned_heatmap,
            out_path,
            cmap="inferno",
            figsize=tuple(args.figsize),
            dpi=args.dpi,
            title=title,
        )

        if baseline_argmax is not None and pruned_argmax is not None:
            argmax_path = (
                output_root
                / (
                    f"layer_{layer:02d}_head_{head:02d}_"
                    f"pruning_argmax_rounds_{args.score_aggregation}.png"
                )
            )
            save_comparison_figure(
                baseline_argmax,
                pruned_argmax,
                argmax_path,
                cmap="binary",
                figsize=tuple(args.figsize),
                dpi=args.dpi,
                title=f"Argmax coverage {title}",
            )

        if args.verbose:
            rel = out_path.relative_to(output_root)
            msg = f"Saved comparison figure {rel}"
            if baseline_argmax is not None:
                rel_arg = argmax_path.relative_to(output_root)
                msg += f" and {rel_arg}"
            print(msg)
        if hit_rate is not None:
            print(f"Layer {layer:02d} Head {head:02d} baseline argmax retention rate: {hit_rate:.4f}")

        if device.type == "cuda":
            torch.cuda.empty_cache()

    if device.type == "cuda":
        torch.cuda.empty_cache()

    if not retention_records:
        print("No retention records computed", file=sys.stderr)
        return

    overall_rate = sum(rate for _, _, rate in retention_records) / len(retention_records)
    per_layer_avg = {layer: sum(vals) / len(vals) for layer, vals in per_layer_rates.items()}

    metrics = {
        "overall_retention": overall_rate,
        "head_count": len(retention_records),
        "max_keys": args.max_keys,
        "round_window": args.round_window,
        "score_aggregation": args.score_aggregation,
        "offset_max_length": args.offset_max_length,
        "per_head": [
            {"layer": layer, "head": head, "retention": rate}
            for layer, head, rate in retention_records
        ],
        "per_layer": per_layer_avg,
        "sample_file": str(args.head_sample_file),
        "trace": str(trace_dir),
        "stats_trace": str(stats_trace_dir),
    }

    metrics_path = output_root / "retention_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(f"Overall retention across {len(retention_records)} heads: {overall_rate:.4f}")

    plot_path = output_root / "layer_retention.png"
    save_layer_retention_plot(per_layer_avg, plot_path, dpi=args.dpi)

    if args.verbose:
        print(f"Saved retention metrics to {metrics_path}")


if __name__ == "__main__":
    main()
