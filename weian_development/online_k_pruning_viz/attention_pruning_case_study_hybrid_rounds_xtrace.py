"""Round-based hybrid frequency key pruning across sampled heads with cross-trace stats."""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from collections import defaultdict
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
from weian_development.attention_qk_analysis.freq_magnitude_single_plot_meanvec_randomk import (
    to_complex_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize hybrid frequency scoring on sampled heads using cross-trace statistics.",
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
        "--head-sample-file",
        type=Path,
        default=Path("weian_development/online_k_pruning_viz/hybrid_sample_heads.json"),
        help="Path to JSON file storing sampled (layer, head) indices (generated if missing).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of heads to sample when creating a new sample file.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used when sampling heads for a missing sample file.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="Base directory for outputs (defaults to <input_root>/../attention_pruning_case_studies_hybrid_rounds_xtrace).",
    )
    parser.add_argument(
        "--stats-trace",
        type=Path,
        required=True,
        help="Trace directory (containing qk.pt) used to compute cross-trace frequency statistics.",
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
        help="Number of queries to process per tile when computing attention logits.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device for attention computation (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE parameters.",
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
        default=(20.48, 10.24),
        help="Figure size (inches) for the side-by-side attention heatmaps.",
    )
    parser.add_argument(
        "--max-keys",
        "--keep-keys",
        dest="max_keys",
        type=int,
        default=2048,
        help="Maximum number of cached keys to retain after each maintenance round (alias: --keep-keys).",
    )
    parser.add_argument(
        "--round-window",
        type=int,
        default=64,
        help="Number of decoded tokens per cache maintenance round.",
    )
    parser.add_argument(
        "--offset-max-length",
        type=int,
        default=65536,
        help="Maximum offset evaluated in the geometric future offset grid (inclusive).",
    )
    parser.add_argument(
        "--score-aggregation",
        choices=["mean", "max"],
        default="mean",
        help="Aggregator used to collapse per-offset scores when ranking keys (mean or max).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for reproducible tie-breaking in top-k selection.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Log detailed progress information.",
    )
    return parser.parse_args()


DTYPE_MAP = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def load_or_create_sample(
    sample_file: Path,
    sample_count: int,
    seed: int,
    layer_count: int,
    head_count: int,
) -> List[Tuple[int, int]]:
    if sample_file.exists():
        with sample_file.open("r", encoding="utf-8") as f:
            data = json.load(f)
        sample = [(int(item[0]), int(item[1])) for item in data]
        return sample

    total_heads = layer_count * head_count
    if sample_count > total_heads:
        raise ValueError(
            f"Sample count {sample_count} exceeds total available heads {total_heads}"
        )

    indices = [(layer, head) for layer in range(layer_count) for head in range(head_count)]
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    perm = torch.randperm(len(indices), generator=generator)
    selected = [indices[idx] for idx in perm[:sample_count].tolist()]

    sample_file.parent.mkdir(parents=True, exist_ok=True)
    with sample_file.open("w", encoding="utf-8") as f:
        json.dump([[layer, head] for layer, head in selected], f, indent=2)

    return selected


def resolve_patch_size(seq_len: int, target_size: int, patch_arg: int | None) -> int:
    if patch_arg and patch_arg > 0:
        return patch_arg
    if seq_len <= target_size:
        return 1
    return math.ceil(seq_len / target_size)


def build_rotary(cache_device: torch.device, model_path: Path, dtype: torch.dtype) -> Qwen3RotaryEmbedding:
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    config.rope_scaling = rope_scaling
    rotary = Qwen3RotaryEmbedding(config=config, device=cache_device)
    rotary.to(dtype=dtype)
    return rotary


def compute_rotary_tables(
    rotary: Qwen3RotaryEmbedding,
    seq_len: int,
    head_dim: int,
    dtype: torch.dtype,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0]
    sin_table = sin_table[0]
    inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float64)
    return cos_table, sin_table, inv_freq


def invert_qk(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    cos_table: torch.Tensor,
    sin_table: torch.Tensor,
    attention_scale: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    q_unrot = invert_rope(q_head, cos_table, sin_table, attention_scale)
    k_unrot = invert_rope(k_head, cos_table, sin_table, attention_scale)
    return q_unrot, k_unrot


def compute_frequency_statistics_from_means(
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    k_unrot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    k_complex = to_complex_pairs(k_unrot)
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    if max_length < 1:
        raise ValueError("offset_max_length must be >= 1")
    offsets: List[float] = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


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

    base_delta = (
        round_start
        - key_indices.to(device=amp.device, dtype=torch.float32)
    )
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

    output_root = args.output_root
    if output_root is None:
        output_root = (
            args.input_root.parent / "attention_pruning_case_studies_hybrid_rounds_xtrace"
        )
    config_dir = f"agg_{args.score_aggregation}_max{args.max_keys}_w{args.round_window}"
    output_root = output_root / trace_dir.name / config_dir
    output_root.mkdir(parents=True, exist_ok=True)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = int(meta["sequence_length"])

    patch_size = resolve_patch_size(seq_len, args.target_size, args.patch_size)

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]
    k_tensor: torch.Tensor = data["k"]

    layer_count = q_tensor.shape[0]
    head_per_layer = q_tensor.shape[1]

    stats_trace_dir = args.stats_trace
    if not stats_trace_dir.is_absolute():
        stats_trace_dir = (Path.cwd() / stats_trace_dir).resolve()
    if not stats_trace_dir.exists():
        raise SystemExit(f"Stats trace directory not found: {stats_trace_dir}")

    stats_qk_path = stats_trace_dir / "qk.pt"
    stats_meta_path = stats_trace_dir / "metadata.json"
    if not stats_qk_path.exists() or not stats_meta_path.exists():
        raise SystemExit(f"Missing qk assets in stats trace {stats_trace_dir}")

    with stats_meta_path.open("r", encoding="utf-8") as f:
        stats_meta = json.load(f)
    stats_seq_len = int(stats_meta["sequence_length"])

    stats_data = torch.load(stats_qk_path, map_location="cpu")
    stats_q_tensor: torch.Tensor = stats_data["q"]
    if stats_q_tensor.shape[0] != layer_count or stats_q_tensor.shape[1] != head_per_layer:
        raise SystemExit("Stats trace dimensions do not match primary trace dimensions")

    sampled_heads = load_or_create_sample(
        args.head_sample_file,
        args.sample_count,
        args.sample_seed,
        layer_count,
        head_per_layer,
    )

    sampled_heads = sorted(sampled_heads)
    if args.verbose:
        print(
            f"Using {len(sampled_heads)} sampled heads from {args.head_sample_file}"  # noqa: E501
        )

    retention_records: List[Tuple[int, int, float]] = []
    per_layer_rates: Dict[int, List[float]] = defaultdict(list)

    rotary = build_rotary(device, args.model_path, dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    head_dim = q_tensor.shape[-1]
    cos_table, sin_table, inv_freq = compute_rotary_tables(
        rotary, seq_len, head_dim, dtype, device
    )
    freq_count = head_dim // 2
    omega = inv_freq[:freq_count].to(device=device, dtype=torch.float32)
    offsets = build_geometric_offsets(args.offset_max_length, device)

    stats_cos_table, stats_sin_table, _ = compute_rotary_tables(
        rotary, stats_seq_len, head_dim, dtype, device
    )

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

    current_layer = None
    q_layer = None
    k_layer = None

    for layer, head in sampled_heads:
        if layer >= layer_count or head >= head_per_layer:
            raise IndexError(
                f"Sampled head ({layer}, {head}) exceeds tensor dimensions"
            )

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

        q_unrot, k_unrot = invert_qk(q_head, k_head, cos_table, sin_table, attention_scale)

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
            args.seed,
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
            print(
                f"Layer {layer:02d} Head {head:02d} baseline argmax retention rate: {hit_rate:.4f}"
            )

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
        print(f"Saved layer retention plot to {plot_path}")


if __name__ == "__main__":
    main()
