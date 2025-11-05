"""Compare baseline attention with hybrid frequency-based key pruning for selected heads."""
from __future__ import annotations

import argparse
import json
import math
import sys
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
from weian_development.attention_qk_analysis.freq_magnitude_single_plot_meanvec_randomk import (
    to_complex_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize baseline attention vs. hybrid frequency scoring key pruning case studies.",
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
        help="Base directory for outputs (defaults to <input_root>/../attention_pruning_case_studies_hybrid).",
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
        "--keep-keys",
        type=int,
        default=1024,
        help="Number of top-scoring keys to retain per query during pruning.",
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


def parse_layer_head_spec(specs: Sequence[str]) -> Dict[int, List[int]]:
    grouped: Dict[int, List[int]] = {}
    for spec in specs:
        if ":" not in spec:
            raise ValueError(f"Invalid layer/head spec: {spec}")
        layer_str, head_str = spec.split(":", 1)
        layer = int(layer_str)
        head = int(head_str)
        grouped.setdefault(layer, []).append(head)
    return grouped


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


def compute_frequency_statistics(
    q_unrot: torch.Tensor,
    k_unrot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    q_complex = to_complex_pairs(q_unrot)
    k_complex = to_complex_pairs(k_unrot)
    q_mean = q_complex.mean(dim=0)
    q_mean_abs = torch.abs(q_mean)
    q_abs_mean = torch.abs(q_complex).mean(dim=0)
    k_abs = torch.abs(k_complex)
    relative = q_mean.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs
    return amp, phi, extra


def compute_pruning_scores(
    amp: torch.Tensor,
    phi: torch.Tensor,
    omega: torch.Tensor,
    extra: torch.Tensor,
) -> torch.Tensor:
    seq_len, freq_count = amp.shape
    scores = torch.full((seq_len, seq_len), float("-inf"), device=amp.device, dtype=torch.float32)
    for q_idx in range(seq_len):
        valid_count = q_idx + 1
        if valid_count == 0:
            continue
        key_idx = torch.arange(valid_count, device=amp.device, dtype=torch.float32)
        delta = (float(q_idx) - key_idx)
        phase = delta.unsqueeze(1) * omega.unsqueeze(0) + phi[:valid_count]
        cos_phase = torch.cos(phase)
        base_scores = (amp[:valid_count] * cos_phase).sum(dim=1)
        additive = extra[:valid_count].sum(dim=1)
        scores[q_idx, :valid_count] = base_scores + additive
    return scores


def build_keep_mask(
    scores: torch.Tensor,
    keep_keys: int,
    seed: int,
) -> torch.Tensor:
    if keep_keys <= 0:
        return torch.zeros_like(scores, dtype=torch.bool)

    seq_len = scores.size(0)
    keep_mask = torch.zeros_like(scores, dtype=torch.bool)
    max_keep = min(keep_keys, seq_len)

    scores_topk = scores.clone()
    invalid_mask = ~torch.isfinite(scores_topk)
    scores_topk[invalid_mask] = float("-inf")

    if max_keep < seq_len:
        generator = torch.Generator(device=scores.device)
        generator.manual_seed(seed)
        noise = torch.rand(
            scores_topk.shape,
            device=scores.device,
            dtype=scores_topk.dtype,
            generator=generator,
        ) * 1e-6
        noise = torch.where(invalid_mask, torch.zeros_like(noise), noise)
        scores_topk = scores_topk + noise

    topk_vals, topk_idx = torch.topk(scores_topk, k=max_keep, dim=1)
    finite_mask = torch.isfinite(topk_vals)
    keep_mask.scatter_(1, topk_idx, finite_mask)
    return keep_mask


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
        output_root = args.input_root.parent / "attention_pruning_case_studies_hybrid"
    output_root = output_root / trace_dir.name
    output_root.mkdir(parents=True, exist_ok=True)

    layer_to_heads = parse_layer_head_spec(args.layer_heads)

    with meta_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    seq_len = int(meta["sequence_length"])

    patch_size = resolve_patch_size(seq_len, args.target_size, args.patch_size)

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    data = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = data["q"]
    k_tensor: torch.Tensor = data["k"]

    rotary = build_rotary(device, args.model_path, dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    head_dim = q_tensor.shape[-1]
    cos_table, sin_table, inv_freq = compute_rotary_tables(
        rotary, seq_len, head_dim, dtype, device
    )
    freq_count = head_dim // 2
    omega = inv_freq[:freq_count].to(device=device, dtype=torch.float32)

    for layer, head_list in layer_to_heads.items():
        if layer >= q_tensor.shape[0]:
            raise IndexError(f"Layer index {layer} exceeds captured tensor dimensions")

        q_layer = q_tensor[layer].to(device=device, dtype=dtype)
        k_layer = k_tensor[layer].to(device=device, dtype=dtype)

        for head in head_list:
            if head >= q_layer.shape[0]:
                raise IndexError(
                    f"Head index {head} exceeds captured tensor dimensions for layer {layer}"
                )

            if args.verbose:
                print(f"Processing layer {layer}, head {head}")

            q_head = q_layer[head, :seq_len, :].contiguous()
            k_head = k_layer[head, :seq_len, :].contiguous()

            q_unrot, k_unrot = invert_qk(q_head, k_head, cos_table, sin_table, attention_scale)

            amp, phi, extra = compute_frequency_statistics(q_unrot, k_unrot)
            amp = amp.to(device=device, dtype=torch.float32)
            phi = phi.to(device=device, dtype=torch.float32)
            extra = extra.to(device=device, dtype=torch.float32)

            scores = compute_pruning_scores(amp, phi, omega, extra)
            prune_mask = build_keep_mask(scores, args.keep_keys, args.seed)

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

            title = f"Layer {layer:02d} Head {head:02d} (keep {args.keep_keys})"
            out_path = (
                output_root
                / f"layer_{layer:02d}_head_{head:02d}_pruning_comparison_hybrid.png"
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
                    / f"layer_{layer:02d}_head_{head:02d}_pruning_argmax_hybrid.png"
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


if __name__ == "__main__":
    main()
