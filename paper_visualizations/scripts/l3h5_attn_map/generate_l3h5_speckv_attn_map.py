"""Generate attention map visualization after SpecKV compression for Layer 3, Head 5.

Simplified single-head version of SpecKV round-based KV pruning simulation.
Statistics are computed from the same trace data (no cross-trace stats needed).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants (matching generate_l3h5_attn_map.py) ============
FONT_SIZE = 16
color_dependent = (85 / 250, 104 / 250, 154 / 250)    # blue for dependent
face_color = (231 / 250, 231 / 250, 240 / 250)        # light gray-purple background

attn_cmap_custom = LinearSegmentedColormap.from_list(
    "attn_custom", [face_color, color_dependent]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate L3H5 SpecKV pruned attention map")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument("--patch-size", type=int, default=64, help="Pooling window size")
    parser.add_argument("--q-tile", type=int, default=512, help="Query tile size")
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument("--output-path", type=Path, default=None, help="Output path")
    parser.add_argument("--layer", type=int, default=3, help="Layer index")
    parser.add_argument("--head", type=int, default=5, help="Head index")
    # SpecKV parameters
    parser.add_argument("--kv-budget", type=int, default=2048, help="Max KV cache size")
    parser.add_argument("--round-window", type=int, default=128, help="Tokens per pruning round")
    parser.add_argument("--offset-max-length", type=int, default=65536, help="Max offset for geometric grid")
    parser.add_argument("--score-aggregation", choices=["mean", "max"], default="mean", help="Score aggregation method")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for tie-breaking")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model path for RoPE parameters",
    )
    parser.add_argument("--side-by-side", action="store_true", help="Show baseline and pruned side by side")
    return parser.parse_args()


# ============ RoPE Inversion Utilities ============

def rotate_half(x: torch.Tensor) -> torch.Tensor:
    d = x.shape[-1] // 2
    return torch.cat((-x[..., d:], x[..., :d]), dim=-1)


def invert_rope(
    rotated: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    scale: float,
) -> torch.Tensor:
    """Invert YaRN-scaled RoPE: recovers x from y = scale * (x * cos + rotate_half(x) * sin)."""
    if scale == 0:
        raise ValueError("attention scaling factor must be non-zero")
    base = rotated / scale
    return base * cos - rotate_half(base) * sin


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Convert [seq, head_dim] to [seq, head_dim/2] complex tensor."""
    freq_count = tensor.shape[-1] // 2
    real = tensor[..., :freq_count].float()
    imag = tensor[..., freq_count:].float()
    return torch.complex(real, imag)


# ============ SpecKV Scoring Functions ============

def build_geometric_offsets(max_length: int, device: torch.device) -> torch.Tensor:
    """Build geometric offset grid: [1, 2, 4, 8, ..., max_length]."""
    offsets = []
    value = 1
    while value <= max_length:
        offsets.append(float(value))
        value *= 2
    return torch.tensor(offsets, device=device, dtype=torch.float32)


def compute_frequency_statistics(
    q_unrot: torch.Tensor,
    k_unrot: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute frequency-domain statistics for SpecKV scoring.

    Args:
        q_unrot: [seq_len, head_dim] unrotated Q vectors
        k_unrot: [seq_len, head_dim] unrotated K vectors

    Returns:
        amp: [seq_len, freq_count] amplitude scores
        phi: [seq_len, freq_count] phase angles
        extra: [seq_len, freq_count] additional amplitude term
        q_mean_complex: [freq_count] mean Q in complex form
    """
    q_complex = to_complex_pairs(q_unrot)
    k_complex = to_complex_pairs(k_unrot)

    # Compute statistics from Q vectors
    q_mean_complex = q_complex.mean(dim=0)
    q_mean_abs = torch.abs(q_mean_complex)
    q_abs_mean = torch.abs(q_complex).mean(dim=0)

    # Compute K-dependent scores
    k_abs = torch.abs(k_complex)
    relative = q_mean_complex.unsqueeze(0) * torch.conj(k_complex)
    phi = torch.atan2(relative.imag, relative.real)
    amp = q_mean_abs.unsqueeze(0) * k_abs
    extra = (q_abs_mean - q_mean_abs).unsqueeze(0) * k_abs

    return amp, phi, extra, q_mean_complex


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
    """Score keys for a pruning round using frequency-based prediction."""
    if key_indices.numel() == 0:
        return torch.empty(0, device=amp.device, dtype=torch.float32)

    # Delta = offset between query position and key positions
    base_delta = round_start - key_indices.float()
    delta_grid = base_delta.unsqueeze(1) + offsets.unsqueeze(0)  # [num_keys, num_offsets]

    # Select amplitude and phase for these keys
    amp_sel = amp.index_select(0, key_indices)      # [num_keys, freq_count]
    phi_sel = phi.index_select(0, key_indices)      # [num_keys, freq_count]
    extra_sel = extra.index_select(0, key_indices)  # [num_keys, freq_count]

    # Predict future attention: sum over frequencies of amp * cos(omega * delta + phi)
    # [num_keys, num_offsets, freq_count]
    phase = delta_grid.unsqueeze(2) * omega.view(1, 1, -1) + phi_sel.unsqueeze(1)
    cos_phase = torch.cos(phase)
    base_scores = (amp_sel.unsqueeze(1) * cos_phase).sum(dim=2)  # [num_keys, num_offsets]
    additive = extra_sel.sum(dim=1, keepdim=True)  # [num_keys, 1]
    combined = base_scores + additive  # [num_keys, num_offsets]

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
    """Simulate SpecKV round-based pruning and return the pruning mask.

    Returns:
        prune_mask: [seq_len, seq_len] boolean tensor where True means key is retained
    """
    device = amp.device
    prune_mask = torch.zeros((seq_len, seq_len), device=device, dtype=torch.bool)
    current_cache: List[int] = []

    generator = torch.Generator(device=device if device.type == "cuda" else "cpu")
    generator.manual_seed(seed)

    for q_idx in range(seq_len):
        # At round boundaries, prune the cache
        if q_idx % round_window == 0:
            remaining = seq_len - q_idx
            upcoming = min(round_window, remaining)
            keep_capacity = max(0, max_keys - upcoming)

            if keep_capacity <= 0:
                current_cache = []
            elif len(current_cache) > keep_capacity:
                key_tensor = torch.tensor(current_cache, device=device, dtype=torch.long)
                aggregated = score_keys_for_round(
                    key_tensor, q_idx, amp, phi, omega, extra, offsets, aggregation
                )
                # Add tiny noise for reproducible tie-breaking
                noise = torch.rand(aggregated.shape, device=device, dtype=aggregated.dtype, generator=generator) * 1e-6
                aggregated = aggregated + noise
                topk = torch.topk(aggregated, k=keep_capacity, largest=True)
                selected = key_tensor.index_select(0, topk.indices)
                current_cache = sorted(selected.tolist())

        # Mark retained keys as visible
        allowed_keys = [idx for idx in current_cache if idx <= q_idx]
        if allowed_keys:
            idx_tensor = torch.tensor(allowed_keys, device=device, dtype=torch.long)
            prune_mask[q_idx, idx_tensor] = True

        # Current position is always visible (diagonal)
        prune_mask[q_idx, q_idx] = True

        # Add current position to cache
        if not current_cache or current_cache[-1] != q_idx:
            current_cache.append(q_idx)

    return prune_mask


# ============ Attention Computation ============

def compute_attention_heatmap(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
    prune_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Compute pooled attention heatmap for a single head."""
    seq_q, head_dim = q_block.shape
    scale = head_dim ** -0.5

    num_q_groups = math.ceil(seq_len / patch_size)
    num_k_groups = math.ceil(seq_len / patch_size)

    # Padding
    q_pad = num_q_groups * patch_size - seq_len
    k_pad = num_k_groups * patch_size - seq_len
    if q_pad > 0:
        q_block = torch.cat([q_block, torch.zeros(q_pad, head_dim, device=device, dtype=q_block.dtype)], dim=0)
    if k_pad > 0:
        k_block = torch.cat([k_block, torch.zeros(k_pad, head_dim, device=device, dtype=k_block.dtype)], dim=0)

    seq_k_padded = k_block.shape[0]
    key_positions = torch.arange(seq_k_padded, device=device)
    key_valid = key_positions < seq_len

    # Pad prune_mask if needed
    if prune_mask is not None and prune_mask.shape[1] < seq_k_padded:
        pad = torch.zeros(prune_mask.shape[0], seq_k_padded - prune_mask.shape[1], device=device, dtype=prune_mask.dtype)
        prune_mask = torch.cat([prune_mask, pad], dim=1)

    pooled_groups = torch.zeros((num_q_groups, num_k_groups), device=device, dtype=torch.float32)
    k_t = k_block.t().contiguous()

    for q_start in range(0, seq_len, q_tile):
        q_end = min(q_start + q_tile, seq_len)
        indices = torch.arange(q_start, q_end, device=device)
        q_slice = q_block[q_start:q_end]

        scores = torch.matmul(q_slice, k_t) * scale
        scores = scores.to(torch.float32)

        # Causal mask
        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        scores = scores.masked_fill(future_mask, float("-inf"))
        scores = scores.masked_fill(~key_valid.view(1, -1), float("-inf"))

        # Apply pruning mask
        if prune_mask is not None:
            prune_slice = prune_mask[q_start:q_end]
            scores = scores.masked_fill(~prune_slice, float("-inf"))

        # Softmax
        scores_flat = scores.view(-1, num_k_groups * patch_size)
        row_max = scores_flat.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        stable = torch.exp(scores_flat - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = (stable / row_sum).view(-1, num_k_groups, patch_size)

        key_mask = key_valid.view(1, num_k_groups, patch_size)
        weights = weights * key_mask

        # Max pool over key dimension
        pooled_k = weights.max(dim=-1).values

        # Aggregate into query groups
        query_groups = indices // patch_size
        for local_q, global_q_group in enumerate(query_groups.tolist()):
            pooled_groups[global_q_group] = torch.maximum(
                pooled_groups[global_q_group], pooled_k[local_q]
            )

    # Normalize per row
    valid_q_groups = math.ceil(seq_len / patch_size)
    valid_k_groups = math.ceil(seq_len / patch_size)
    pooled_groups = pooled_groups[:valid_q_groups, :valid_k_groups]

    row_min = pooled_groups.amin(dim=1, keepdim=True)
    row_max = pooled_groups.amax(dim=1, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_groups - row_min) / denom
    return norm.detach().cpu()


# ============ RoPE Setup ============

def build_rotary_tables(
    model_path: Path,
    seq_len: int,
    head_dim: int,
    device: torch.device,
    dtype: torch.dtype,
):
    """Build RoPE cos/sin tables and get inv_freq."""
    from transformers import AutoConfig
    try:
        from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding
    except ImportError:
        from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding as Qwen3RotaryEmbedding

    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    rope_scaling = dict(config.rope_scaling or {})
    if "attn_factor" in rope_scaling and "attention_factor" not in rope_scaling:
        rope_scaling["attention_factor"] = rope_scaling["attn_factor"]
    rope_scaling.pop("attn_factor", None)
    if "rope_type" not in rope_scaling:
        rope_scaling["rope_type"] = rope_scaling.get("type", "default")
    rope_scaling.pop("type", None)
    config.rope_scaling = rope_scaling

    rotary = Qwen3RotaryEmbedding(config=config, device=device)
    rotary.to(dtype=dtype)

    position_ids = torch.arange(seq_len, device=device).unsqueeze(0)
    base = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(base, position_ids)
    cos_table = cos_table[0]
    sin_table = sin_table[0]
    inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float64)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))

    return cos_table, sin_table, inv_freq, attention_scale


# ============ Main Visualization ============

def generate_figure(
    trace_dir: Path,
    device: torch.device,
    patch_size: int,
    q_tile: int,
    dpi: int,
    output_path: Path,
    layer: int,
    head: int,
    kv_budget: int,
    round_window: int,
    offset_max_length: int,
    score_aggregation: str,
    seed: int,
    model_path: Path,
    side_by_side: bool,
) -> None:
    """Generate the SpecKV pruned attention map figure."""
    mask_process_command("PD-L1_binder_speckv_viz")

    # Load data
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")
    print(f"SpecKV config: kv_budget={kv_budget}, round_window={round_window}")

    # Load Q/K tensors
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]
    head_dim = q_tensor.shape[-1]

    print(f"Processing Layer {layer}, Head {head}...")
    q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
    k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

    # Build RoPE tables
    print("Building RoPE tables...")
    cos_table, sin_table, inv_freq, attention_scale = build_rotary_tables(
        model_path, token_count, head_dim, device, torch.float32
    )
    freq_count = head_dim // 2
    omega = inv_freq[:freq_count].to(device=device, dtype=torch.float32)
    offsets = build_geometric_offsets(offset_max_length, device)

    # Invert RoPE to get unrotated Q/K
    print("Inverting RoPE...")
    q_unrot = invert_rope(q_block, cos_table, sin_table, attention_scale)
    k_unrot = invert_rope(k_block, cos_table, sin_table, attention_scale)

    # Compute frequency statistics
    print("Computing frequency statistics...")
    amp, phi, extra, _ = compute_frequency_statistics(q_unrot, k_unrot)

    # Simulate round-based pruning
    print("Simulating SpecKV pruning...")
    prune_mask = simulate_round_pruning(
        amp, phi, omega, extra, token_count,
        kv_budget, round_window, offsets, score_aggregation, seed
    )

    # Count retained keys
    retained_per_query = prune_mask.sum(dim=1).float()
    avg_retained = retained_per_query.mean().item()
    print(f"Average retained keys per query: {avg_retained:.1f}")

    # Compute attention maps
    print("Computing attention heatmaps...")
    with torch.no_grad():
        pruned_heatmap = compute_attention_heatmap(
            q_block, k_block, token_count, patch_size, q_tile, device, prune_mask
        )

        if side_by_side:
            baseline_heatmap = compute_attention_heatmap(
                q_block, k_block, token_count, patch_size, q_tile, device, None
            )

    # Create figure
    if side_by_side:
        fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=dpi)
        ax_base, ax_prune = axes

        ax_base.imshow(baseline_heatmap.numpy(), cmap=attn_cmap_custom, aspect="equal", origin="upper")
        ax_base.set_xticks([])
        ax_base.set_yticks([])
        for spine in ax_base.spines.values():
            spine.set_visible(False)
        ax_base.set_title(f"L{layer}H{head} Full KV", fontsize=FONT_SIZE, fontweight='bold')

        ax_prune.imshow(pruned_heatmap.numpy(), cmap=attn_cmap_custom, aspect="equal", origin="upper")
        ax_prune.set_xticks([])
        ax_prune.set_yticks([])
        for spine in ax_prune.spines.values():
            spine.set_visible(False)
        ax_prune.set_title(f"L{layer}H{head} SpecKV (budget={kv_budget})", fontsize=FONT_SIZE, fontweight='bold')

        plt.tight_layout()
    else:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)
        ax.imshow(pruned_heatmap.numpy(), cmap=attn_cmap_custom, aspect="equal", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)
        ax.set_title(f"L{layer}H{head} SpecKV (budget={kv_budget})", fontsize=FONT_SIZE, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


def main() -> None:
    args = parse_args()

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/l3h5_attn_map")
        suffix = "_comparison" if args.side_by_side else ""
        output_path = output_dir / f"fig_l{args.layer}h{args.head}_speckv_budget{args.kv_budget}{suffix}.png"
    else:
        output_path = args.output_path

    device = torch.device(args.device)

    generate_figure(
        trace_dir=args.trace_dir,
        device=device,
        patch_size=args.patch_size,
        q_tile=args.q_tile,
        dpi=args.dpi,
        output_path=output_path,
        layer=args.layer,
        head=args.head,
        kv_budget=args.kv_budget,
        round_window=args.round_window,
        offset_max_length=args.offset_max_length,
        score_aggregation=args.score_aggregation,
        seed=args.seed,
        model_path=args.model_path,
        side_by_side=args.side_by_side,
    )


if __name__ == "__main__":
    main()
