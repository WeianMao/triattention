"""RKV Attention Miss Analysis Visualization.

Simulates RKV's attention-score-based key selection (without similarity removal)
and visualizes which attention positions would be "missed" (key evicted before being attended to).

For L17H25 (Retrieval Attention), this demonstrates the limitation of history-based
attention score selection for irregular/retrieval-style attention patterns.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Set, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Patch
import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 14
color_dependent = (85 / 250, 104 / 250, 154 / 250)    # blue for normal attention
color_miss = (0.85, 0.2, 0.2)                          # red for missed attention
face_color = (231 / 250, 231 / 250, 240 / 250)

# Custom colormaps
attn_cmap_normal = LinearSegmentedColormap.from_list(
    "attn_normal", [face_color, color_dependent]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="RKV attention miss analysis")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--layer", type=int, default=17, help="Layer index")
    parser.add_argument("--head", type=int, default=25, help="Head index")
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument(
        "--budget",
        type=int,
        default=2048,
        help="RKV budget (max keys to keep)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=8,
        help="RKV window size (recent queries for attention score)",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=7,
        help="RKV max pooling kernel size",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=64,
        help="Pooling window size for visualization",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for figure",
    )
    return parser.parse_args()


def compute_rkv_attention_scores(
    q_window: torch.Tensor,
    k_candidates: torch.Tensor,
    kernel_size: int,
) -> torch.Tensor:
    """
    Compute RKV-style attention scores for key selection.

    Args:
        q_window: [window_size, head_dim] recent queries
        k_candidates: [num_candidates, head_dim] candidate keys
        kernel_size: max pooling kernel size

    Returns:
        scores: [num_candidates] attention score for each candidate key
    """
    head_dim = q_window.shape[-1]
    scale = head_dim ** -0.5

    # Compute attention: [window_size, num_candidates]
    attn_logits = torch.matmul(q_window, k_candidates.t()) * scale

    # Softmax over keys for each query
    attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32)

    # Mean over query window: [num_candidates]
    attn_mean = attn_weights.mean(dim=0)

    # Max pooling
    num_candidates = attn_mean.shape[0]
    if num_candidates < kernel_size:
        return attn_mean

    attn_mean_3d = attn_mean.unsqueeze(0).unsqueeze(0)  # [1, 1, num_candidates]
    attn_pooled = F.max_pool1d(
        attn_mean_3d,
        kernel_size=kernel_size,
        padding=kernel_size // 2,
        stride=1,
    ).squeeze()  # [num_candidates]

    return attn_pooled


def simulate_rkv_and_detect_miss(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    budget: int,
    window_size: int,
    kernel_size: int,
    device: torch.device,
) -> Tuple[List[Tuple[int, int]], dict]:
    """
    Simulate RKV compression and detect miss positions.

    Returns:
        miss_positions: List of (query_pos, key_pos) tuples where miss occurred
        stats: Dictionary with statistics
    """
    head_dim = q_block.shape[-1]
    scale = head_dim ** -0.5

    # Track which original positions are still alive
    # Use a list to maintain order for efficient indexing
    alive_positions: List[int] = list(range(seq_len))  # Initially all alive
    alive_set: Set[int] = set(alive_positions)  # For O(1) lookup

    miss_positions: List[Tuple[int, int]] = []
    total_queries_after_budget = 0

    print(f"Simulating RKV compression (budget={budget}, window={window_size})...")

    for t in range(seq_len):
        if t % 2000 == 0:
            print(f"  Processing position {t}/{seq_len}, alive_keys={len(alive_set)}")

        # Step 1: Check if compression is needed
        # Note: In real RKV, compression happens when cache exceeds budget
        # We simulate this by checking current alive count
        current_cache_size = len(alive_set)

        if current_cache_size > budget and t >= window_size:
            # Need to compress
            # Candidate keys: all alive keys except recent window
            # Recent window: positions [t - window_size, t - 1]
            recent_window = set(range(max(0, t - window_size), t))
            candidate_positions = [p for p in alive_positions if p not in recent_window and p < t]

            if len(candidate_positions) > 0:
                num_to_keep = budget - window_size
                if num_to_keep < len(candidate_positions):
                    # Get Q window and K candidates
                    q_start = max(0, t - window_size)
                    q_window = q_block[q_start:t].to(device)  # [window_size, head_dim]

                    # Gather candidate keys by their original positions
                    candidate_indices = torch.tensor(candidate_positions, dtype=torch.long)
                    k_candidates = k_block[candidate_indices].to(device)  # [num_candidates, head_dim]

                    # Compute attention scores
                    scores = compute_rkv_attention_scores(q_window, k_candidates, kernel_size)

                    # Select top-k
                    _, top_indices = scores.topk(min(num_to_keep, len(candidate_positions)), dim=-1)
                    keep_positions_set = {candidate_positions[i.item()] for i in top_indices}

                    # Evict keys not in top-k
                    evicted = set(candidate_positions) - keep_positions_set
                    alive_set -= evicted

                    # Update alive_positions list
                    alive_positions = [p for p in alive_positions if p in alive_set]

        # Step 2: Compute current query's attention and find argmax
        if t > 0:
            # Full attention over all keys up to t (for finding argmax)
            q_t = q_block[t:t+1].to(device)  # [1, head_dim]
            k_all = k_block[:t].to(device)  # [t, head_dim]

            attn_logits = torch.matmul(q_t, k_all.t()) * scale  # [1, t]
            attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).squeeze()  # [t]

            argmax_key = attn_weights.argmax().item()

            # Step 3: Check if argmax key is alive
            if t > budget:
                total_queries_after_budget += 1
                if argmax_key not in alive_set:
                    miss_positions.append((t, argmax_key))

    stats = {
        'total_queries_after_budget': total_queries_after_budget,
        'total_miss': len(miss_positions),
        'miss_rate': len(miss_positions) / max(total_queries_after_budget, 1),
        'final_alive_count': len(alive_set),
    }

    return miss_positions, stats


def simulate_keynorm_and_detect_miss(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    budget: int,
    window_size: int,
    device: torch.device,
) -> Tuple[List[Tuple[int, int]], dict]:
    """
    Simulate Key-Norm based compression and detect miss positions.

    Selection criterion: Keep keys with largest L2 norm (discard smallest).
    Uses same budget and window_size as RKV for fair comparison.

    Returns:
        miss_positions: List of (query_pos, key_pos) tuples where miss occurred
        stats: Dictionary with statistics
    """
    head_dim = q_block.shape[-1]
    scale = head_dim ** -0.5

    # Track which original positions are still alive
    alive_positions: List[int] = list(range(seq_len))
    alive_set: Set[int] = set(alive_positions)

    miss_positions: List[Tuple[int, int]] = []
    total_queries_after_budget = 0

    print(f"Simulating Key-Norm compression (budget={budget}, window={window_size})...")

    for t in range(seq_len):
        if t % 2000 == 0:
            print(f"  Processing position {t}/{seq_len}, alive_keys={len(alive_set)}")

        current_cache_size = len(alive_set)

        if current_cache_size > budget and t >= window_size:
            # Need to compress
            recent_window = set(range(max(0, t - window_size), t))
            candidate_positions = [p for p in alive_positions if p not in recent_window and p < t]

            if len(candidate_positions) > 0:
                num_to_keep = budget - window_size
                if num_to_keep < len(candidate_positions):
                    # Gather candidate keys
                    candidate_indices = torch.tensor(candidate_positions, dtype=torch.long)
                    k_candidates = k_block[candidate_indices].to(device)  # [num_candidates, head_dim]

                    # Compute key norms (L2 norm)
                    key_norms = k_candidates.norm(dim=-1)  # [num_candidates]

                    # Select top-k by norm (keep largest)
                    _, top_indices = key_norms.topk(min(num_to_keep, len(candidate_positions)), dim=-1)
                    keep_positions_set = {candidate_positions[i.item()] for i in top_indices}

                    # Evict keys not in top-k
                    evicted = set(candidate_positions) - keep_positions_set
                    alive_set -= evicted

                    # Update alive_positions list
                    alive_positions = [p for p in alive_positions if p in alive_set]

        # Step 2: Compute current query's attention and find argmax
        if t > 0:
            q_t = q_block[t:t+1].to(device)
            k_all = k_block[:t].to(device)

            attn_logits = torch.matmul(q_t, k_all.t()) * scale
            attn_weights = F.softmax(attn_logits, dim=-1, dtype=torch.float32).squeeze()

            argmax_key = attn_weights.argmax().item()

            if t > budget:
                total_queries_after_budget += 1
                if argmax_key not in alive_set:
                    miss_positions.append((t, argmax_key))

    stats = {
        'total_queries_after_budget': total_queries_after_budget,
        'total_miss': len(miss_positions),
        'miss_rate': len(miss_positions) / max(total_queries_after_budget, 1),
        'final_alive_count': len(alive_set),
    }

    return miss_positions, stats


def compute_attention_heatmap(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute attention heatmap (same logic as original script)."""
    head_dim = q_block.shape[-1]
    scale = head_dim ** -0.5

    num_patches = math.ceil(seq_len / patch_size)
    pooled_groups = torch.zeros((num_patches, num_patches), device=device, dtype=torch.float32)

    q_tile = 512  # Process queries in tiles for memory efficiency

    for q_start in range(0, seq_len, q_tile):
        q_end = min(q_start + q_tile, seq_len)
        q_slice = q_block[q_start:q_end].to(device)
        k_all = k_block[:seq_len].to(device)

        # Compute attention scores
        scores = torch.matmul(q_slice, k_all.t()) * scale  # [q_tile, seq_len]

        # Causal mask
        q_indices = torch.arange(q_start, q_end, device=device).unsqueeze(1)
        k_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = k_indices > q_indices
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Softmax
        weights = F.softmax(scores, dim=-1, dtype=torch.float32)  # [q_tile, seq_len]

        # Aggregate into patches using max pooling
        for local_q in range(q_end - q_start):
            global_q = q_start + local_q
            q_patch = global_q // patch_size

            for k_patch in range(num_patches):
                k_start_p = k_patch * patch_size
                k_end_p = min((k_patch + 1) * patch_size, global_q + 1)
                if k_start_p >= k_end_p:
                    continue

                max_weight = weights[local_q, k_start_p:k_end_p].max().item()
                pooled_groups[q_patch, k_patch] = max(pooled_groups[q_patch, k_patch].item(), max_weight)

    # Row-wise normalization
    row_min = pooled_groups.amin(dim=1, keepdim=True)
    row_max = pooled_groups.amax(dim=1, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_groups - row_min) / denom

    return norm.cpu()


def create_miss_overlay_map(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    miss_positions: List[Tuple[int, int]],
    seq_len: int,
    patch_size: int,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Create attention map with miss overlay.

    Returns:
        attn_map: [num_patches, num_patches] normalized attention values
        miss_mask: [num_patches, num_patches] boolean mask (True = has miss in this patch)
    """
    head_dim = q_block.shape[-1]
    scale = head_dim ** -0.5

    num_patches = math.ceil(seq_len / patch_size)
    attn_map = torch.zeros((num_patches, num_patches), device='cpu', dtype=torch.float32)
    miss_mask = torch.zeros((num_patches, num_patches), device='cpu', dtype=torch.bool)

    # Convert miss_positions to a set for O(1) lookup
    miss_set = set(miss_positions)

    q_tile = 512

    print("Computing attention map with miss detection...")
    for q_start in range(0, seq_len, q_tile):
        if q_start % 4096 == 0:
            print(f"  Processing queries {q_start}/{seq_len}")

        q_end = min(q_start + q_tile, seq_len)
        q_slice = q_block[q_start:q_end].to(device)
        k_all = k_block[:seq_len].to(device)

        # Compute attention scores
        scores = torch.matmul(q_slice, k_all.t()) * scale

        # Causal mask
        q_indices = torch.arange(q_start, q_end, device=device).unsqueeze(1)
        k_indices = torch.arange(seq_len, device=device).unsqueeze(0)
        causal_mask = k_indices > q_indices
        scores = scores.masked_fill(causal_mask, float("-inf"))

        # Softmax
        weights = F.softmax(scores, dim=-1, dtype=torch.float32)  # [q_tile, seq_len]

        # For each query, find argmax and check miss
        for local_q in range(q_end - q_start):
            global_q = q_start + local_q
            q_patch = global_q // patch_size

            if global_q == 0:
                continue

            # Get argmax key for this query
            valid_weights = weights[local_q, :global_q+1]
            argmax_key = valid_weights.argmax().item()
            argmax_weight = valid_weights[argmax_key].item()
            k_patch = argmax_key // patch_size

            # Update attention map
            attn_map[q_patch, k_patch] = max(attn_map[q_patch, k_patch].item(), argmax_weight)

            # Check if this is a miss
            if (global_q, argmax_key) in miss_set:
                miss_mask[q_patch, k_patch] = True

    # Row-wise normalization
    row_min = attn_map.amin(dim=1, keepdim=True)
    row_max = attn_map.amax(dim=1, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    attn_map_norm = (attn_map - row_min) / denom

    return attn_map_norm, miss_mask


def create_rgb_image_from_miss(
    attn_map: torch.Tensor,
    miss_mask: torch.Tensor,
) -> np.ndarray:
    """Create RGB image with miss overlay (red for miss, blue for normal)."""
    attn_np = attn_map.numpy()
    miss_np = miss_mask.numpy()

    rgb_img = np.zeros((*attn_np.shape, 3))

    for i in range(attn_np.shape[0]):
        for j in range(attn_np.shape[1]):
            attn_val = attn_np[i, j]
            if miss_np[i, j]:
                # Red for miss
                rgb_img[i, j] = [
                    face_color[0] + (color_miss[0] - face_color[0]) * attn_val,
                    face_color[1] + (color_miss[1] - face_color[1]) * attn_val,
                    face_color[2] + (color_miss[2] - face_color[2]) * attn_val,
                ]
            else:
                # Blue for normal
                rgb_img[i, j] = [
                    face_color[0] + (color_dependent[0] - face_color[0]) * attn_val,
                    face_color[1] + (color_dependent[1] - face_color[1]) * attn_val,
                    face_color[2] + (color_dependent[2] - face_color[2]) * attn_val,
                ]

    return rgb_img


def generate_figure(args: argparse.Namespace) -> None:
    """Generate the miss analysis figure."""
    mask_process_command("PD-L1_binder_rkv_miss")

    trace_dir = args.trace_dir
    if not trace_dir.is_absolute():
        trace_dir = ROOT / trace_dir

    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")
    print(f"Layer: {args.layer}, Head: {args.head}")
    print(f"RKV params: budget={args.budget}, window_size={args.window_size}, kernel_size={args.kernel_size}")

    device = torch.device(args.device)

    # Load Q/K tensors
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    q_block = q_tensor[args.layer, args.head, :token_count].to(dtype=torch.float32)
    k_block = k_tensor[args.layer, args.head, :token_count].to(dtype=torch.float32)

    # Step 1: Simulate RKV (attention-based) and detect misses
    print("\n" + "="*50)
    print("Method 1: RKV (Attention-based)")
    print("="*50)
    with torch.no_grad():
        rkv_miss_positions, rkv_stats = simulate_rkv_and_detect_miss(
            q_block, k_block, token_count,
            args.budget, args.window_size, args.kernel_size, device
        )

    print(f"\nRKV Miss Statistics:")
    print(f"  Queries after budget exceeded: {rkv_stats['total_queries_after_budget']}")
    print(f"  Total miss: {rkv_stats['total_miss']}")
    print(f"  Miss rate: {rkv_stats['miss_rate']:.2%}")

    # Step 2: Simulate Key-Norm based compression
    print("\n" + "="*50)
    print("Method 2: Key-Norm based")
    print("="*50)
    with torch.no_grad():
        keynorm_miss_positions, keynorm_stats = simulate_keynorm_and_detect_miss(
            q_block, k_block, token_count,
            args.budget, args.window_size, device
        )

    print(f"\nKey-Norm Miss Statistics:")
    print(f"  Queries after budget exceeded: {keynorm_stats['total_queries_after_budget']}")
    print(f"  Total miss: {keynorm_stats['total_miss']}")
    print(f"  Miss rate: {keynorm_stats['miss_rate']:.2%}")

    # Step 3: Compute attention maps
    print("\n" + "="*50)
    print("Computing attention maps...")
    print("="*50)

    print("\nComputing original attention heatmap...")
    with torch.no_grad():
        original_attn_map = compute_attention_heatmap(
            q_block, k_block, token_count, args.patch_size, device
        )

    print("Computing RKV miss overlay map...")
    with torch.no_grad():
        rkv_attn_map, rkv_miss_mask = create_miss_overlay_map(
            q_block, k_block, rkv_miss_positions, token_count, args.patch_size, device
        )

    print("Computing Key-Norm miss overlay map...")
    with torch.no_grad():
        keynorm_attn_map, keynorm_miss_mask = create_miss_overlay_map(
            q_block, k_block, keynorm_miss_positions, token_count, args.patch_size, device
        )

    # Step 4: Create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), dpi=args.dpi)

    # Left: Original attention map
    ax1 = axes[0]
    ax1.imshow(original_attn_map.numpy(), cmap=attn_cmap_normal, aspect="equal", origin="upper")
    ax1.set_title(f"Original Attention\n(L{args.layer}H{args.head})", fontsize=FONT_SIZE)
    ax1.set_xlabel("Key Position", fontsize=FONT_SIZE - 2)
    ax1.set_ylabel("Query Position", fontsize=FONT_SIZE - 2)
    for spine in ax1.spines.values():
        spine.set_visible(False)
    ax1.set_xticks([])
    ax1.set_yticks([])

    # Middle: RKV (attention-based) with miss overlay
    ax2 = axes[1]
    rkv_rgb_img = create_rgb_image_from_miss(rkv_attn_map, rkv_miss_mask)
    ax2.imshow(rkv_rgb_img, aspect="equal", origin="upper")
    ax2.set_title(f"RKV (Attention-based)\nMiss Rate: {rkv_stats['miss_rate']:.1%}", fontsize=FONT_SIZE)
    ax2.set_xlabel("Key Position", fontsize=FONT_SIZE - 2)
    for spine in ax2.spines.values():
        spine.set_visible(False)
    ax2.set_xticks([])
    ax2.set_yticks([])

    # Right: Key-Norm based with miss overlay
    ax3 = axes[2]
    keynorm_rgb_img = create_rgb_image_from_miss(keynorm_attn_map, keynorm_miss_mask)
    ax3.imshow(keynorm_rgb_img, aspect="equal", origin="upper")
    ax3.set_title(f"Key-Norm based\nMiss Rate: {keynorm_stats['miss_rate']:.1%}", fontsize=FONT_SIZE)
    ax3.set_xlabel("Key Position", fontsize=FONT_SIZE - 2)
    for spine in ax3.spines.values():
        spine.set_visible(False)
    ax3.set_xticks([])
    ax3.set_yticks([])

    # Add legend to rightmost plot
    legend_elements = [
        Patch(facecolor=color_dependent, label='Normal (key alive)'),
        Patch(facecolor=color_miss, label='Miss (key evicted)'),
    ]
    ax3.legend(handles=legend_elements, loc='upper right', fontsize=FONT_SIZE - 4)

    # Add common settings info
    fig.suptitle(f"Budget={args.budget}, Window={args.window_size}", fontsize=FONT_SIZE - 2, y=0.02)

    plt.tight_layout()
    plt.subplots_adjust(bottom=0.08)

    # Save figure
    if args.output_path is None:
        output_dir = ROOT / "paper_visualizations/outputs/rkv_miss_analysis"
        output_path = output_dir / f"fig_rkv_miss_L{args.layer}H{args.head}.png"
    else:
        output_path = args.output_path

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


def main() -> None:
    args = parse_args()
    generate_figure(args)


if __name__ == "__main__":
    main()
