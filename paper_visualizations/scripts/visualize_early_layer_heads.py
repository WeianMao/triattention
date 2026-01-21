"""Visualize attention maps for specific heads from early layers.

Visualizes layer_01_head_18, layer_01_head_20, and 10 additional heads from
the first 3 layers.

Visualization modes:
- PRIMARY_HEADS (L1H18, L1H20): Full attention map with pooling (like original script)
- ADDITIONAL_HEADS (10 heads): Center 500x500 crop, NO pooling
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Dict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 12
color_dependent = (85 / 250, 104 / 250, 154 / 250)    # blue for dependent
color_independent = (187 / 250, 130 / 250, 90 / 250)  # warm brown for independent
face_color = (231 / 250, 231 / 250, 240 / 250)        # light gray-purple background

# Custom colormap
attn_cmap_custom = LinearSegmentedColormap.from_list(
    "attn_custom", [face_color, color_dependent]
)


# ============ Head Configurations ============
# Single head to visualize with different temperatures
TARGET_HEAD = {"layer": 1, "head": 18, "label": "L1H18"}

# Temperature values to compare ("orig" means original pooling method)
TEMPERATURES = ["orig"]  # Only show original pooling method


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize early layer attention heads")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument(
        "--crop-size",
        type=int,
        default=500,
        help="Size of center crop for attention maps (default: 500)",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Pooling window size for PRIMARY_HEADS attention maps",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=512,
        help="Query tile size for memory-efficient attention computation",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for figure",
    )
    return parser.parse_args()


def compute_attention_heatmap_with_pooling(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute pooled attention heatmap for full sequence (from original script)."""
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


def compute_attention_heatmap_no_pooling(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    crop_size: int,
    q_tile: int,
    device: torch.device,
    temperature: float = 1.0,
    use_minmax_norm: bool = False,
) -> torch.Tensor:
    """Compute raw attention logits heatmap without pooling, cropped to center region.

    Args:
        q_block: Query tensor [seq_len, head_dim]
        k_block: Key tensor [seq_len, head_dim]
        seq_len: Sequence length
        crop_size: Size of center crop (e.g., 500 for 500x500)
        q_tile: Tile size for memory-efficient computation
        device: Computation device
        temperature: Softmax temperature (default 1.0)
        use_minmax_norm: If True, use (x-min)/(max-min) normalization instead of x/max

    Returns:
        Normalized attention heatmap of shape [crop_size, crop_size]
    """
    seq_q, head_dim = q_block.shape
    scale = head_dim ** -0.5

    # Determine crop region (centered)
    center = seq_len // 2
    half_crop = crop_size // 2
    q_start = max(0, center - half_crop)
    q_end = min(seq_len, q_start + crop_size)
    # Adjust if we hit the boundary
    if q_end - q_start < crop_size:
        q_start = max(0, q_end - crop_size)

    k_start = q_start
    k_end = q_end

    actual_crop_q = q_end - q_start
    actual_crop_k = k_end - k_start

    print(f"  Crop region: Q[{q_start}:{q_end}], K[{k_start}:{k_end}] ({actual_crop_q}x{actual_crop_k})")

    # Initialize output heatmap
    heatmap = torch.zeros((actual_crop_q, actual_crop_k), device=device, dtype=torch.float32)

    # Compute attention in tiles for memory efficiency
    k_t = k_block[k_start:k_end].t().contiguous()  # [head_dim, crop_k]
    key_positions = torch.arange(k_start, k_end, device=device)

    for tile_start in range(0, actual_crop_q, q_tile):
        tile_end = min(tile_start + q_tile, actual_crop_q)
        global_q_start = q_start + tile_start
        global_q_end = q_start + tile_end

        q_slice = q_block[global_q_start:global_q_end]  # [tile_size, head_dim]
        query_positions = torch.arange(global_q_start, global_q_end, device=device)

        # Compute attention scores (logits before softmax)
        scores = torch.matmul(q_slice, k_t) * scale  # [tile_size, crop_k]
        scores = scores.to(torch.float32)

        # Apply causal mask
        future_mask = key_positions.unsqueeze(0) > query_positions.unsqueeze(1)
        scores = scores.masked_fill(future_mask, float("-inf"))

        # Apply softmax with temperature along key dimension
        attn_weights = torch.softmax(scores / temperature, dim=-1)

        # Row-wise normalization
        if use_minmax_norm:
            # Original method: (x - min) / (max - min)
            row_min = attn_weights.amin(dim=-1, keepdim=True)
            row_max = attn_weights.amax(dim=-1, keepdim=True)
            denom = (row_max - row_min).clamp_min(1e-12)
            attn_weights = (attn_weights - row_min) / denom
        else:
            # Scale each row by max value
            row_max = attn_weights.amax(dim=-1, keepdim=True).clamp_min(1e-12)
            attn_weights = attn_weights / row_max

        # Store in heatmap
        heatmap[tile_start:tile_end] = attn_weights

    return heatmap.detach().cpu()


def generate_figure(
    trace_dir: Path,
    device: torch.device,
    crop_size: int,
    patch_size: int,
    q_tile: int,
    dpi: int,
    output_path: Path,
) -> None:
    """Generate the visualization figure."""
    mask_process_command("PD-L1_binder_early_heads")

    # Load data
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")

    if token_count < crop_size:
        print(f"Warning: sequence length ({token_count}) < crop_size ({crop_size}), adjusting crop_size")
        crop_size = token_count

    # Load Q/K tensors
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    layer = TARGET_HEAD["layer"]
    head = TARGET_HEAD["head"]
    label = TARGET_HEAD["label"]

    q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
    k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

    num_temps = len(TEMPERATURES)

    # Layout: 1 row x N cols for different temperatures
    fig, axes = plt.subplots(1, num_temps, figsize=(3.5 * num_temps, 3.5), dpi=dpi)
    fig.suptitle(f"{label} Attention Map at Different Temperatures",
                 fontsize=FONT_SIZE + 2, fontweight='bold')

    plt.rcParams.update({
        'font.size': FONT_SIZE,
        'axes.labelsize': FONT_SIZE,
        'axes.titlesize': FONT_SIZE,
    })

    for idx, temperature in enumerate(TEMPERATURES):
        ax = axes[idx] if num_temps > 1 else axes

        if temperature == "orig":
            # Original method from generate_comparison_figure.py: pooling + min-max normalization
            print(f"Processing {label} with ORIGINAL POOLING method...")
            with torch.no_grad():
                heatmap = compute_attention_heatmap_with_pooling(
                    q_block, k_block, token_count, patch_size, q_tile, device
                )
            title = "Original"
        else:
            print(f"Processing {label} with temperature={temperature}...")
            with torch.no_grad():
                heatmap = compute_attention_heatmap_no_pooling(
                    q_block, k_block, token_count, crop_size, q_tile, device, temperature
                )
            title = f"T={temperature}"
        print(f"  Heatmap shape: {heatmap.shape}")

        # Plot
        ax.imshow(heatmap.numpy(), cmap=attn_cmap_custom, aspect="equal", origin="upper")
        ax.set_title(title, fontsize=FONT_SIZE, fontweight='bold')
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

    plt.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


def main() -> None:
    args = parse_args()

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/early_layer_heads")
        output_path = output_dir / "fig_early_layer_heads.png"
    else:
        output_path = args.output_path

    device = torch.device(args.device)

    generate_figure(
        trace_dir=args.trace_dir,
        device=device,
        crop_size=args.crop_size,
        patch_size=args.patch_size,
        q_tile=args.q_tile,
        dpi=args.dpi,
        output_path=output_path,
    )


if __name__ == "__main__":
    main()
