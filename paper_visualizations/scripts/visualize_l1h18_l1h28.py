"""Visualize L1H18 and L1H28 attention maps using pooled heatmap method.

Uses the same visualization method as generate_comparison_figure.py:
- Max pooling over patches
- Softmax with causal mask
- Row-wise min-max normalization
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 14
face_color = (231 / 250, 231 / 250, 240 / 250)  # light gray-purple background
hotspot_darker_navy = (30 / 255, 50 / 255, 120 / 255)  # darker navy

attn_cmap_custom = LinearSegmentedColormap.from_list(
    "attn_darker_navy", [face_color, hotspot_darker_navy]
)

# Target heads
HEADS = [
    {"layer": 0, "head": 1, "label": "L0H1"},
    {"layer": 1, "head": 28, "label": "L1H28"},
    {"layer": 1, "head": 18, "label": "L1H18"},
]

MODEL_NAME = "Qwen3-8B"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize L1H18 and L1H28 attention maps")
    parser.add_argument(
        "trace_dir",
        type=Path,
        nargs="?",
        default=Path("outputs/deepseek_r1_qwen3_8b/qk_bf16_traces/qid0003_trace34"),
        help="Directory containing qk.pt and metadata.json",
    )
    parser.add_argument("--device", default="cuda:0", help="Computation device")
    parser.add_argument(
        "--patch-size",
        type=int,
        default=60,
        help="Pooling window size (smaller = higher resolution)",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=512,
        help="Query tile size for memory-efficient computation",
    )
    parser.add_argument("--dpi", type=int, default=200, help="DPI for saved figure")
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Output path for figure",
    )
    return parser.parse_args()


def compute_attention_heatmap(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
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


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_l1h18_l1h28")

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/l1h18_l1h28")
        # Generate filename from head IDs
        head_ids = "_".join([f"L{h['layer']}H{h['head']}" for h in HEADS])
        output_path = output_dir / f"fig_{head_ids}.png"
    else:
        output_path = args.output_path

    device = torch.device(args.device)

    # Load data
    qk_path = args.trace_dir / "qk.pt"
    meta_path = args.trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {args.trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")

    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    # Create figure: 1 row x 3 cols
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), dpi=args.dpi)
    plt.subplots_adjust(wspace=0.02)

    for col_idx, head_info in enumerate(HEADS):
        layer = head_info["layer"]
        head = head_info["head"]
        label = head_info["label"]

        print(f"Processing {label} (Layer {layer}, Head {head})...")

        q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
        k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

        with torch.no_grad():
            heatmap = compute_attention_heatmap(
                q_block, k_block, token_count, args.patch_size, args.q_tile, device
            )

        ax = axes[col_idx]
        ax.imshow(heatmap.numpy(), cmap=attn_cmap_custom, aspect="equal", origin="upper")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        # Add head ID annotation in top-right corner (paper format)
        # Use Liberation Serif (Times New Roman equivalent)
        ax.text(
            0.97, 0.97,
            f"Layer {layer}\nHead {head}",
            transform=ax.transAxes,
            fontsize=FONT_SIZE * 1.8,
            fontfamily='Liberation Serif',
            ha='right', va='top',
            multialignment='left',
            linespacing=1.2,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='none')
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=args.dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


if __name__ == "__main__":
    main()
