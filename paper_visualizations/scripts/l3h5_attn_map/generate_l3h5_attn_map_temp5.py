"""Generate attention map visualization for Layer 3, Head 5 with temperature scaling."""
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

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# ============ Styling Constants ============
FONT_SIZE = 16
color_dependent = (85 / 250, 104 / 250, 154 / 250)    # blue for dependent
face_color = (231 / 250, 231 / 250, 240 / 250)        # light gray-purple background

# Custom colormap: face_color (low attention) -> color_dependent (high attention)
attn_cmap_custom = LinearSegmentedColormap.from_list(
    "attn_custom", [face_color, color_dependent]
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate L3H5 attention map with temperature")
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
        default=32,
        help="Pooling window size for attention maps (larger = coarser)",
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
    parser.add_argument("--layer", type=int, default=3, help="Layer index")
    parser.add_argument("--head", type=int, default=5, help="Head index")
    parser.add_argument("--temperature", type=float, default=5.0, help="Temperature scaling for softmax")
    return parser.parse_args()


def compute_attention_heatmap(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
    temperature: float = 1.0,
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

        scores = torch.matmul(q_slice, k_t) * scale / temperature
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


def generate_figure(
    trace_dir: Path,
    device: torch.device,
    patch_size: int,
    q_tile: int,
    dpi: int,
    output_path: Path,
    layer: int,
    head: int,
    temperature: float,
) -> None:
    """Generate the attention map figure."""
    mask_process_command("PD-L1_binder_l3h5_temp")

    # Load data
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise FileNotFoundError(f"Missing qk.pt or metadata.json in {trace_dir}")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    print(f"Token count: {token_count}")
    print(f"Temperature: {temperature}")

    # Load Q/K tensors
    print("Loading Q/K tensors...")
    data = torch.load(qk_path, map_location="cpu")
    q_tensor = data["q"]
    k_tensor = data["k"]

    print(f"Processing Layer {layer}, Head {head}...")
    q_block = q_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)
    k_block = k_tensor[layer, head, :token_count].to(device=device, dtype=torch.float32)

    # Compute attention map
    with torch.no_grad():
        heatmap = compute_attention_heatmap(q_block, k_block, token_count, patch_size, q_tile, device, temperature)

    # Create figure
    fig, ax = plt.subplots(figsize=(6, 6), dpi=dpi)

    ax.imshow(heatmap.numpy(), cmap=attn_cmap_custom, aspect="equal", origin="upper")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    ax.set_title(f"L{layer}H{head} Attention Map (T={temperature})", fontsize=FONT_SIZE, fontweight='bold')

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    print(f"\nFigure saved to {output_path}")


def main() -> None:
    args = parse_args()

    if args.output_path is None:
        output_dir = Path("paper_visualizations/outputs/l3h5_attn_map")
        output_path = output_dir / f"fig_l{args.layer}h{args.head}_attn_map_temp{int(args.temperature)}.png"
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
        temperature=args.temperature,
    )


if __name__ == "__main__":
    main()
