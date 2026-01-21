"""生成 2048 token 区域的注意力热图（8x8 池化），仅可视化序列前 2048 token。"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


# 固定可视化区域大小和池化窗口
VIS_SIZE = 2048
POOL_SIZE = 8  # 8x8 pooling -> 输出 256x256


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 qk.pt 生成前 2048 token 的注意力热图（8x8 池化 -> 256x256）")
    parser.add_argument(
        "input_root",
        type=Path,
        help="包含 qid*_trace*/qk.pt 的目录",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="输出根目录（默认在 paper_visualizations/outputs/attention_maps_2048_raw/）",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=256,
        help="每次送入 GPU 计算的 query 数量",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="计算所用设备",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="计算时的张量精度",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=100,
        help="保存图片的 DPI",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(20.48, 20.48),
        help="matplotlib figsize (inch)",
    )
    parser.add_argument(
        "--colormap",
        default="inferno",
        help="热图配色方案",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="打印每个 head 的进度",
    )
    parser.add_argument(
        "--max-layers",
        type=int,
        default=None,
        help="可选限制处理的层数",
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=None,
        help="可选限制每层处理的头数",
    )
    parser.add_argument(
        "--head-batch",
        type=int,
        default=4,
        help="每次并行处理的注意力头数量",
    )
    return parser.parse_args()


def select_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    return mapping[name]


def compute_attention_heatmap_pooled(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    vis_size: int,
    pool_size: int,
    q_tile: int,
    device: torch.device,
) -> torch.Tensor:
    """在全部 Q/K 上计算 attention，只可视化前 vis_size×vis_size 区域并池化。

    返回 [heads, vis_size//pool_size, vis_size//pool_size]。
    """
    head_count, full_seq_len, head_dim = q_block.shape
    scale = head_dim ** -0.5

    num_groups = vis_size // pool_size  # 256 for 2048/8

    # 存储池化后的结果
    pooled_attn = torch.zeros(
        (head_count, num_groups, num_groups),
        device=device,
        dtype=torch.float32,
    )

    k_t = k_block.transpose(1, 2).contiguous()  # [H, D, full_seq_len]
    key_positions = torch.arange(full_seq_len, device=device)

    # 只处理前 vis_size 个 query
    for q_start in range(0, vis_size, q_tile):
        q_end = min(q_start + q_tile, vis_size)
        indices = torch.arange(q_start, q_end, device=device)
        q_slice = q_block[:, q_start:q_end, :]  # [H, tile, D]

        # 计算对全部 key 的 attention score
        scores = torch.matmul(q_slice, k_t) * scale  # [H, tile, full_seq_len]
        scores = scores.to(torch.float32)

        # causal mask (对全部 key)
        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        scores = scores.masked_fill(future_mask.unsqueeze(0), float("-inf"))

        # softmax (分母包含所有有效 key)
        row_max = scores.max(dim=-1, keepdim=True).values
        row_max = torch.where(
            torch.isfinite(row_max),
            row_max,
            torch.zeros_like(row_max),
        )
        stable = torch.exp(scores - row_max)
        row_sum = stable.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        weights = stable / row_sum  # [H, tile, full_seq_len]

        # 只取前 vis_size 个 key 的 attention weights 进行可视化
        weights_vis = weights[:, :, :vis_size]  # [H, tile, vis_size]

        # 对 key 维度做 max pooling: [H, tile, vis_size] -> [H, tile, num_groups]
        weights_reshaped = weights_vis.view(head_count, -1, num_groups, pool_size)
        pooled_k = weights_reshaped.max(dim=-1).values  # [H, tile, num_groups]

        # 对 query 维度做 max pooling (scatter_reduce)
        query_groups = indices // pool_size
        base_group = int(query_groups.min().item())
        local_groups = (query_groups - base_group).to(torch.int64)
        groups_in_tile = int(local_groups.max().item()) + 1

        expanded_index = local_groups.view(1, -1, 1).expand(head_count, -1, num_groups)
        tile_max = torch.zeros(
            (head_count, groups_in_tile, num_groups),
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
        pooled_attn[:, base_group:end_group] = torch.maximum(
            pooled_attn[:, base_group:end_group], tile_max
        )

    # 行归一化
    row_min = pooled_attn.amin(dim=2, keepdim=True)
    row_max = pooled_attn.amax(dim=2, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_attn - row_min) / denom
    norm = torch.clamp(norm, 0.0, 1.0)
    return norm.detach().cpu()


def save_heatmap_image(
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
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_trace_readme(
    trace_dir: Path,
    meta: Dict,
    vis_size: int,
    pool_size: int,
    dpi: int,
    figsize: Tuple[float, float],
) -> None:
    readme = trace_dir / "README.md"
    width_px = int(figsize[0] * dpi)
    height_px = int(figsize[1] * dpi)
    out_size = vis_size // pool_size
    content = f"""# 注意力热图说明（前 {vis_size} tokens，{pool_size}x{pool_size} 池化）

- 原始序列长度：{meta['sequence_length']} tokens
- 可视化区域：前 {vis_size} tokens（Q 和 K 都取前 {vis_size}）
- Softmax 计算：在全部 {meta['sequence_length']} 个 key 上计算（分母包含所有有效 key）
- 池化窗口：{pool_size}x{pool_size} -> 输出 {out_size}x{out_size}
- 图片尺寸：约 {width_px}×{height_px} 像素（figsize={figsize}, dpi={dpi}）
- 热图计算：全序列 softmax 后取前 {vis_size}×{vis_size} 区域，{pool_size}x{pool_size} max pooling，行归一化
- 文件命名：`layer_{{L:02d}}_head_{{H:02d}}.png`
- 生成脚本：`visualize_attention_maps_2048_raw.py`
"""
    readme.write_text(content, encoding="utf-8")


def process_trace(trace_dir: Path, args: argparse.Namespace) -> None:
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        return

    # 确定输出目录
    if args.output_root:
        trace_root = args.output_root / trace_dir.name
    else:
        default_output = ROOT / "paper_visualizations" / "outputs" / "attention_maps_2048_raw"
        trace_root = default_output / trace_dir.name

    output_dir = trace_root / "attention_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])

    # 确保至少有 VIS_SIZE 个 token
    vis_size = min(VIS_SIZE, token_count)
    if vis_size < VIS_SIZE:
        print(f"Warning: sequence length {token_count} < {VIS_SIZE}, using {vis_size}")

    device = torch.device(args.device)
    compute_dtype = select_dtype(args.dtype)

    mask_process_command("PD-L1_binder_vis")
    torch.backends.cuda.matmul.allow_tf32 = True

    load_start = time.perf_counter()
    data = torch.load(qk_path, map_location="cpu")
    load_time = time.perf_counter() - load_start
    if args.verbose:
        print(f"[timing] load qk.pt: {load_time:.2f}s")
    q_tensor = data["q"]  # [layers, heads, seq, dim]
    k_tensor = data["k"]
    num_layers, num_heads, _, _ = q_tensor.shape

    layer_limit = args.max_layers if args.max_layers is not None else num_layers
    head_limit = args.max_heads if args.max_heads is not None else num_heads
    layer_limit = min(layer_limit, num_layers)
    head_limit = min(head_limit, num_heads)

    head_batch = max(1, args.head_batch)

    compute_time = 0.0
    save_time = 0.0

    for layer in range(layer_limit):
        head_idx = 0
        while head_idx < head_limit:
            batch_size = min(head_batch, head_limit - head_idx)
            q_block = q_tensor[layer, head_idx : head_idx + batch_size].to(device=device, dtype=compute_dtype)
            k_block = k_tensor[layer, head_idx : head_idx + batch_size].to(device=device, dtype=compute_dtype)

            compute_start = time.perf_counter()
            with torch.no_grad():
                heatmaps = compute_attention_heatmap_pooled(
                    q_block,
                    k_block,
                    vis_size,
                    POOL_SIZE,
                    args.q_tile,
                    device,
                )
            compute_time += time.perf_counter() - compute_start

            for local_h in range(batch_size):
                head_id = head_idx + local_h
                heatmap = heatmaps[local_h]
                out_size = vis_size // POOL_SIZE
                title = f"Layer {layer} Head {head_id} | {vis_size} tokens, {POOL_SIZE}x{POOL_SIZE} pool -> {out_size}x{out_size}"
                out_path = output_dir / f"layer_{layer:02d}_head_{head_id:02d}.png"
                save_start = time.perf_counter()
                save_heatmap_image(
                    heatmap,
                    out_path,
                    cmap=args.colormap,
                    figsize=args.figsize,
                    dpi=args.dpi,
                    title=title,
                )
                save_time += time.perf_counter() - save_start
                if args.verbose:
                    rel = out_path.relative_to(output_dir.parent)
                    print(f"Saved {rel}")

            del q_block, k_block, heatmaps
            head_idx += batch_size

    generate_trace_readme(
        output_dir,
        meta,
        vis_size,
        POOL_SIZE,
        args.dpi,
        tuple(args.figsize),
    )

    del q_tensor, k_tensor, data
    if device.type == "cuda":
        torch.cuda.empty_cache()

    if args.verbose:
        total_time = time.perf_counter() - wall_start
        print(
            f"[timing] trace done | total={total_time:.2f}s compute={compute_time:.2f}s save={save_time:.2f}s"
        )


def main() -> None:
    args = parse_args()
    mask_process_command("PD-L1_binder_ctl")

    if not args.input_root.exists():
        raise SystemExit(f"输入目录不存在：{args.input_root}")

    trace_dirs = sorted([p for p in args.input_root.iterdir() if p.is_dir() and p.name.startswith("qid")])
    if not trace_dirs:
        raise SystemExit(f"在 {args.input_root} 下未找到 qid*_trace* 目录")

    for trace_dir in trace_dirs:
        if args.verbose:
            print(f"Processing {trace_dir}")
        process_trace(trace_dir, args)


if __name__ == "__main__":
    main()
