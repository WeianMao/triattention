"""生成离线 Q/K 资产的注意力热图，可对长序列做分块与池化。"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, Tuple

import torch
import matplotlib
matplotlib.use("Agg")  # headless 环境下保存图片
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in __import__("sys").path:
    __import__("sys").path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="基于 qk.pt 生成注意力热图")
    parser.add_argument(
        "input_root",
        type=Path,
        help="包含 qid*_trace*/qk.pt 的目录，例如 outputs/deepseek_.../qk_bf16_traces",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=None,
        help="输出根目录（默认写到各 trace 子目录下的 attention_maps/）",
    )
    parser.add_argument(
        "--target-size",
        type=int,
        default=4096,
        help="目标像素数（行/列），在未指定 `--patch-size` 时用于推断池化窗口",
    )
    parser.add_argument(
        "--patch-size",
        type=int,
        default=32,
        help="Query/Key 方向统一采用的最大池化窗口（默认 32）",
    )
    parser.add_argument(
        "--q-tile",
        type=int,
        default=512,
        help="每次送入 GPU 计算的 query 数量，避免一次性显存爆炸",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="计算所用设备，例如 cuda:0 或 cpu",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "bfloat16", "float16"],
        default="float32",
        help="计算时的张量精度（建议 float32 以保留数值稳定性）",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="保存图片的 DPI，配合 figsize 生成约 4K 像素的图片",
    )
    parser.add_argument(
        "--figsize",
        type=float,
        nargs=2,
        default=(20.48, 20.48),
        help="matplotlib figsize (inch)，默认对应 4096x4096 @ 200dpi",
    )
    parser.add_argument(
        "--colormap",
        default="inferno",
        help="热图配色方案（传给 matplotlib.imshow）",
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
        help="可选限制处理的层数（从 0 开始按顺序截断）",
    )
    parser.add_argument(
        "--max-heads",
        type=int,
        default=None,
        help="可选限制每层处理的头数（从 0 开始按顺序截断）",
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


def resolve_patch_size(seq_len: int, target_size: int, patch_arg: int | None) -> int:
    if patch_arg and patch_arg > 0:
        return patch_arg
    if seq_len <= target_size:
        return 1
    return math.ceil(seq_len / target_size)


def compute_attention_heatmap_block(
    q_block: torch.Tensor,
    k_block: torch.Tensor,
    seq_len: int,
    patch_size: int,
    q_tile: int,
    device: torch.device,
) -> torch.Tensor:
    """基于多头 Q/K 块生成池化后的注意力热图，返回 [heads, num_q_groups, num_k_groups]。"""

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

    k_t = k_block.transpose(1, 2).contiguous()  # [H, D, seq_k]

    for q_start in range(0, seq_q_real, q_tile):
        q_end = min(q_start + q_tile, seq_q_real)
        indices = torch.arange(q_start, q_end, device=device)
        q_slice = q_block[:, q_start:q_end, :]  # [H, tile, D]

        scores = torch.matmul(q_slice, k_t) * scale  # [H, tile, seq_k]
        scores = scores.to(torch.float32)

        future_mask = key_positions.unsqueeze(0) > indices.unsqueeze(1)
        scores = scores.masked_fill(future_mask.unsqueeze(0), float("-inf"))
        scores = scores.masked_fill(~key_valid.view(1, 1, -1), float("-inf"))

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

    row_min = pooled_groups.amin(dim=2, keepdim=True)
    row_max = pooled_groups.amax(dim=2, keepdim=True)
    denom = (row_max - row_min).clamp_min(1e-12)
    norm = (pooled_groups - row_min) / denom
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
    ax.set_xlabel("Key group index")
    ax.set_ylabel("Query group index")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def generate_trace_readme(
    trace_dir: Path,
    meta: Dict,
    pool_q: int,
    pool_k: int,
    num_q_groups: int,
    num_k_groups: int,
    target_size: int,
    dpi: int,
    figsize: Tuple[float, float],
) -> None:
    readme = trace_dir / "README.md"
    width_px = int(figsize[0] * dpi)
    height_px = int(figsize[1] * dpi)
    content = f"""# 注意力热图说明

- 原始序列长度：{meta['token_count']} tokens
- Query 池化窗口：{pool_q} → {num_q_groups} 个 query 组
- Key 池化窗口：{pool_k} → {num_k_groups} 个 key 组
- 图片尺寸：约 {width_px}×{height_px} 像素（figsize={figsize}, dpi={dpi}）
- 热图计算：对 Q·K^T 分块后进行 softmax，随后分别对 key / query 维度执行最大池化，并在对数域做归一化拉伸
- 文件命名：`layer_{{L:02d}}_head_{{H:02d}}.png`
- 生成脚本：`visualize_attention_maps.py`（target_size={target_size}）
"""
    readme.write_text(content, encoding="utf-8")


def process_trace(trace_dir: Path, args: argparse.Namespace) -> None:
    qk_path = trace_dir / "qk.pt"
    meta_path = trace_dir / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        return

    if args.output_root:
        trace_root = args.output_root / trace_dir.name
    else:
        trace_root = trace_dir

    output_dir = trace_root / "attention_maps"
    output_dir.mkdir(parents=True, exist_ok=True)

    wall_start = time.perf_counter()
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    token_count = int(meta["sequence_length"])
    patch_size = resolve_patch_size(token_count, args.target_size, args.patch_size)
    pool_q = pool_k = patch_size
    num_q_groups = math.ceil(token_count / pool_q)
    num_k_groups = math.ceil(token_count / pool_k)

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
                heatmaps = compute_attention_heatmap_block(
                    q_block,
                    k_block,
                    token_count,
                    patch_size,
                    args.q_tile,
                    device,
                )
            compute_time += time.perf_counter() - compute_start

            for local_h in range(batch_size):
                head_id = head_idx + local_h
                heatmap = heatmaps[local_h]
                title = f"Layer {layer} Head {head_id} | patch={patch_size}"
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
            # 避免频繁 empty_cache 导致同步

            head_idx += batch_size

    generate_trace_readme(
        output_dir,
        meta,
        pool_q,
        pool_k,
        num_q_groups,
        num_k_groups,
        args.target_size,
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
