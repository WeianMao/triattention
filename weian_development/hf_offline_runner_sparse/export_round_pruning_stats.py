"""Export per-head frequency statistics for round-based sparse pruning."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import torch

from weian_development.process_utils import mask_process_command
from weian_development.hf_offline_runner_sparse.round_pruning_utils import (
    DTYPE_MAP,
    HeadFrequencyStats,
    build_rotary,
    compute_rotary_tables,
    load_or_create_sample,
    save_head_frequency_stats,
    invert_rope,
    to_complex_pairs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export cached per-head frequency stats for sparse KV pruning.",
    )
    parser.add_argument(
        "stats_trace",
        type=Path,
        help="Trace directory containing qk.pt and metadata.json (e.g., qid0008_trace46).",
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        required=True,
        help="Destination path for the serialized stats (torch.save).",
    )
    parser.add_argument(
        "--head-sample-file",
        type=Path,
        default=Path("weian_development/online_k_pruning_viz/hybrid_sample_heads.json"),
        help="JSON file describing sampled (layer, head) pairs (created if missing).",
    )
    parser.add_argument(
        "--sample-count",
        type=int,
        default=100,
        help="Number of heads to sample when generating a new sample file.",
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used for head sampling when file is missing.",
    )
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
        help="Model directory for recovering RoPE parameters.",
    )
    parser.add_argument(
        "--device",
        default="cuda:0",
        help="Device used for computing statistics (e.g., cuda:0 or cpu).",
    )
    parser.add_argument(
        "--dtype",
        choices=list(DTYPE_MAP.keys()),
        default="float32",
        help="Computation dtype for head statistics.",
    )
    return parser.parse_args()


def main() -> None:
    mask_process_command("PD-L1_binder")
    args = parse_args()

    stats_trace = args.stats_trace
    if not stats_trace.is_absolute():
        stats_trace = (Path.cwd() / stats_trace).resolve()
    if not stats_trace.exists():
        raise SystemExit(f"Stats trace directory not found: {stats_trace}")

    qk_path = stats_trace / "qk.pt"
    meta_path = stats_trace / "metadata.json"
    if not qk_path.exists() or not meta_path.exists():
        raise SystemExit("stats_trace must contain qk.pt and metadata.json")

    with meta_path.open("r", encoding="utf-8") as handle:
        meta = json.load(handle)
    seq_len = int(meta["sequence_length"])

    tensors = torch.load(qk_path, map_location="cpu")
    q_tensor: torch.Tensor = tensors["q"]
    layer_count = q_tensor.shape[0]
    head_per_layer = q_tensor.shape[1]
    head_dim = q_tensor.shape[-1]

    device = torch.device(args.device)
    dtype = DTYPE_MAP[args.dtype]

    sampled_heads = load_or_create_sample(
        args.head_sample_file,
        args.sample_count,
        args.sample_seed,
        layer_count,
        head_per_layer,
    )

    rotary = build_rotary(device, args.model_path, dtype)
    attention_scale = float(getattr(rotary, "attention_scaling", 1.0))
    cos_table, sin_table, _, _ = compute_rotary_tables(
        rotary, seq_len, head_dim, dtype, device
    )

    stats_map: Dict[Tuple[int, int], HeadFrequencyStats] = {}
    for layer, head in sampled_heads:
        layer_tensor = q_tensor[layer].to(device=device, dtype=dtype)
        q_head = layer_tensor[head, :seq_len, :].contiguous()
        q_unrot = invert_rope(q_head, cos_table, sin_table, attention_scale)
        q_complex = to_complex_pairs(q_unrot)
        q_mean_complex = q_complex.mean(dim=0).to(torch.complex64)
        q_abs_mean = torch.abs(q_complex).mean(dim=0).to(torch.float32)
        stats_map[(layer, head)] = HeadFrequencyStats(
            q_mean_complex=q_mean_complex,
            q_abs_mean=q_abs_mean,
        )

    metadata = {
        "sequence_length": seq_len,
        "head_dim": head_dim,
        "dtype": args.dtype,
        "model_path": str(args.model_path),
        "trace": str(stats_trace),
    }
    save_head_frequency_stats(args.output_path, sampled_heads, stats_map, metadata)
    print(f"Saved stats for {len(sampled_heads)} heads to {args.output_path}")


if __name__ == "__main__":
    main()
