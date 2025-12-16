#!/usr/bin/env python3
"""Oracle Upper Bound experiment for Module 1 Key Pruning.

Phase A: Use ground-truth labels as oracle predictions to establish
theoretical maximum compression rate while maintaining 100% Argmax Hit Rate.

Following specification in docs/01_module1_key_pruning.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

# Add project root for imports
ROOT = Path(__file__).resolve().parents[4]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from weian_development.process_utils import mask_process_command

# Local imports
from data_utils import (
    load_trace_data,
    load_trace_metadata,
    load_head_data,
    load_head_indices,
    compute_attention_matrix,
    get_query_argmax_info,
)
from labels import extract_pruning_labels_vectorized
from metrics import compute_oracle_metrics, compute_round_statistics


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Oracle Upper Bound experiment for Module 1 Key Pruning."
    )
    parser.add_argument(
        "--trace",
        type=Path,
        required=True,
        help="Path to trace directory containing qk.pt.",
    )
    parser.add_argument(
        "--head-indices",
        type=Path,
        default=None,
        help="Path to head indices JSON file (defaults to hybrid_sample_heads_lowret_top10.json).",
    )
    parser.add_argument(
        "--round-window",
        type=int,
        default=128,
        help="Round window size (default: 128 tokens per round).",
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
        help="Device for computation.",
    )
    parser.add_argument(
        "--dtype",
        choices=["float32", "float16", "bfloat16"],
        default="float32",
        help="Computation dtype.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "output",
        help="Output directory for results.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print detailed progress.",
    )
    return parser.parse_args()


def run_oracle_experiment_for_head(
    q_head: torch.Tensor,
    k_head: torch.Tensor,
    seq_len: int,
    round_window: int,
    verbose: bool = False,
) -> Dict:
    """
    Run oracle experiment for a single (layer, head) pair.

    Args:
        q_head: (seq_len, head_dim) Query tensor.
        k_head: (seq_len, head_dim) Key tensor.
        seq_len: Sequence length.
        round_window: Round window size.
        verbose: Print progress.

    Returns:
        Dict with per-round and aggregated metrics.
    """
    # Compute full attention matrix
    attention = compute_attention_matrix(q_head, k_head, apply_causal_mask=True)

    round_results = []
    all_retention_rates = []
    all_argmax_hit_rates = []
    all_keys_per_query = []

    # Iterate through rounds (starting from round_window to have history)
    for round_start in range(round_window, seq_len, round_window):
        # Extract labels for historical keys (positions 0 to round_start-1)
        labels = extract_pruning_labels_vectorized(
            attention,
            round_start=round_start,
            seq_len=seq_len,
        )

        if labels.numel() == 0:
            continue

        # Get query argmax info for this round
        query_argmax_indices, argmax_in_history = get_query_argmax_info(
            attention, round_start
        )

        # Compute oracle metrics (using labels as perfect predictions)
        metrics = compute_oracle_metrics(
            labels=labels,
            query_argmax_indices=query_argmax_indices,
            argmax_in_history=argmax_in_history,
        )

        # Compute round statistics
        round_stats = compute_round_statistics(
            labels=labels,
            round_start=round_start,
            seq_len=seq_len,
            round_window=round_window,
        )

        round_result = {
            "round_start": round_start,
            **metrics,
            **round_stats,
        }
        round_results.append(round_result)

        all_retention_rates.append(metrics["retention_rate"])
        all_argmax_hit_rates.append(metrics["argmax_hit_rate"])
        all_keys_per_query.append(metrics["keys_per_query"])

        if verbose:
            print(
                f"  Round {round_start}: retention={metrics['retention_rate']:.4f}, "
                f"argmax_hit={metrics['argmax_hit_rate']:.4f}, "
                f"keys_per_query={metrics['keys_per_query']:.1f}"
            )

    # Aggregate metrics across rounds
    if all_retention_rates:
        aggregated = {
            "mean_retention_rate": sum(all_retention_rates) / len(all_retention_rates),
            "min_retention_rate": min(all_retention_rates),
            "max_retention_rate": max(all_retention_rates),
            "mean_argmax_hit_rate": sum(all_argmax_hit_rates) / len(all_argmax_hit_rates),
            "min_argmax_hit_rate": min(all_argmax_hit_rates),
            "mean_keys_per_query": sum(all_keys_per_query) / len(all_keys_per_query),
            "num_rounds": len(all_retention_rates),
        }
    else:
        aggregated = {
            "mean_retention_rate": 0.0,
            "min_retention_rate": 0.0,
            "max_retention_rate": 0.0,
            "mean_argmax_hit_rate": 0.0,
            "min_argmax_hit_rate": 0.0,
            "mean_keys_per_query": 0.0,
            "num_rounds": 0,
        }

    return {
        "rounds": round_results,
        "aggregated": aggregated,
    }


def run_oracle_experiment(
    trace_dir: Path,
    head_indices: List[Tuple[int, int]],
    round_window: int,
    device: torch.device,
    dtype: str,
    verbose: bool = False,
) -> Dict:
    """
    Run oracle experiment on all specified heads.

    Uses memory-efficient loading: loads one head at a time instead of
    entire Q/K tensors.

    Args:
        trace_dir: Path to trace directory.
        head_indices: List of (layer, head) tuples.
        round_window: Round window size.
        device: Computation device.
        dtype: Data type.
        verbose: Print progress.

    Returns:
        Dict with per-head and overall results.
    """
    if verbose:
        print(f"Loading trace metadata from {trace_dir}...")

    # First load only metadata to get seq_len
    metadata_info = load_trace_metadata(trace_dir)
    seq_len = metadata_info["seq_len"]
    q_shape = metadata_info["q_shape"]

    if verbose:
        print(f"Trace shape: {q_shape}, seq_len: {seq_len}")
        print(f"Processing {len(head_indices)} heads with round_window={round_window}")

    head_results = {}
    all_mean_retention_rates = []
    all_mean_argmax_hit_rates = []

    for i, (layer, head) in enumerate(head_indices):
        head_key = f"L{layer}H{head}"
        if verbose:
            print(f"\n[{i+1}/{len(head_indices)}] Processing {head_key}...")

        # Load only this head's data (memory efficient)
        q_head, k_head, head_seq_len = load_head_data(
            trace_dir, layer, head, device=device, dtype=dtype
        )

        if verbose:
            print(f"  Loaded Q: {q_head.shape}, K: {k_head.shape}")

        result = run_oracle_experiment_for_head(
            q_head=q_head,
            k_head=k_head,
            seq_len=head_seq_len,
            round_window=round_window,
            verbose=verbose,
        )

        # Free memory after processing each head
        del q_head, k_head
        if device.type == "cuda":
            torch.cuda.empty_cache()

        head_results[head_key] = result
        all_mean_retention_rates.append(result["aggregated"]["mean_retention_rate"])
        all_mean_argmax_hit_rates.append(result["aggregated"]["mean_argmax_hit_rate"])

    # Overall aggregation
    if all_mean_retention_rates:
        overall = {
            "overall_mean_retention_rate": sum(all_mean_retention_rates) / len(all_mean_retention_rates),
            "overall_min_retention_rate": min(all_mean_retention_rates),
            "overall_max_retention_rate": max(all_mean_retention_rates),
            "overall_mean_argmax_hit_rate": sum(all_mean_argmax_hit_rates) / len(all_mean_argmax_hit_rates),
            "overall_min_argmax_hit_rate": min(all_mean_argmax_hit_rates),
            "num_heads": len(head_indices),
            "seq_len": seq_len,
            "round_window": round_window,
        }
    else:
        overall = {}

    return {
        "head_results": head_results,
        "overall": overall,
        "config": {
            "trace_dir": str(trace_dir),
            "head_indices": head_indices,
            "round_window": round_window,
            "dtype": dtype,
        },
    }


def plot_oracle_results(
    results: Dict,
    output_dir: Path,
    verbose: bool = False,
) -> None:
    """Generate visualization figures for oracle experiment results."""
    figures_dir = output_dir / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    head_results = results["head_results"]
    if not head_results:
        if verbose:
            print("No head results to plot.")
        return

    # Extract data for plotting
    head_names = list(head_results.keys())
    retention_rates = [head_results[h]["aggregated"]["mean_retention_rate"] for h in head_names]
    argmax_hit_rates = [head_results[h]["aggregated"]["mean_argmax_hit_rate"] for h in head_names]

    # Figure 1: Retention Rate per Head
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(head_names, retention_rates, color="steelblue", edgecolor="black")
    ax.set_xlabel("Layer-Head", fontsize=12)
    ax.set_ylabel("Mean Retention Rate", fontsize=12)
    ax.set_title("Oracle Experiment: Mean Retention Rate per Head", fontsize=14)
    ax.set_ylim(0, 1)
    ax.axhline(y=sum(retention_rates) / len(retention_rates), color="red",
               linestyle="--", label=f"Overall Mean: {sum(retention_rates) / len(retention_rates):.4f}")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(figures_dir / "retention_rate_per_head.png", dpi=150)
    plt.close(fig)

    # Figure 2: Argmax Hit Rate per Head (should be 100% for oracle)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.bar(head_names, argmax_hit_rates, color="forestgreen", edgecolor="black")
    ax.set_xlabel("Layer-Head", fontsize=12)
    ax.set_ylabel("Mean Argmax Hit Rate", fontsize=12)
    ax.set_title("Oracle Experiment: Mean Argmax Hit Rate per Head", fontsize=14)
    ax.set_ylim(0, 1.05)
    ax.axhline(y=1.0, color="red", linestyle="--", label="Target: 100%")
    ax.legend()
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(figures_dir / "argmax_hit_rate_per_head.png", dpi=150)
    plt.close(fig)

    # Figure 3: Combined summary
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Retention Rate
    axes[0].bar(head_names, retention_rates, color="steelblue", edgecolor="black")
    axes[0].set_xlabel("Layer-Head", fontsize=11)
    axes[0].set_ylabel("Mean Retention Rate", fontsize=11)
    axes[0].set_title("Retention Rate (Lower = More Compression)", fontsize=12)
    axes[0].set_ylim(0, 1)
    for tick in axes[0].get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    # Computation Reduction (1 - retention rate)
    compression_rates = [1 - r for r in retention_rates]
    axes[1].bar(head_names, compression_rates, color="coral", edgecolor="black")
    axes[1].set_xlabel("Layer-Head", fontsize=11)
    axes[1].set_ylabel("Computation Reduction", fontsize=11)
    axes[1].set_title("Computation Reduction (Higher = Better)", fontsize=12)
    axes[1].set_ylim(0, 1)
    for tick in axes[1].get_xticklabels():
        tick.set_rotation(45)
        tick.set_ha("right")

    plt.tight_layout()
    fig.savefig(figures_dir / "oracle_summary.png", dpi=150)
    plt.close(fig)

    if verbose:
        print(f"Figures saved to {figures_dir}/")


def main():
    mask_process_command()
    args = parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    results_dir = output_dir / "results"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Load head indices
    head_indices = load_head_indices(args.head_indices)
    if args.verbose:
        print(f"Loaded {len(head_indices)} head indices: {head_indices}")

    # Run experiment
    results = run_oracle_experiment(
        trace_dir=args.trace,
        head_indices=head_indices,
        round_window=args.round_window,
        device=device,
        dtype=args.dtype,
        verbose=args.verbose,
    )

    # Save results
    results_path = results_dir / "oracle_results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*60}")
    print("Oracle Upper Bound Experiment Results")
    print(f"{'='*60}")

    overall = results.get("overall", {})
    if overall:
        print(f"Sequence Length: {overall.get('seq_len', 'N/A')}")
        print(f"Round Window: {overall.get('round_window', 'N/A')}")
        print(f"Number of Heads: {overall.get('num_heads', 'N/A')}")
        print()
        print(f"Overall Mean Retention Rate: {overall.get('overall_mean_retention_rate', 0):.4f}")
        print(f"Overall Min Retention Rate:  {overall.get('overall_min_retention_rate', 0):.4f}")
        print(f"Overall Max Retention Rate:  {overall.get('overall_max_retention_rate', 0):.4f}")
        print()
        print(f"Overall Mean Argmax Hit Rate: {overall.get('overall_mean_argmax_hit_rate', 0):.4f}")
        print(f"Overall Min Argmax Hit Rate:  {overall.get('overall_min_argmax_hit_rate', 0):.4f}")
        print()
        max_compression = 1 - overall.get("overall_min_retention_rate", 0)
        print(f"Maximum Theoretical Compression: {max_compression:.4f}")

    print(f"\nResults saved to: {results_path}")

    # Generate figures
    plot_oracle_results(results, output_dir, verbose=args.verbose)
    print(f"Figures saved to: {output_dir / 'figures'}/")


if __name__ == "__main__":
    main()
