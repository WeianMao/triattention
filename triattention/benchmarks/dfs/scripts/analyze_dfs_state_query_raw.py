#!/usr/bin/env python3
"""Analyze DFS state query raw records by num_step bins."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        help="Path to raw JSONL produced by eval_dfs_state_query_raw.py.",
    )
    parser.add_argument(
        "--bins",
        type=str,
        default="0-5,6-10,11-20,21-30,31-9999",
        help="Comma-separated num_step bins (inclusive), e.g. '0-5,6-10,11-20'.",
    )
    return parser.parse_args()


def parse_bins(spec: str) -> List[Tuple[int, int]]:
    bins = []
    for part in spec.split(","):
        part = part.strip()
        if not part:
            continue
        if "-" not in part:
            raise ValueError(f"Invalid bin spec: {part}")
        start_str, end_str = part.split("-", 1)
        start = int(start_str)
        end = int(end_str)
        if end < start:
            raise ValueError(f"Invalid bin range: {part}")
        bins.append((start, end))
    return bins


def iter_records(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def init_stats() -> Dict[str, float]:
    return {
        "count": 0,
        "is_correct": 0,
        "current_node_correct": 0,
        "stack_exact_match": 0,
        "visited_exact_match": 0,
    }


def main() -> None:
    args = parse_args()
    bins = parse_bins(args.bins)
    bin_stats = {bin_range: init_stats() for bin_range in bins}

    overall = init_stats()

    for record in iter_records(args.input):
        num_step = record.get("num_step")
        if num_step is None:
            continue
        metrics = record.get("metrics") or {}

        overall["count"] += 1
        overall["is_correct"] += 1 if record.get("is_correct") else 0
        overall["current_node_correct"] += 1 if metrics.get("current_node_correct") else 0
        overall["stack_exact_match"] += 1 if metrics.get("stack_exact_match") else 0
        overall["visited_exact_match"] += 1 if metrics.get("visited_exact_match") else 0

        for bin_range, stats in bin_stats.items():
            if bin_range[0] <= num_step <= bin_range[1]:
                stats["count"] += 1
                stats["is_correct"] += 1 if record.get("is_correct") else 0
                stats["current_node_correct"] += 1 if metrics.get("current_node_correct") else 0
                stats["stack_exact_match"] += 1 if metrics.get("stack_exact_match") else 0
                stats["visited_exact_match"] += 1 if metrics.get("visited_exact_match") else 0
                break

    def rate(count: float, denom: float) -> float:
        return count / denom if denom else 0.0

    print("Overall:")
    if overall["count"] == 0:
        print("  No records with num_step found.")
    else:
        print(f"  count: {int(overall['count'])}")
        print(f"  is_correct: {rate(overall['is_correct'], overall['count']):.4f}")
        print(f"  current_node_correct: {rate(overall['current_node_correct'], overall['count']):.4f}")
        print(f"  stack_exact_match: {rate(overall['stack_exact_match'], overall['count']):.4f}")
        print(f"  visited_exact_match: {rate(overall['visited_exact_match'], overall['count']):.4f}")

    print("\nBy num_step bins:")
    for bin_range in bins:
        stats = bin_stats[bin_range]
        label = f"{bin_range[0]}-{bin_range[1]}"
        if stats["count"] == 0:
            print(f"  {label}: count=0")
            continue
        print(
            f"  {label}: count={int(stats['count'])} "
            f"is_correct={rate(stats['is_correct'], stats['count']):.4f} "
            f"current_node={rate(stats['current_node_correct'], stats['count']):.4f} "
            f"stack={rate(stats['stack_exact_match'], stats['count']):.4f} "
            f"visited={rate(stats['visited_exact_match'], stats['count']):.4f}"
        )


if __name__ == "__main__":
    main()
