#!/usr/bin/env python3
"""
Verify uniform step distribution for DFS state query datasets.

Usage:
    python dfs_state_query/scripts/verify_step_uniformity.py \
        --dataset dfs_state_query/datasets/dfs_state_query_small.json
"""

import argparse
import json
from collections import Counter
from pathlib import Path


def load_dataset(path: Path):
    with path.open("r") as f:
        return json.load(f)


def main():
    parser = argparse.ArgumentParser(
        description="Check uniform distribution of step counts in a dataset"
    )
    parser.add_argument("--dataset", type=str, required=True, help="Dataset JSON path")
    parser.add_argument(
        "--min-steps",
        type=int,
        default=None,
        help="Minimum step to check (defaults to dataset min)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum step to check (defaults to dataset max)",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=1,
        help="Allowed max-min count gap across steps",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=None,
        help="Require every step to have this exact count",
    )

    args = parser.parse_args()
    dataset_path = Path(args.dataset)
    dataset = load_dataset(dataset_path)

    if not dataset:
        print("ERROR: Empty dataset.")
        raise SystemExit(1)

    step_counts = Counter(item["steps"] for item in dataset)
    min_step = args.min_steps if args.min_steps is not None else min(step_counts)
    max_step = args.max_steps if args.max_steps is not None else max(step_counts)

    counts = []
    for step in range(min_step, max_step + 1):
        counts.append(step_counts.get(step, 0))

    print(f"Dataset: {dataset_path}")
    print(f"Step range: {min_step}-{max_step}")
    for step in range(min_step, max_step + 1):
        print(f"  step {step}: {step_counts.get(step, 0)}")

    if args.expected_count is not None:
        ok = all(count == args.expected_count for count in counts)
        if not ok:
            print(
                "FAIL: Expected count mismatch. "
                f"expected={args.expected_count}, "
                f"min={min(counts)}, max={max(counts)}"
            )
            raise SystemExit(1)
    else:
        gap = max(counts) - min(counts)
        if gap > args.tolerance:
            print(
                "FAIL: Non-uniform distribution. "
                f"min={min(counts)}, max={max(counts)}, tolerance={args.tolerance}"
            )
            raise SystemExit(1)

    print("PASS: Uniform step distribution.")


if __name__ == "__main__":
    main()
