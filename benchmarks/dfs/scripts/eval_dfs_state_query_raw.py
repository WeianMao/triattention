#!/usr/bin/env python3
"""Evaluate merged DFS state query outputs into raw per-sample records."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable

from eval_dfs_state_query import evaluate_prediction, parse_json_response


def is_fully_correct(metrics: Dict) -> bool:
    """Check if all core metrics are correct."""
    return bool(
        metrics.get("current_node_correct")
        and metrics.get("stack_exact_match")
        and metrics.get("visited_exact_match")
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--merged-path",
        type=Path,
        help="Path to merged.jsonl (optional if --base-dir is provided).",
    )
    parser.add_argument(
        "--base-dir",
        type=Path,
        help="Merged directory containing merged.jsonl.",
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to DFS state query dataset JSON.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output path for raw JSONL records.",
    )
    parser.add_argument(
        "--max-samples",
        type=int,
        default=None,
        help="Optional cap on number of merged records to evaluate.",
    )
    return parser.parse_args()


def load_dataset_map(path: Path) -> Dict[int, Dict]:
    with path.open() as f:
        data = json.load(f)
    mapping: Dict[int, Dict] = {}
    for item in data:
        item_id = int(item.get("id", len(mapping)))
        mapping[item_id] = item
    return mapping


def resolve_merged_path(args: argparse.Namespace) -> Path:
    if args.merged_path:
        return args.merged_path
    if args.base_dir:
        return args.base_dir / "merged.jsonl"
    raise ValueError("Either --merged-path or --base-dir must be provided.")


def iter_records(path: Path) -> Iterable[Dict]:
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def compute_counts(records: list[Dict]) -> Dict[str, float]:
    if not records:
        return {}
    totals = {
        "current_node_correct": 0,
        "stack_exact_match": 0,
        "visited_exact_match": 0,
        "is_correct": 0,
        "parse_error": 0,
    }
    for record in records:
        metrics = record.get("metrics") or {}
        totals["current_node_correct"] += 1 if metrics.get("current_node_correct") else 0
        totals["stack_exact_match"] += 1 if metrics.get("stack_exact_match") else 0
        totals["visited_exact_match"] += 1 if metrics.get("visited_exact_match") else 0
        totals["is_correct"] += 1 if record.get("is_correct") else 0
        totals["parse_error"] += 1 if record.get("parse_error") else 0
    n = len(records)
    return {key: value / n for key, value in totals.items()}


def main() -> None:
    args = parse_args()
    merged_path = resolve_merged_path(args)
    if not merged_path.exists():
        raise FileNotFoundError(f"Merged file not found: {merged_path}")

    dataset_map = load_dataset_map(args.dataset)
    raw_records = []

    for idx, record in enumerate(iter_records(merged_path)):
        if args.max_samples and idx >= args.max_samples:
            break

        record_id = int(record.get("id", record.get("sample_idx", idx)))
        dataset_item = dataset_map.get(record_id, {})
        answer = record.get("answer") or dataset_item.get("answer")

        response = record.get("output", "")
        prediction = parse_json_response(response) if response else None
        parse_error = prediction is None

        metrics: Dict[str, object] = {}
        if not parse_error and answer is not None:
            metrics = evaluate_prediction(prediction, answer)

        metadata = record.get("metadata") or dataset_item.get("metadata") or {}
        graph = record.get("graph") or dataset_item.get("graph") or {}
        num_node = metadata.get("graph_nodes")
        if num_node is None:
            num_node = len(graph.get("nodes", []))
        num_edge = metadata.get("graph_edges")
        if num_edge is None:
            num_edge = len(graph.get("edges", []))

        raw_entry = {
            "problem_id": record_id,
            "num_node": num_node,
            "num_edge": num_edge,
            "num_step": record.get("steps", dataset_item.get("steps")),
            "total_dfs_steps": metadata.get("total_dfs_steps"),
            "graph_type": metadata.get("graph_type"),
            "action": metadata.get("action"),
            "is_correct": is_fully_correct(metrics) if metrics else False,
            "metrics": metrics,
            "parse_error": parse_error,
            "prediction": prediction,
            "ground_truth": answer,
            "response": response,
            "sample_idx": record.get("sample_idx"),
            "draw_idx": record.get("draw_idx"),
            "prefill_tokens": record.get("prefill_tokens"),
            "output_tokens": record.get("output_tokens"),
            "total_tokens": record.get("total_tokens"),
        }
        raw_records.append(raw_entry)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w") as f:
        for entry in raw_records:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    summary = compute_counts(raw_records)
    print(f"Wrote {len(raw_records)} raw records to {args.output}")
    if summary:
        print("Summary rates:")
        for key, value in summary.items():
            print(f"  {key}: {value:.4f}")


if __name__ == "__main__":
    main()
