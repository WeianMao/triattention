#!/usr/bin/env python3
"""Merge per-shard LazyEviction outputs back into a single metrics/predictions set."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def read_predictions(path: Path) -> List[Dict]:
    if not path.exists():
        return []
    records: List[Dict] = []
    with path.open() as fp:
        for line in fp:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    return records


def write_jsonl(path: Path, records: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fp:
        for record in records:
            fp.write(json.dumps(record, ensure_ascii=False) + "\n")


def merge_capacity(samples_dirs: List[Path]) -> Dict[str, float]:
    merged_predictions: List[Dict] = []
    total_samples = 0
    weighted_acc = 0.0
    weighted_cot = 0.0
    weighted_latency = 0.0

    for samples_dir in samples_dirs:
        preds = read_predictions(samples_dir / "predictions.jsonl")
        merged_predictions.extend(preds)
        metrics_path = samples_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        metrics = json.loads(metrics_path.read_text())
        n_samples = metrics.get("n_samples", 0)
        total_samples += n_samples
        weighted_acc += metrics.get("accuracy", 0.0) * n_samples
        weighted_cot += metrics.get("avg_cot_length", 0.0) * n_samples
        weighted_latency += metrics.get("sample_latency", 0.0) * n_samples

    merged_predictions.sort(key=lambda item: item.get("id", ""))
    return {
        "predictions": merged_predictions,
        "n_samples": total_samples,
        "weighted_accuracy": weighted_acc,
        "weighted_cot": weighted_cot,
        "weighted_latency": weighted_latency,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--method-output-dir",
        type=Path,
        required=True,
        help="Path to the method directory (ending with .../<method>) that contains shard_* subdirs",
    )
    parser.add_argument(
        "--merged-dir-name",
        type=str,
        default="shard_merged",
        help="Name of the directory that will hold merged outputs inside the method directory",
    )
    args = parser.parse_args()

    base_dir = args.method_output_dir.expanduser().resolve()
    if not base_dir.exists():
        raise FileNotFoundError(f"{base_dir} does not exist")

    shard_dirs = sorted([p for p in base_dir.iterdir() if p.is_dir() and p.name.startswith("shard_")])
    if not shard_dirs:
        raise RuntimeError(f"No shard_* directories found under {base_dir}")

    capacity_map: Dict[str, List[Path]] = {}
    for shard in shard_dirs:
        for subdir in shard.iterdir():
            if not subdir.is_dir() or subdir.name == "test_data":
                continue
            samples_dir = subdir / "samples"
            if not samples_dir.is_dir():
                continue
            capacity_map.setdefault(subdir.name, []).append(samples_dir)

    if not capacity_map:
        raise RuntimeError("No <capacity>/samples directories detected in shards")

    merged_root = base_dir / args.merged_dir_name
    merged_root.mkdir(parents=True, exist_ok=True)

    for capacity, samples_dirs in sorted(capacity_map.items()):
        merged = merge_capacity(samples_dirs)
        capacity_dir = merged_root / capacity / "samples"
        capacity_dir.mkdir(parents=True, exist_ok=True)
        write_jsonl(capacity_dir / "predictions.jsonl", merged["predictions"])

        if merged["n_samples"] == 0:
            continue
        metrics = {
            "n_samples": merged["n_samples"],
            "accuracy": merged["weighted_accuracy"] / merged["n_samples"],
            "avg_cot_length": merged["weighted_cot"] / merged["n_samples"],
            "sample_latency": merged["weighted_latency"] / merged["n_samples"],
        }
        (capacity_dir / "metrics.json").write_text(json.dumps(metrics, indent=4))

    print(f"Merged shards written under {merged_root}")


if __name__ == "__main__":
    main()
