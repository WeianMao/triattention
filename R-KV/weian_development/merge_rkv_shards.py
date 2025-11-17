#!/usr/bin/env python3
"""Merge R-KV shard outputs (jsonl) sorted by sample_idx."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Dict


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-output-dir", type=Path, required=True, help="Directory containing shard outputs")
    parser.add_argument("--merged-dir-name", type=str, default="merged", help="Name for merged subdir")
    parser.add_argument("--pattern", type=str, default="*.jsonl", help="Glob pattern for shard files")
    return parser.parse_args()


def load_shard(path: Path) -> List[Dict]:
    items: List[Dict] = []
    with path.open() as fp:
        for line in fp:
            items.append(json.loads(line))
    return items


def main() -> None:
    args = parse_args()
    shard_dir = args.method_output_dir
    merged_dir = shard_dir.parent / args.merged_dir_name
    merged_dir.mkdir(parents=True, exist_ok=True)
    shard_files = sorted(shard_dir.glob(args.pattern))
    if not shard_files:
        raise FileNotFoundError(f"No shard files matching {args.pattern} under {shard_dir}")

    all_items: List[Dict] = []
    for path in shard_files:
        all_items.extend(load_shard(path))
    all_items.sort(key=lambda x: x.get("index", x.get("sample_idx", 0)))

    merged_path = merged_dir / "merged.jsonl"
    with merged_path.open("w") as fp:
        for item in all_items:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Merged {len(all_items)} records into {merged_path}")


if __name__ == "__main__":
    main()
