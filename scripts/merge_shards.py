#!/usr/bin/env python3
"""Merge TriAttention shard outputs (jsonl) sorted by sample_idx."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--method-output-dir", type=Path, required=True, help="Directory containing shard outputs")
    parser.add_argument("--merged-dir-name", type=str, default="merged", help="Name for merged subdir")
    parser.add_argument("--pattern", type=str, default="", help="Glob pattern for shard files (optional override)")
    return parser.parse_args()


def load_shard(path: Path, expect_meta: bool) -> List[Dict]:
    meta_path = path.with_suffix(".meta.json")
    if expect_meta:
        if not meta_path.exists():
            print(f"[merge] skip {path}, missing meta.")
            return []
        try:
            with meta_path.open() as fp:
                meta = json.load(fp)
        except Exception:
            print(f"[merge] skip {path}, meta unreadable.")
            return []
        if meta.get("status") != "complete":
            print(f"[merge] skip {path}, meta status={meta.get('status')}.")
            return []
    items: List[Dict] = []
    with path.open() as fp:
        for line in fp:
            items.append(json.loads(line))
    return items


def collect_shard_files(base_dir: Path, pattern_override: str | None = None) -> tuple[List[Path], bool]:
    if pattern_override:
        files = sorted(base_dir.glob(pattern_override))
        return files, False
    run_files = sorted(base_dir.glob("shard*/run*.jsonl"))
    if run_files:
        return run_files, True
    legacy = sorted(base_dir.glob("*.jsonl"))
    return legacy, False


def main() -> None:
    args = parse_args()
    shard_dir = args.method_output_dir
    merged_dir = shard_dir.parent / args.merged_dir_name
    merged_dir.mkdir(parents=True, exist_ok=True)
    shard_files, expect_meta = collect_shard_files(shard_dir, args.pattern or None)
    if not shard_files:
        raise FileNotFoundError(f"No shard files found under {shard_dir}")

    def sort_key(item: Dict) -> tuple:
        sample_idx = item.get("sample_idx", item.get("index", 0))
        draw_idx = item.get("draw_idx", 0)
        return (sample_idx, draw_idx)

    all_items: List[Dict] = []
    for path in shard_files:
        all_items.extend(load_shard(path, expect_meta))
    all_items.sort(key=sort_key)

    merged_path = merged_dir / "merged.jsonl"
    with merged_path.open("w") as fp:
        for item in all_items:
            fp.write(json.dumps(item, ensure_ascii=False) + "\n")
    print(f"Merged {len(all_items)} records into {merged_path}")


if __name__ == "__main__":
    main()
