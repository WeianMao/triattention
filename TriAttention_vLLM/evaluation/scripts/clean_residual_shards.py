#!/usr/bin/env python3
"""Clean residual data from 8-shard runs mixed into 3-shard outputs.

For the 3-shard run:
  shard00: sample_idx 0-9   (already clean)
  shard01: sample_idx 10-19 (has residual 4-7)
  shard02: sample_idx 20-29 (has residual 8-11)
  shard03-07: entirely old 8-shard data -> remove

This script:
1. Filters shard01 run files to keep only sample_idx 10-19
2. Filters shard02 run files to keep only sample_idx 20-29
3. Removes shard03-07 directories entirely
"""
from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path


SHARDS_DIR = Path(
    "/data/rbg/users/weian/project/rl/dc/"
    "TriAttention_vLLM/evaluation/outputs/triattention_aime24/shards"
)

# Valid sample_idx ranges for the 3-shard configuration
VALID_RANGES = {
    "shard01": range(10, 20),
    "shard02": range(20, 30),
}

# Old 8-shard directories to remove entirely
OLD_SHARDS = ["shard03", "shard04", "shard05", "shard06", "shard07"]


def filter_jsonl(path: Path, valid_range: range) -> int:
    """Filter a jsonl file to keep only records with sample_idx in valid_range.

    Returns the number of records kept.
    """
    kept = []
    removed = 0
    with path.open() as f:
        for line in f:
            record = json.loads(line)
            if record["sample_idx"] in valid_range:
                kept.append(line.rstrip("\n"))
            else:
                removed += 1

    if removed > 0:
        with path.open("w") as f:
            for line in kept:
                f.write(line + "\n")
        print(f"  {path.name}: kept {len(kept)}, removed {removed}")
    else:
        print(f"  {path.name}: already clean ({len(kept)} records)")

    return len(kept)


def main() -> None:
    if not SHARDS_DIR.exists():
        print(f"ERROR: {SHARDS_DIR} does not exist")
        sys.exit(1)

    # Step 1: Filter shard01 and shard02
    for shard_name, valid_range in VALID_RANGES.items():
        shard_dir = SHARDS_DIR / shard_name
        print(f"\n=== Cleaning {shard_name} (keeping sample_idx {valid_range.start}-{valid_range.stop - 1}) ===")

        run_files = sorted(shard_dir.glob("run*.jsonl"))
        for run_file in run_files:
            count = filter_jsonl(run_file, valid_range)
            if count != 10:
                print(f"  WARNING: {run_file.name} has {count} records (expected 10)")

    # Step 2: Remove old 8-shard directories
    print("\n=== Removing old 8-shard directories ===")
    for shard_name in OLD_SHARDS:
        shard_dir = SHARDS_DIR / shard_name
        if shard_dir.exists():
            shutil.rmtree(shard_dir)
            print(f"  Removed {shard_name}/")
        else:
            print(f"  {shard_name}/ already absent")

    # Step 3: Verify
    print("\n=== Verification ===")
    all_ok = True
    for shard_name in ["shard00", "shard01", "shard02"]:
        shard_dir = SHARDS_DIR / shard_name
        run_files = sorted(shard_dir.glob("run*.jsonl"))
        for run_file in run_files:
            with run_file.open() as f:
                count = sum(1 for _ in f)
            if count != 10:
                print(f"  FAIL: {shard_name}/{run_file.name} has {count} records")
                all_ok = False
        print(f"  {shard_name}: {len(run_files)} run files, all 10 records each" if all_ok else "")

    remaining = sorted(d.name for d in SHARDS_DIR.iterdir() if d.is_dir())
    print(f"\n  Remaining shard dirs: {remaining}")

    if all_ok:
        print("\nAll clean. Ready for merge.")
    else:
        print("\nWARNING: Some files have unexpected record counts!")
        sys.exit(1)


if __name__ == "__main__":
    main()
