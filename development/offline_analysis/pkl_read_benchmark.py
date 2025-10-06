#!/usr/bin/env python3
"""Utility for timing pickle load durations across large offline artifacts."""

import argparse
import pathlib
import pickle
import random
import statistics
import sys
import time
from typing import List, Optional


def resolve_files(root: pathlib.Path, pattern: str) -> List[pathlib.Path]:
    """Return sorted files matching the glob pattern under the root."""
    if root.is_file():
        return [root]
    return sorted(root.glob(pattern))


def pick_files(
    files: List[pathlib.Path],
    samples: Optional[int],
    seed: Optional[int],
) -> List[pathlib.Path]:
    """Select up to `samples` files from the list, optionally at random."""
    if not files:
        return []
    if samples is None or samples >= len(files):
        return files
    rng = random.Random(seed)
    return rng.sample(files, samples)


def load_once(path: pathlib.Path) -> tuple[float, str, str]:
    """Load the pickle file once and report duration, type, and len info."""
    start = time.perf_counter()
    with path.open("rb") as fh:
        obj = pickle.load(fh)
    duration = time.perf_counter() - start
    obj_type = type(obj).__name__
    try:
        obj_len = str(len(obj))
    except Exception:  # noqa: BLE001 - len() may not be defined
        obj_len = "n/a"
    return duration, obj_type, obj_len


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how long it takes to load .pkl files using pickle.load().",
    )
    parser.add_argument(
        "target",
        nargs="?",
        default=".",
        help="Directory containing PKL files or a direct file path (default: current directory).",
    )
    parser.add_argument(
        "--pattern",
        default="*.pkl",
        help="Glob pattern to match files when target is a directory (default: *.pkl).",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of files to sample randomly; use 0 to read none, omit to read one.",
    )
    parser.add_argument(
        "--repeat",
        type=int,
        default=1,
        help="Number of repeats per file to average load times (default: 1).",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Random seed for file sampling to make runs reproducible.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List the selected files without loading them.",
    )
    args = parser.parse_args()

    target_path = pathlib.Path(args.target).expanduser().resolve()
    if not target_path.exists():
        print(f"Target path not found: {target_path}", file=sys.stderr)
        raise SystemExit(1)

    files = resolve_files(target_path, args.pattern)
    if not files:
        print("No files matched the requested pattern.", file=sys.stderr)
        raise SystemExit(1)

    if args.samples == 0:
        selection = []
    else:
        selection = pick_files(files, args.samples, args.seed)

    print(f"Discovered {len(files)} candidate file(s).")
    if not selection:
        print("No files selected for reading (check --samples).")
        return

    print("Selected file(s):")
    for path in selection:
        size_bytes = path.stat().st_size
        size_gb = size_bytes / (1024 ** 3)
        print(f"  - {path} ({size_gb:.2f} GiB)")

    if args.dry_run:
        return

    repeat = max(1, args.repeat)
    for index, path in enumerate(selection, start=1):
        print(f"\n[{index}/{len(selection)}] Loading {path.name}")
        durations: List[float] = []
        obj_type = "n/a"
        obj_len = "n/a"
        for attempt in range(1, repeat + 1):
            duration, obj_type, obj_len = load_once(path)
            durations.append(duration)
            print(f"  repeat {attempt:02d}: {duration:.3f} s")
        if len(durations) == 1:
            stats_line = f"single run: {durations[0]:.3f} s"
        else:
            stats_line = (
                f"min {min(durations):.3f} s, "
                f"median {statistics.median(durations):.3f} s, "
                f"mean {statistics.mean(durations):.3f} s"
            )
        print(
            "  info: "
            f"type={obj_type}, len={obj_len}, size={path.stat().st_size / (1024 ** 3):.2f} GiB"
        )
        print(f"  stats: {stats_line}")


if __name__ == "__main__":
    main()
