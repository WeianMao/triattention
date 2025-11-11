#!/usr/bin/env python3
"""Compute per-file and average accuracy from DeepConf msgpack outputs."""
from __future__ import annotations

import argparse
import json
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List


def ensure_vllm_stub() -> None:
    """Provide a minimal vllm.logprobs.Logprob stub if vllm is unavailable."""
    try:
        import vllm.logprobs  # type: ignore  # noqa: F401
        return
    except Exception:  # noqa: BLE001 - any import failure should trigger the stub
        pass

    vllm_module = types.ModuleType("vllm")
    logprobs_module = types.ModuleType("vllm.logprobs")

    class Logprob:  # noqa: D401 - simple data holder
        """Replacement for vllm.logprobs.Logprob holding basic fields."""

        def __init__(self, logprob, rank, decoded_token):
            self.logprob = logprob
            self.rank = rank
            self.decoded_token = decoded_token

    logprobs_module.Logprob = Logprob  # type: ignore[attr-defined]
    vllm_module.logprobs = logprobs_module  # type: ignore[attr-defined]
    sys.modules.setdefault("vllm", vllm_module)
    sys.modules.setdefault("vllm.logprobs", logprobs_module)


ensure_vllm_stub()

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from development.serialization_utils import load_msgpack  # noqa: E402  - depends on stub


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True, help="Directory containing msgpack outputs")
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.msgpack.gz",
        help="Glob pattern (relative to input dir) to match result files",
    )
    parser.add_argument(
        "--output-file",
        type=Path,
        required=True,
        help="Path to write JSON statistics (directories created if needed)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=4,
        help="Parallel workers for decoding (set 1 to disable multiprocessing)",
    )
    return parser.parse_args()


def load_accuracy(path: Path) -> Dict[str, object]:
    payload = load_msgpack(str(path))
    traces = payload.get("all_traces") or []
    ground_truth = str(payload.get("ground_truth", "")).strip()

    total_traces = len(traces)
    answered = 0
    correct = 0

    for trace in traces:
        extracted = trace.get("extracted_answer")
        if extracted is None:
            continue
        answered += 1
        if str(extracted).strip() == ground_truth:
            correct += 1

    accuracy = (correct / total_traces) if total_traces else 0.0

    result: Dict[str, object] = {
        "file": path.name,
        "qid": payload.get("qid"),
        "ground_truth": ground_truth,
        "total_traces": total_traces,
        "answered_traces": answered,
        "correct_traces": correct,
        "accuracy": accuracy,
    }
    return result


def main() -> int:
    args = parse_args()
    files = sorted(args.input_dir.glob(args.pattern))
    stats: List[Dict[str, object]] = []

    valid_files = [
        path
        for path in files
        if path.is_file() and any(path.name.endswith(suffix) for suffix in (".msgpack.gz", ".msgpack", ".msgpack.zst"))
    ]

    if not valid_files:
        print("No matching files found.")
        return 1

    if args.max_workers <= 1:
        for path in valid_files:
            stats.append(load_accuracy(path))
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(load_accuracy, path): path for path in valid_files}
            for future in as_completed(futures):
                stats.append(future.result())

    stats.sort(key=lambda entry: (entry.get("qid"), entry["file"]))

    mean_accuracy = sum(entry["accuracy"] for entry in stats) / len(stats) if stats else 0.0
    output_payload = {
        "input_dir": str(args.input_dir),
        "pattern": args.pattern,
        "files_processed": len(stats),
        "mean_accuracy": mean_accuracy,
        "details": stats,
    }

    args.output_file.parent.mkdir(parents=True, exist_ok=True)
    with args.output_file.open("w", encoding="utf-8") as handle:
        json.dump(output_payload, handle, ensure_ascii=False, indent=2)

    print(
        f"Processed {len(stats)} files from {args.input_dir} | mean_accuracy={mean_accuracy:.4f} | "
        f"wrote stats to {args.output_file}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
