#!/usr/bin/env python3
"""Aggregate offline HF runner accuracy over serialized outputs."""
from __future__ import annotations

import argparse
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from development.serialization_utils import SerializationError, load_msgpack

ALLOWED_SUFFIXES = (".msgpack.gz", ".msgpack.zst", ".msgpack", ".pkl", ".pickle")
FILENAME_REGEX = re.compile(r"deepthink_offline(?:_sparse)?_qid(\d+)_rid([^_]+)_")


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute accuracy for HF offline runs")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--rid", required=True, type=str)
    parser.add_argument("--log-path", required=True, type=Path)
    parser.add_argument("--dataset", type=Path, default=None, help="Optional dataset jsonl to infer total qids")
    parser.add_argument(
        "--qid-list",
        type=str,
        default=None,
        help="Optional comma-separated qids to evaluate (overrides dataset size)",
    )
    return parser.parse_args(argv)


def extract_qid(name: str) -> Optional[int]:
    match = FILENAME_REGEX.search(name)
    if match:
        return int(match.group(1))
    return None


def load_result(path: Path) -> Optional[Dict]:
    suffixes = ''.join(path.suffixes)
    try:
        if suffixes.endswith((".pkl", ".pickle")):
            import pickle

            with path.open("rb") as handle:  # noqa: S301
                return pickle.load(handle)
        return load_msgpack(str(path))
    except (SerializationError, Exception) as exc:  # noqa: BLE001
        print(f"Failed to load {path.name}: {exc}")
        return None


def list_result_files(output_dir: Path, rid: str) -> List[Path]:
    pattern = f"deepthink_offline*_rid{rid}_*"
    return sorted(
        path
        for path in output_dir.glob(pattern)
        if path.is_file() and any(path.name.endswith(ext) for ext in ALLOWED_SUFFIXES)
    )


def iterable_to_set(text: Optional[str]) -> Optional[set[int]]:
    if not text:
        return None
    values = [token.strip() for token in text.split(",") if token.strip()]
    return {int(token) for token in values}


def load_expected_qids(dataset: Optional[Path], qid_list: Optional[str]) -> Optional[set[int]]:
    explicit = iterable_to_set(qid_list)
    if explicit is not None:
        return explicit
    if dataset is None:
        return None
    if not dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {dataset}")
    with dataset.open("r", encoding="utf-8") as handle:
        total = sum(1 for _ in handle)
    return set(range(total))


def summarize(results: List[Dict], expected_qids: Optional[set[int]]) -> Dict:
    processed_qids = {entry["qid"] for entry in results}
    correct = sum(1 for entry in results if entry["is_correct"])
    answered = len(results)
    accuracy = (correct / answered) if answered else 0.0
    summary: Dict[str, object] = {
        "answered": answered,
        "correct": correct,
        "accuracy": accuracy,
        "processed_qids": sorted(processed_qids),
    }
    if expected_qids is not None:
        missing = sorted(expected_qids - processed_qids)
        total_expected = len(expected_qids)
        coverage = (answered / total_expected) if total_expected else 0.0
        summary.update(
            {
                "expected_total": total_expected,
                "coverage": coverage,
                "missing_qids": missing,
            }
        )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    files = list_result_files(output_dir, args.rid)
    expected_qids = load_expected_qids(args.dataset, args.qid_list)

    results: List[Dict] = []
    for path in files:
        qid = extract_qid(path.name)
        if qid is None:
            continue
        if expected_qids is not None and qid not in expected_qids:
            continue
        payload = load_result(path)
        if payload is None:
            continue
        evaluation = payload.get("evaluation") or {}
        is_correct = bool(evaluation.get("is_correct"))
        prediction = (payload.get("predicted_answer") or payload.get("response") or "").strip()
        ground_truth = str(payload.get("ground_truth", "")).strip()
        results.append(
            {
                "qid": qid,
                "file": path.name,
                "is_correct": is_correct,
                "prediction": prediction,
                "ground_truth": ground_truth,
            }
        )

    summary = summarize(results, expected_qids)
    log_payload = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "rid": args.rid,
        "output_dir": str(output_dir),
        "summary": summary,
        "records": results,
    }

    args.log_path.parent.mkdir(parents=True, exist_ok=True)
    with args.log_path.open("w", encoding="utf-8") as handle:
        json.dump(log_payload, handle, ensure_ascii=False, indent=2)
    print(
        f"Accuracy log saved to {args.log_path} | answered={summary['answered']} correct={summary['correct']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
