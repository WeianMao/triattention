#!/usr/bin/env python3
"""Find the longest response text among DeepConf msgpack traces."""
from __future__ import annotations

import argparse
import sys
import types
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, Optional


def ensure_vllm_stub() -> None:
    try:
        import vllm.logprobs  # type: ignore  # noqa: F401
        return
    except Exception:  # noqa: BLE001
        pass

    vllm_module = types.ModuleType("vllm")
    logprobs_module = types.ModuleType("vllm.logprobs")

    class Logprob:
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


SUFFIXES = (".msgpack.gz", ".msgpack", ".msgpack.zst")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--pattern", type=str, default="*.msgpack.gz")
    parser.add_argument("--max-workers", type=int, default=6)
    return parser.parse_args()


def scan_file(path: Path) -> Optional[Dict[str, object]]:
    payload = load_msgpack(str(path))
    traces = payload.get("all_traces") or []
    best: Optional[Dict[str, object]] = None

    for idx, trace in enumerate(traces):
        text = trace.get("text") or ""
        if not isinstance(text, str):
            text = str(text)
        length = len(text)
        if best is None or length > best["length"]:
            best = {
                "length": length,
                "trace_index": idx,
                "text": text,
                "num_tokens": trace.get("num_tokens"),
                "stop_reason": trace.get("stop_reason"),
            }

    if best is None:
        return None

    best.update(
        {
            "file": path.name,
            "file_path": str(path),
            "qid": payload.get("qid"),
            "ground_truth": payload.get("ground_truth"),
        }
    )
    return best


def main() -> int:
    args = parse_args()
    candidates = [
        path
        for path in sorted(args.input_dir.glob(args.pattern))
        if path.is_file() and any(path.name.endswith(suffix) for suffix in SUFFIXES)
    ]
    if not candidates:
        print("No matching files found.")
        return 1

    best_overall: Optional[Dict[str, object]] = None

    if args.max_workers <= 1:
        for path in candidates:
            result = scan_file(path)
            if result and (best_overall is None or result["length"] > best_overall["length"]):
                best_overall = result
    else:
        with ProcessPoolExecutor(max_workers=args.max_workers) as executor:
            futures = {executor.submit(scan_file, path): path for path in candidates}
            for future in as_completed(futures):
                result = future.result()
                if result and (best_overall is None or result["length"] > best_overall["length"]):
                    best_overall = result

    if not best_overall:
        print("No trace texts found.")
        return 1

    print(
        "Longest response: length={length} chars | qid={qid} | file={file} | trace_index={trace_index}"
        .format(**best_overall)
    )
    print(f"File path: {best_overall['file_path']}")
    print(f"Stop reason: {best_overall['stop_reason']} | num_tokens={best_overall['num_tokens']}")
    preview = best_overall["text"][:200].replace("\n", " ")
    print(f"Preview: {preview}...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
