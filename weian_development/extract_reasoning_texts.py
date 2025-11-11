"""Utilities to extract reasoning traces from DeepConf msgpack outputs."""
from __future__ import annotations

import argparse
import json
import gzip
import time
from contextlib import ExitStack
from pathlib import Path
from typing import Iterable, List, Dict, Any

import msgpack

MAGIC = b"DCMP"
SUPPORTED_VERSION = 1
METHOD_GZIP = b"G"
METHOD_NONE = b"N"
TARGET_FIELDS = {
    "question",
    "qid",
    "run_id",
    "ground_truth",
    "final_answer",
    "voted_answer",
}


def extract_reasoning_from_file(path: Path) -> Dict[str, Any]:
    """Stream a DeepConf msgpack bundle and collect reasoning texts."""
    start = time.perf_counter()
    with path.open("rb") as raw, ExitStack() as stack:
        magic = raw.read(len(MAGIC))
        if magic != MAGIC:
            raise RuntimeError(f"Unexpected magic header in {path}: {magic!r}")
        version = raw.read(1)
        if not version:
            raise RuntimeError(f"Missing version byte in {path}")
        if version != SUPPORTED_VERSION.to_bytes(1, "big"):
            raise RuntimeError(
                f"Unsupported serializer version {int.from_bytes(version, 'big')} in {path}"
            )
        method = raw.read(1)
        if method == METHOD_GZIP:
            stream = stack.enter_context(gzip.GzipFile(fileobj=raw))
        elif method == METHOD_NONE:
            stream = raw
        else:
            raise RuntimeError(f"Unsupported compression method {method!r} in {path}")

        unpacker = msgpack.Unpacker(stream, raw=False, strict_map_key=False)
        top_len = unpacker.read_map_header()

        metadata: Dict[str, Any] = {}
        traces: List[str] = []

        for _ in range(top_len):
            key = unpacker.unpack()
            if key == "all_traces":
                trace_count = unpacker.read_array_header()
                for _ in range(trace_count):
                    entry_len = unpacker.read_map_header()
                    text_value = None
                    for _ in range(entry_len):
                        field = unpacker.unpack()
                        if field == "text":
                            text_value = unpacker.unpack()
                        else:
                            unpacker.skip()
                    if text_value is not None:
                        traces.append(text_value)
            elif key in TARGET_FIELDS:
                metadata[key] = unpacker.unpack()
            else:
                unpacker.skip()

    elapsed = time.perf_counter() - start
    return {
        "source_file": path.name,
        "trace_count": len(traces),
        **{field: metadata.get(field) for field in TARGET_FIELDS},
        "traces": [
            {
                "index": idx + 1,
                "text": text,
            }
            for idx, text in enumerate(traces)
        ],
        "elapsed_seconds": elapsed,
    }


def extract_directory(src: Path, dst: Path) -> Iterable[Dict[str, Any]]:
    dst.mkdir(parents=True, exist_ok=True)
    summary = []

    for path in sorted(src.glob("*.msgpack.gz")):
        record = extract_reasoning_from_file(path)
        out_path = dst / (path.name.replace(".msgpack.gz", ".json"))
        with out_path.open("w", encoding="utf-8") as f:
            json.dump(record, f, ensure_ascii=False, indent=2)
        summary.append(
            {
                "file": path.name,
                "trace_count": record["trace_count"],
                "elapsed_seconds": round(record["elapsed_seconds"], 3),
                "output_file": out_path.name,
            }
        )
        yield summary[-1]

    summary_path = dst / "extraction_stats.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract reasoning traces from msgpack outputs")
    parser.add_argument("src", type=Path, help="Directory containing DeepConf msgpack outputs")
    parser.add_argument("dst", type=Path, help="Directory to write JSON reasoning traces")
    args = parser.parse_args()

    stats = list(extract_directory(args.src, args.dst))
    total_traces = sum(item["trace_count"] for item in stats)
    print(
        json.dumps(
            {
                "processed_files": len(stats),
                "total_traces": total_traces,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
