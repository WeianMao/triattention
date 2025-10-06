"""Visualize per-trace log probabilities with per-bin reductions."""
from __future__ import annotations

import argparse
import gzip
import json
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np

StatFn = Callable[[np.ndarray], float]

STAT_FUNCS: Dict[str, StatFn] = {
    "mean": np.nanmean,
    "median": np.nanmedian,
    "min": np.nanmin,
    "max": np.nanmax,
}


def load_logprobs(sequences_path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(sequences_path)
    return data["logprobs"], data["offsets"]


def parse_is_correct(trace_path: Path) -> bool:
    marker = '"is_correct":'
    marker_bytes = marker.encode("utf-8")
    buffer_size = 65536

    with trace_path.open("rb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            data = b""
            while True:
                chunk = gz.read(buffer_size)
                if not chunk:
                    break
                data += chunk
                if marker_bytes in data:
                    break
    text = data.decode("utf-8", errors="ignore")
    idx = text.find(marker)
    if idx != -1:
        tail = text[idx + len(marker):].lstrip()
        if tail.startswith("true"):
            return True
        if tail.startswith("false"):
            return False

    with trace_path.open("rb") as raw:
        with gzip.GzipFile(fileobj=raw, mode="rb") as gz:
            record = json.load(gz)
    return bool(record.get("is_correct", False))


def load_statuses(traces_dir: Path, num_traces: int) -> List[bool]:
    statuses: List[bool] = []
    for idx in range(num_traces):
        trace_path = traces_dir / f"trace_{idx:04d}.json.gz"
        if not trace_path.exists():
            raise FileNotFoundError(f"Missing trace file: {trace_path}")
        statuses.append(parse_is_correct(trace_path))
    return statuses


def reduce_trace(values: np.ndarray, bins: int, reducer: StatFn) -> np.ndarray:
    if values.size == 0:
        return np.full(bins, np.nan, dtype=np.float32)

    edges = np.linspace(0, values.size, bins + 1)
    reduced = np.empty(bins, dtype=np.float32)

    for i in range(bins):
        start = int(np.floor(edges[i]))
        end = int(np.floor(edges[i + 1]))

        if end <= start:
            end = min(start + 1, values.size)
        segment = values[start:end]
        if segment.size == 0:
            reduced[i] = reduced[i - 1] if i > 0 else float(values[start])
        else:
            with np.errstate(all="ignore"):
                reduced[i] = float(reducer(segment))
    return reduced


def collect_resampled(
    run_dir: Path,
    bins: int,
) -> Dict[str, Dict[str, List[np.ndarray]]]:
    sequences_path = run_dir / "trace_sequences.npz"
    traces_dir = run_dir / "traces"

    if not sequences_path.exists():
        raise FileNotFoundError(f"Missing sequences file: {sequences_path}")
    if not traces_dir.exists():
        raise FileNotFoundError(f"Missing traces directory: {traces_dir}")

    logprobs, offsets = load_logprobs(sequences_path)
    num_traces = len(offsets) - 1
    statuses = load_statuses(traces_dir, num_traces)

    result: Dict[str, Dict[str, List[np.ndarray]]] = {
        stat: {"correct": [], "incorrect": []} for stat in STAT_FUNCS
    }

    for idx in range(num_traces):
        start = int(offsets[idx])
        end = int(offsets[idx + 1])
        trace_vals = logprobs[start:end]

        for stat_name, reducer in STAT_FUNCS.items():
            reduced = reduce_trace(trace_vals, bins, reducer)
            key = "correct" if statuses[idx] else "incorrect"
            result[stat_name][key].append(reduced)

    return result


def plot_stat(
    run_dir: Path,
    stat_name: str,
    curves: Dict[str, List[np.ndarray]],
    bins: int,
) -> Path:
    x = np.arange(bins)
    plt.figure(figsize=(13, 6))
    plt.title(f"{run_dir.name} — {stat_name.capitalize()} logprobs (per trace)")
    plt.xlabel("Token index (resampled)")
    plt.ylabel("Log probability")

    for arr in curves["correct"]:
        plt.plot(x, arr, color="steelblue", alpha=0.15, linewidth=0.8)
    for arr in curves["incorrect"]:
        plt.plot(x, arr, color="crimson", alpha=0.35, linewidth=0.9)

    plt.plot([], [], color="steelblue", linewidth=2.0, label="Correct traces")
    plt.plot([], [], color="crimson", linewidth=2.0, label="Incorrect traces")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()

    out_dir = run_dir / "logprob_stats"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stat_name}_logprobs.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot per-trace log probabilities with bin reductions.")
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--bins", type=int, default=100)
    args = parser.parse_args()

    resampled = collect_resampled(args.run_dir, args.bins)
    outputs = []
    for stat_name, curves in resampled.items():
        outputs.append(plot_stat(args.run_dir, stat_name, curves, args.bins))
    for out_path in outputs:
        print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
