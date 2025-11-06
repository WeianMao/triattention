"""Distributed runner for streaming perplexity evaluation."""
from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import os
import queue
from pathlib import Path
from typing import Dict, List, Optional

import sys

SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
WEIAN_DEV = PROJECT_ROOT / "weian_development"
if str(WEIAN_DEV) not in sys.path:
    sys.path.insert(0, str(WEIAN_DEV))

from weian_development.process_utils import mask_process_command
from weian_development.compute_reasoning_perplexity import (
    list_json_files,
    save_perplexity_artifacts,
)
from weian_development.streaming_perplexity.streaming_perplexity import (
    StreamingPerplexityConfig,
    StreamingPerplexityEvaluator,
)


def worker_main(
    worker_id: int,
    gpu_id: str,
    task_queue: "mp.Queue[str]",
    report_queue: "mp.Queue[Dict[str, str]]",
    config_dict: Dict[str, Optional[str]],
    output_dir: Path,
) -> None:
    alias = f"PD-L1_stream_{gpu_id}"[:15]
    mask_process_command(alias)
    os.environ["CUDA_VISIBLE_DEVICES"] = gpu_id
    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    config = StreamingPerplexityConfig(
        model_path=Path(config_dict["model_path"]),
        precision=config_dict["precision"],
        chunk_size=int(config_dict["chunk_size"]),
        model_type=config_dict["model_type"],
        reasoning_effort=config_dict["reasoning_effort"],
        limit_traces=(int(config_dict["limit_traces"]) if config_dict["limit_traces"] else None),
        device_override=config_dict["device_override"],
        stream_window=int(config_dict["stream_window"]),
    )

    evaluator = StreamingPerplexityEvaluator(config)

    try:
        while True:
            try:
                file_path_str = task_queue.get(timeout=1)
            except queue.Empty:
                continue
            if file_path_str is None:
                break
            file_path = Path(file_path_str)
            try:
                record = evaluator.evaluate_file(file_path)
                tensor_path = save_perplexity_artifacts(record, output_dir)
                summary = record["summary"]  # type: ignore[index]
                report_queue.put(
                    {
                        "gpu": gpu_id,
                        "file": file_path.name,
                        "output": tensor_path.name,
                        "tokens": str(summary.get("token_count", 0)),
                    }
                )
            except RuntimeError as exc:
                report_queue.put(
                    {
                        "gpu": gpu_id,
                        "file": file_path.name,
                        "output": "__error__",
                        "tokens": str(exc),
                    }
                )
                raise
    finally:
        evaluator.shutdown()
        report_queue.put(
            {
                "gpu": gpu_id,
                "file": "__worker_done__",
                "output": "",
                "tokens": "0",
            }
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Streaming perplexity runner")
    parser.add_argument("json_dir", type=Path, help="Directory containing reasoning JSON files")
    parser.add_argument("output_dir", type=Path, help="Directory to store perplexity tensors")
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B"),
    )
    parser.add_argument("--gpus", default="0", help="Comma-separated list of GPUs")
    parser.add_argument("--precision", choices=["float32", "float16", "bfloat16"], default="float16")
    parser.add_argument("--chunk-size", type=int, default=256)
    parser.add_argument("--stream-window", type=int, default=4096)
    parser.add_argument("--model-type", choices=["deepseek", "gpt"], default="deepseek")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument("--limit-files", type=int, default=None)
    parser.add_argument("--limit-traces", type=int, default=None)
    parser.add_argument("--device", default=None)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args()


def main() -> None:
    mask_process_command("PD-L1_stream_ctl")
    mp.set_start_method("spawn", force=True)
    args = parse_args()

    files = list_json_files(args.json_dir, args.limit_files)
    if not files:
        raise SystemExit(f"No JSON files found in {args.json_dir}")

    gpus = [gpu.strip() for gpu in args.gpus.split(",") if gpu.strip()]
    if not gpus:
        raise SystemExit("No GPUs specified")

    args.output_dir.mkdir(parents=True, exist_ok=True)

    if args.dry_run:
        print(f"Would process {len(files)} files across GPUs {gpus}")
        return

    task_queue: mp.Queue[str] = mp.Queue()
    report_queue: mp.Queue[Dict[str, str]] = mp.Queue()

    for file_path in files:
        task_queue.put(str(file_path))

    for _ in gpus:
        task_queue.put(None)

    config_dict = {
        "model_path": str(args.model_path),
        "precision": args.precision,
        "chunk_size": str(args.chunk_size),
        "model_type": args.model_type,
        "reasoning_effort": args.reasoning_effort,
        "limit_traces": str(args.limit_traces) if args.limit_traces is not None else "",
        "device_override": args.device,
        "stream_window": str(args.stream_window),
    }

    workers: List[mp.Process] = []
    for worker_id, gpu in enumerate(gpus):
        proc = mp.Process(
            target=worker_main,
            args=(worker_id, gpu, task_queue, report_queue, config_dict, args.output_dir),
            daemon=False,
        )
        proc.start()
        workers.append(proc)

    active_workers = len(workers)
    completed_files = 0
    total_files = len(files)

    while active_workers > 0:
        try:
            report = report_queue.get(timeout=5)
        except queue.Empty:
            continue

        if report["file"] == "__worker_done__":
            active_workers -= 1
            if args.verbose:
                print(f"Worker on GPU {report['gpu']} completed")
            continue

        if report["output"] == "__error__":
            for proc in workers:
                proc.terminate()
            raise SystemExit(
                f"Worker on GPU {report['gpu']} failed while processing {report['file']}: {report['tokens']}"
            )

        completed_files += 1
        if args.verbose:
            print(
                f"[{completed_files}/{total_files}] GPU {report['gpu']} processed {report['file']} -> {report['output']}"
            )

    for proc in workers:
        proc.join()
        if proc.exitcode != 0:
            raise SystemExit(f"Worker exited with code {proc.exitcode}")


if __name__ == "__main__":
    main()
