#!/usr/bin/env python3
"""Dispatch DeepConf runners with a HuggingFace-based offline option."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_loader import PROJECT_ROOT as CONFIG_PROJECT_ROOT, load_config
from weian_development.process_utils import mask_process_command

PROJECT_ROOT = CONFIG_PROJECT_ROOT


@dataclass(frozen=True)
class ModeSpec:
    name: str
    script_relpath: Path
    config_section: str
    result_prefix: str

    @property
    def script_path(self) -> Path:
        return PROJECT_ROOT / self.script_relpath


MODE_SPECS: Dict[str, ModeSpec] = {
    "online": ModeSpec(
        name="online",
        script_relpath=Path("examples/example_online.py"),
        config_section="online",
        result_prefix="deepthink_online_qid",
    ),
    "baseline": ModeSpec(
        name="baseline",
        script_relpath=Path("examples/example_online_baseline.py"),
        config_section="baseline",
        result_prefix="deepthink_online_baseline_qid",
    ),
    "offline": ModeSpec(
        name="offline",
        script_relpath=Path("weian_development/hf_offline_runner/example_offline_hf_serialized.py"),
        config_section="offline",
        result_prefix="deepthink_offline_qid",
    ),
}


def parse_gpu_argument(raw: str | None) -> List[str]:
    if raw:
        tokens = [token.strip() for token in raw.replace(" ", ",").split(",") if token.strip()]
        if not tokens:
            raise ValueError("GPU list is empty; please provide at least one device ID.")
        return tokens

    env_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env_devices:
        tokens = [token.strip() for token in env_devices.split(",") if token.strip()]
        if tokens:
            return tokens

    return [str(index) for index in range(8)]


def count_questions(dataset_path: Path) -> int:
    with dataset_path.open("r", encoding="utf-8") as handle:
        return sum(1 for _ in handle)


def collect_completed_qids(outputs_dir: Path, rid: str, prefix: str) -> set[int]:
    if not outputs_dir.exists():
        return set()

    marker = f"_rid{rid}_"
    completed: set[int] = set()

    for entry in outputs_dir.iterdir():
        if not entry.is_file():
            continue
        name = entry.name
        if not name.startswith(prefix) or marker not in name:
            continue
        try:
            qid_str = name[len(prefix): name.index("_", len(prefix))]
            completed.add(int(qid_str))
        except (ValueError, IndexError):
            continue

    return completed


def split_qids(qids: Sequence[int], slots: int) -> List[List[int]]:
    if slots <= 0:
        raise ValueError("Number of GPU slots must be positive.")

    if not qids:
        return [[] for _ in range(slots)]

    base = len(qids) // slots
    remainder = len(qids) % slots
    batches: List[List[int]] = []
    start = 0
    for index in range(slots):
        extra = 1 if index < remainder else 0
        stop = start + base + extra
        batches.append(list(qids[start:stop]))
        start = stop
    return batches


def resolve_path(path_value: str | Path, must_exist: bool = False) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")
    return path


def build_common_args(config: Dict, dataset_path: Path, output_dir: Path, rid: str) -> List[str]:
    args: List[str] = [
        "--model",
        str(config["model"]),
        "--dataset",
        str(dataset_path),
        "--rid",
        rid,
        "--output_dir",
        str(output_dir),
    ]

    tensor_parallel = config.get("tensor_parallel_size")
    if tensor_parallel is not None:
        args.extend(["--tensor_parallel_size", str(tensor_parallel)])

    model_type = config.get("model_type")
    if model_type:
        args.extend(["--model_type", str(model_type)])

    max_tokens = config.get("max_tokens")
    if max_tokens is not None:
        args.extend(["--max_tokens", str(max_tokens)])
        args.extend(["--max_model_len", str(max_tokens)])

    sampling = config.get("sampling", {})
    for key in ("temperature", "top_p", "top_k"):
        value = sampling.get(key)
        if value is not None:
            args.extend([f"--{key}", str(value)])

    vllm_conf = config.get("vllm", {})
    gpu_mem_util = vllm_conf.get("gpu_memory_utilization")
    if gpu_mem_util is not None:
        args.extend(["--gpu_memory_utilization", str(gpu_mem_util)])

    if not config.get("multiple_voting", True):
        args.append("--no_multiple_voting")

    return args


def build_mode_specific_args(mode: str, config: Dict) -> List[str]:
    args: List[str] = []
    if mode == "online":
        args.extend([
            "--warmup_traces",
            str(config.get("warmup_traces", 16)),
            "--total_budget",
            str(config.get("total_budget", 256)),
            "--confidence_percentile",
            str(config.get("confidence_percentile", 90)),
            "--window_size",
            str(config.get("window_size", 2048)),
        ])
    elif mode == "baseline":
        args.extend([
            "--budget",
            str(config.get("budget", config.get("total_budget", 256))),
            "--window_size",
            str(config.get("window_size", 2048)),
        ])
    elif mode == "offline":
        args.extend([
            "--budget",
            str(config.get("budget", 512)),
            "--window_size",
            str(config.get("window_size", 2048)),
        ])
        reasoning_effort = config.get("reasoning_effort")
        if reasoning_effort:
            args.extend(["--reasoning_effort", str(reasoning_effort)])
    else:
        raise ValueError(f"Unsupported mode '{mode}'")
    return args


def build_base_command(spec: ModeSpec, config: Dict, dataset_path: Path, output_dir: Path, rid: str) -> List[str]:
    if "model" not in config:
        raise KeyError(f"'model' must be specified in section '{spec.config_section}'")
    command = [sys.executable, str(spec.script_path)]
    command.extend(build_common_args(config, dataset_path, output_dir, rid))
    command.extend(build_mode_specific_args(spec.name, config))
    return command


def launch_worker(
    gpu_id: str,
    qid_batch: Sequence[int],
    base_command: Sequence[str],
    extra_args: Sequence[str],
) -> None:
    if not qid_batch:
        print(f"[GPU {gpu_id}] No questions assigned; skipping.")
        return

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu_id

    for qid in qid_batch:
        cmd = list(base_command) + ["--qid", str(qid)] + list(extra_args)
        print(f"[GPU {gpu_id}] Starting question {qid}: {' '.join(cmd)}")
        subprocess.run(cmd, env=env, cwd=str(PROJECT_ROOT), check=True)
        print(f"[GPU {gpu_id}] Finished question {qid}.")


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Dispatch DeepConf example runs")
    parser.add_argument("--mode", choices=sorted(MODE_SPECS.keys()), default="offline")
    parser.add_argument("--config", type=str, default="scripts/configs/deepseek_r1_qwen3_8b.yaml")
    parser.add_argument("--rid", type=str, default="0")
    parser.add_argument("--gpus", type=str, default=None)
    parser.add_argument("--max-workers", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory for result files")
    parser.add_argument("--qids", type=str, default=None, help="Optional comma-separated QIDs to run")
    return parser.parse_known_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    mask_process_command("PD-L1_binder")

    args, extra_args = parse_args(argv)

    spec = MODE_SPECS[args.mode]

    config, resolved_config = load_config(args.config, section=spec.config_section)
    if resolved_config:
        print(f"Loaded configuration from {resolved_config}")

    if "dataset" not in config:
        raise KeyError(f"'dataset' must be specified in section '{spec.config_section}'")
    dataset = resolve_path(config["dataset"], must_exist=True)

    if args.output_dir:
        config = dict(config)
        config["output_dir"] = args.output_dir
    output_dir_value = config.get("output_dir", "outputs")
    output_dir = resolve_path(output_dir_value)
    output_dir.mkdir(parents=True, exist_ok=True)

    gpu_ids = parse_gpu_argument(args.gpus)
    if not gpu_ids:
        raise ValueError("At least one GPU must be specified.")

    total_questions = count_questions(dataset)
    if total_questions == 0:
        print("Dataset is empty. Nothing to schedule.")
        return 0

    if args.qids:
        qids = sorted({int(qid.strip()) for qid in args.qids.split(",") if qid.strip()})
    else:
        qids = list(range(total_questions))

    completed = collect_completed_qids(output_dir, args.rid, spec.result_prefix)
    if completed:
        print("Skipping completed questions:", ", ".join(str(qid) for qid in sorted(completed)))
    qids = [qid for qid in qids if qid not in completed]

    if not qids:
        print("All requested questions already processed.")
        return 0

    assignments = split_qids(qids, len(gpu_ids))

    worker_count = args.max_workers or len(gpu_ids)
    worker_count = min(worker_count, len(gpu_ids))
    if worker_count <= 0:
        raise ValueError("max-workers must be positive if provided.")

    base_command = build_base_command(spec, config, dataset, output_dir, args.rid)

    plan = [
        (gpu_ids[index], batch)
        for index, batch in enumerate(assignments)
        if batch
    ]

    if not plan:
        print("No batches assigned; exiting.")
        return 0

    print(f"Dispatch plan for {spec.name} mode:")
    for gpu, batch in plan:
        print(f"  GPU {gpu}: questions {batch[0]}..{batch[-1]} ({len(batch)} total)")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = [
            executor.submit(launch_worker, gpu, batch, base_command, extra_args)
            for gpu, batch in plan
        ]
        for future in as_completed(futures):
            future.result()

    print("All batches completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
