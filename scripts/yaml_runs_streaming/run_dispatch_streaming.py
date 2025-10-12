#!/usr/bin/env python3
"""Dispatch HuggingFace StreamingLLM offline runs using YAML configs."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = CURRENT_DIR.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from scripts.config_loader import PROJECT_ROOT as CONFIG_PROJECT_ROOT, load_config

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


MODE_SPEC = ModeSpec(
    name="hf_streaming_offline",
    script_relpath=Path("development/example_offline_streaming_hf.py"),
    config_section="hf_streaming_offline",
    result_prefix="deepthink_offline_qid",
)


def resolve_path(path_value: str | Path, must_exist: bool = False) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    if must_exist and not path.exists():
        raise FileNotFoundError(f"Required path not found: {path}")
    return path


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


def parse_qids(raw: str | None, total: int) -> List[int]:
    if not raw:
        return list(range(total))
    qids: List[int] = []
    for chunk in raw.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start_str, end_str = chunk.split("-", maxsplit=1)
            start = int(start_str)
            end = int(end_str)
            qids.extend(range(start, end + 1))
        else:
            qids.append(int(chunk))
    return sorted({qid for qid in qids if 0 <= qid < total})


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

    mapping = {
        "model_type": "--model_type",
        "reasoning_effort": "--reasoning_effort",
        "max_tokens": "--max_tokens",
        "temperature": "--temperature",
        "top_p": "--top_p",
        "top_k": "--top_k",
        "start_size": "--start_size",
        "recent_size": "--recent_size",
        "dtype": "--dtype",
        "device_map": "--device_map",
        "serializer": "--serializer",
        "compression_level": "--compression_level",
        "seed": "--seed",
        "attention_backend": "--attention_backend",
    }
    for key, flag in mapping.items():
        value = config.get(key)
        if value is not None:
            args.extend([flag, str(value)])

    if not config.get("multiple_voting", True):
        args.append("--no_multiple_voting")

    if not config.get("store_logprobs", True):
        args.append("--no_store_logprobs")

    return args


def build_mode_args(config: Dict) -> List[str]:
    args: List[str] = [
        "--budget",
        str(config.get("budget", 64)),
        "--window_size",
        str(config.get("window_size", 2048)),
    ]
    return args


def main() -> None:
    parser = argparse.ArgumentParser(description="StreamingLLM offline dispatcher")
    parser.add_argument("--config", required=True, help="YAML 配置路径")
    parser.add_argument("--rid", required=True, help="运行标识")
    parser.add_argument("--output-dir", required=True, help="输出目录")
    parser.add_argument("--gpus", help="设置 CUDA_VISIBLE_DEVICES，例如 0 或 0,1")
    parser.add_argument("--qids", help="指定 qid 范围，例如 0-9 或 1,3,5")
    parser.add_argument("--resume", action="store_true", help="跳过已存在结果的 qid")
    parser.add_argument("--max-workers", type=int, default=None, help="最大同时运行的 GPU 数")
    parser.add_argument(
        "--attention-backend",
        choices=["auto", "flash_attn2", "flash_attn3", "sdpa", "eager"],
        default="auto",
        help="覆盖 Streaming 推理脚本的注意力后端配置",
    )
    args = parser.parse_args()

    merged_conf, resolved_path = load_config(args.config, section=MODE_SPEC.config_section)
    if resolved_path:
        print(f"Loaded configuration from {resolved_path}")

    dataset_path = resolve_path(merged_conf["dataset"], must_exist=True)
    output_dir = resolve_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    total_questions = count_questions(dataset_path)
    qids = parse_qids(args.qids, total_questions)

    if args.resume:
        completed = collect_completed_qids(output_dir, args.rid, MODE_SPEC.result_prefix)
        qids = [qid for qid in qids if qid not in completed]

    if not qids:
        print("No qids to process.")
        return

    base_args = build_common_args(merged_conf, dataset_path, output_dir, args.rid)
    if args.attention_backend and args.attention_backend != "auto":
        base_args.extend(["--attention_backend", args.attention_backend])
    base_args += build_mode_args(merged_conf)

    gpu_ids = parse_gpu_argument(args.gpus)
    if not gpu_ids:
        raise ValueError("At least one GPU must be specified.")

    worker_count = args.max_workers or len(gpu_ids)
    worker_count = min(worker_count, len(gpu_ids))
    if worker_count <= 0:
        raise ValueError("max-workers must be positive if provided.")

    assignments = split_qids(qids, worker_count)

    script_path = MODE_SPEC.script_path
    env = os.environ.copy()

    def run_batch(gpu: str, batch: List[int]) -> None:
        if not batch:
            return
        gpu_env = env.copy()
        gpu_env["CUDA_VISIBLE_DEVICES"] = gpu
        gpu_env["PD_L1_AFFINITY_ALIAS"] = "1"
        for qid in batch:
            command = ["PD_L1_affinity", str(script_path)] + base_args + ["--qid", str(qid)]
            print(f"[GPU {gpu}] Running:", " ".join([sys.executable] + command[1:]))
            completed = subprocess.run(
                command,
                env=gpu_env,
                cwd=str(PROJECT_ROOT),
                executable=sys.executable,
            )
            if completed.returncode != 0:
                raise SystemExit(f"Command failed for qid {qid}: {' '.join(command)}")

    with ThreadPoolExecutor(max_workers=worker_count) as executor:
        futures = []
        for gpu, batch in zip(gpu_ids, assignments):
            futures.append(executor.submit(run_batch, gpu, batch))
        for future in as_completed(futures):
            future.result()


if __name__ == "__main__":
    main()
