#!/usr/bin/env python3
"""Sharded launcher for R-KV HuggingFace AIME runs (non-intrusive copy of LazyEviction dispatcher)."""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO

import yaml

from weian_development.process_utils import mask_process_command

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "R-KV" / "weian_script" / "configs" / "rkv_aime24_sharded.yaml"
MERGE_SCRIPT = PROJECT_ROOT / "R-KV" / "weian_development" / "merge_rkv_shards.py"
EVAL_SCRIPT = PROJECT_ROOT / "R-KV" / "HuggingFace" / "evaluation" / "eval_math.py"
PATH_ARG_KEYS = {"output_dir", "dataset_path", "model_path", "tokenizer_path"}


@dataclass
class ActiveShard:
    shard_id: int
    gpu: str
    process: subprocess.Popen
    log_handle: TextIO
    log_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML config file")
    parser.add_argument("--gpus", type=str, help="Comma/space separated GPU ids (overrides config)")
    parser.add_argument("--num-shards", type=int, dest="num_shards", help="Override total shard count")
    parser.add_argument("--log-dir", type=str, help="Override log directory")
    parser.add_argument("--output-dir", type=str, help="Override runner output_dir argument")
    parser.add_argument("--method-output-dir", type=str, help="Override merge target directory")
    parser.add_argument("--gpu-memory-threshold", type=int, help="Override GPU memory threshold for auto selection")
    parser.add_argument("--skip-merge", action="store_true", help="Skip shard merge step")
    parser.add_argument("--run-eval", action="store_true", help="Run eval_math.py on merged outputs")
    parser.add_argument("--eval-output-dir", type=str, help="Directory to write eval results")
    parser.add_argument("--dataset", type=str, default="aime24", help="Dataset name for eval script")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without launching processes")
    return parser.parse_args()


def load_config(path: Path) -> Dict:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open() as fp:
        data = yaml.safe_load(fp)
    if not isinstance(data, dict):
        raise ValueError("Config file must contain a mapping at the top level")
    return data


def parse_gpu_string(value: str) -> List[str]:
    tokens = value.replace(",", " ").split()
    return [token.strip() for token in tokens if token.strip()]


def auto_detect_gpus(threshold: int) -> List[str]:
    cmd = [
        "nvidia-smi",
        "--query-gpu=index,memory.used",
        "--format=csv,noheader,nounits",
    ]
    try:
        output = subprocess.check_output(cmd, text=True)
    except (FileNotFoundError, subprocess.CalledProcessError):
        return []
    detected: List[str] = []
    for line in output.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) < 2:
            continue
        if not parts[0].isdigit() or not parts[1].isdigit():
            continue
        if int(parts[1]) <= threshold:
            detected.append(parts[0])
    return detected


def determine_gpus(args: argparse.Namespace, experiment: Dict) -> List[str]:
    if args.gpus:
        gpus = parse_gpu_string(args.gpus)
        if gpus:
            return gpus
    cfg_gpus = experiment.get("gpus", [])
    fallback = experiment.get("auto_gpu_fallback", [])
    threshold = args.gpu_memory_threshold or experiment.get("gpu_memory_threshold", 200)
    if isinstance(cfg_gpus, str) and cfg_gpus.strip().lower() == "auto":
        detected = auto_detect_gpus(threshold)
        if detected:
            print(f"Auto-selected GPUs (<= {threshold} MiB used): {detected}")
            return detected
        if fallback:
            print("Auto GPU selection failed; falling back to config list.")
            return [str(item) for item in fallback]
        return []
    if isinstance(cfg_gpus, (list, tuple)):
        return [str(item) for item in cfg_gpus]
    if isinstance(cfg_gpus, int):
        return [str(cfg_gpus)]
    if isinstance(cfg_gpus, str):
        return parse_gpu_string(cfg_gpus)
    return []


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = (PROJECT_ROOT / path).resolve()
    return path


def format_runner_args(args_dict: Dict[str, object], total_shards: int) -> List[str]:
    formatted: List[str] = []
    for key, value in args_dict.items():
        if value is None:
            continue
        flag = f"--{key}"
        if key in PATH_ARG_KEYS:
            value = resolve_path(str(value))
        if isinstance(value, bool):
            formatted.extend([flag, "True" if value else "False"])
        elif isinstance(value, (list, tuple)):
            for item in value:
                formatted.extend([flag, str(item)])
        else:
            formatted.extend([flag, str(value)])
    formatted.extend(["--num_shards", str(total_shards)])
    return formatted


def build_base_command(conda_env: str, runner_path: Path, runner_args: List[str]) -> List[str]:
    return ["conda", "run", "-n", conda_env, "python", str(runner_path)] + runner_args


def prepare_environment(env_overrides: Dict[str, str]) -> Dict[str, str]:
    merged = os.environ.copy()
    for key, value in env_overrides.items():
        merged[key] = str(value)
    return merged


def launch_shard(
    gpu: str,
    shard_id: int,
    base_cmd: List[str],
    base_env: Dict[str, str],
    log_dir: Path,
) -> ActiveShard:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"rkv_aime24_shard{shard_id:02d}.log"
    shard_cmd = base_cmd + ["--shard_id", str(shard_id)]
    env = base_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_handle = log_path.open("w", buffering=1)
    print(f"[launch] shard {shard_id} -> GPU {gpu}, log {log_path}")
    process = subprocess.Popen(
        shard_cmd,
        cwd=str(PROJECT_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    return ActiveShard(shard_id=shard_id, gpu=gpu, process=process, log_handle=log_handle, log_path=log_path)


def terminate_active(active: Iterable[ActiveShard]) -> None:
    for shard in active:
        try:
            shard.process.terminate()
        except Exception:
            pass
    for shard in active:
        try:
            shard.process.wait(timeout=5)
        except Exception:
            shard.process.kill()
        finally:
            shard.log_handle.close()


def run_shards(
    gpus: List[str],
    total_shards: int,
    base_cmd: List[str],
    base_env: Dict[str, str],
    log_dir: Path,
    dry_run: bool,
    output_dir: Path,
) -> None:
    if not gpus:
        raise ValueError("No GPUs available to schedule shards")
    shards_to_run: List[int] = []
    for shard_id in range(total_shards):
        expected = output_dir / f"shard{shard_id:02d}.jsonl"
        if expected.exists() and expected.stat().st_size > 0:
            print(f"[skip] shard {shard_id} output exists -> {expected}")
            continue
        shards_to_run.append(shard_id)
    if not shards_to_run:
        print("All shard outputs already exist; skipping shard launch.")
        return
    if dry_run:
        for shard_id in shards_to_run:
            gpu = gpus[shard_id % len(gpus)]
            log_path = (log_dir / f"rkv_aime24_shard{shard_id:02d}.log").resolve()
            cmd_preview = base_cmd + ["--shard_id", str(shard_id)]
            print(f"[dry-run] shard {shard_id} -> GPU {gpu}\n  log: {log_path}\n  cmd: {' '.join(cmd_preview)}")
        return
    shard_queue: deque[int] = deque(shards_to_run)
    available: deque[str] = deque(gpus)
    active: Dict[str, ActiveShard] = {}
    try:
        while shard_queue or active:
            while shard_queue and available:
                gpu = available.popleft()
                shard_id = shard_queue.popleft()
                active[gpu] = launch_shard(gpu, shard_id, base_cmd, base_env, log_dir)
            if not active:
                continue
            time.sleep(5)
            finished = [gpu for gpu, shard in active.items() if shard.process.poll() is not None]
            for gpu in finished:
                shard = active.pop(gpu)
                return_code = shard.process.wait()
                shard.log_handle.close()
                if return_code != 0:
                    terminate_active(active.values())
                    raise RuntimeError(f"Shard {shard.shard_id} on GPU {gpu} exited with code {return_code}")
                print(f"[done] shard {shard.shard_id} completed.")
                available.append(gpu)
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, terminating active shards...")
        terminate_active(active.values())
        raise


def merge_outputs(shard_output_dir: Path, merged_dir_name: str, skip_merge: bool, dry_run: bool) -> None:
    if skip_merge:
        print("Skipping merge (--skip-merge enabled).")
        return
    if dry_run:
        cmd = [sys.executable, str(MERGE_SCRIPT), "--method-output-dir", str(shard_output_dir), "--merged-dir-name", merged_dir_name]
        print(f"[dry-run] merge command: {' '.join(cmd)}")
        return
    cmd = [sys.executable, str(MERGE_SCRIPT), "--method-output-dir", str(shard_output_dir), "--merged-dir-name", merged_dir_name]
    print(f"[merge] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))


def run_evaluation(base_dir: Path, dataset: str, exp_name: str, output_dir: Optional[Path], conda_env: str, dry_run: bool) -> None:
    if not base_dir.exists():
        print(f"[eval] skip, base_dir not found: {base_dir}")
        return
    if not any(base_dir.glob("*.jsonl")):
        print(f"[eval] skip, no jsonl under {base_dir}")
        return
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(EVAL_SCRIPT),
        "--base_dir",
        str(base_dir),
        "--dataset",
        dataset,
        "--exp_name",
        exp_name,
    ]
    if output_dir:
        cmd.extend(["--output_dir", str(output_dir)])
    if dry_run:
        print(f"[dry-run] eval command: {' '.join(cmd)}")
        return
    print(f"[eval] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(PROJECT_ROOT))


def main() -> None:
    mask_process_command("PD-L1_binder")
    args = parse_args()
    config = load_config(args.config)
    experiment = config.get("experiment", {})

    conda_env = experiment.get("conda_env", "rkv")
    runner_path = resolve_path(experiment["runner_path"])
    total_shards = args.num_shards or experiment.get("num_shards", 1)
    gpus = determine_gpus(args, experiment)
    log_dir = resolve_path(args.log_dir or experiment.get("log_dir", "R-KV/logs/rkv_aime24_sharded"))
    method_output_dir = resolve_path(args.method_output_dir or experiment.get("method_output_dir", "R-KV/outputs/rkv_aime24_sharded"))
    merged_dir_name = experiment.get("merged_dir_name", "merged")
    eval_output_dir = resolve_path(args.eval_output_dir) if args.eval_output_dir else None

    runner_args = experiment.get("runner_args", {}).copy()
    if args.output_dir:
        runner_args["output_dir"] = args.output_dir
    base_env = prepare_environment(experiment.get("env", {}))
    base_env.setdefault("VLLM_PROCESS_NAME_PREFIX", "PD-L1_binder")

    runner_args["output_dir"] = resolve_path(runner_args.get("output_dir", method_output_dir / "shards"))
    runner_args["dataset_path"] = resolve_path(runner_args["dataset_path"])
    runner_args["model_path"] = resolve_path(runner_args["model_path"])

    base_cmd = build_base_command(conda_env, runner_path, format_runner_args(runner_args, total_shards))

    run_shards(gpus, total_shards, base_cmd, base_env, log_dir, args.dry_run, runner_args["output_dir"])
    merge_outputs(runner_args["output_dir"], merged_dir_name, args.skip_merge, args.dry_run)
    merged_dir = runner_args["output_dir"].parent / merged_dir_name
    if args.run_eval:
        exp_name = experiment.get("name", merged_dir_name)
        run_evaluation(merged_dir, args.dataset, exp_name, eval_output_dir, conda_env, args.dry_run)


if __name__ == "__main__":
    main()
