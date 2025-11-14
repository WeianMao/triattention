#!/usr/bin/env python3
"""Config-driven launcher for LazyEviction experiments (FullKV, Window_LAZY, SparseRound, etc.)."""
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

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONFIG = PROJECT_ROOT / "LazyEviction" / "weian_script" / "configs" / "sparse_prefill_keep_aime.yaml"
MERGE_SCRIPT = PROJECT_ROOT / "weian_development" / "merge_lazy_eviction_shards.py"
PATH_ARG_KEYS = {"output_dir", "model_path", "tokenizer_path", "sparse_stats_path"}


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
    log_path = log_dir / f"sparse_prefill_shard{shard_id:02d}.log"
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
) -> None:
    if not gpus:
        raise ValueError("No GPUs available to schedule shards")
    if dry_run:
        for shard_id in range(total_shards):
            gpu = gpus[shard_id % len(gpus)]
            log_path = (log_dir / f"sparse_prefill_shard{shard_id:02d}.log").resolve()
            cmd_preview = base_cmd + ["--shard_id", str(shard_id)]
            print(f"[dry-run] GPU {gpu}: {' '.join(cmd_preview)} | log -> {log_path}")
        return

    shard_queue: deque[int] = deque(range(total_shards))
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
                    raise RuntimeError(
                        f"Shard {shard.shard_id} on GPU {gpu} exited with code {return_code}."
                    )
                print(f"[done] shard {shard.shard_id} completed.")
                available.append(gpu)
    except KeyboardInterrupt:
        print("\nReceived Ctrl+C, terminating active shards...")
        terminate_active(active.values())
        raise


def run_merge(conda_env: str, method_output_dir: Path, merged_name: Optional[str], env: Dict[str, str], dry_run: bool) -> None:
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(MERGE_SCRIPT),
        "--method-output-dir",
        str(method_output_dir),
    ]
    if merged_name:
        cmd.extend(["--merged-dir-name", merged_name])
    merge_env = env.copy()
    merge_env.pop("CUDA_VISIBLE_DEVICES", None)
    if dry_run:
        print(f"[dry-run] merge command: {' '.join(cmd)}")
        return
    print(f"[merge] Aggregating shards under {method_output_dir}")
    subprocess.run(cmd, cwd=str(PROJECT_ROOT), check=True, env=merge_env)


def main() -> None:
    mask_process_command("PD-L1_binder")
    args = parse_args()
    config = load_config(args.config)
    experiment = config.get("experiment", config)

    runner_args = dict(experiment.get("runner_args", {}))
    if not runner_args:
        raise ValueError("runner_args must be provided in the config")
    runner_args.pop("num_shards", None)
    runner_args.pop("shard_id", None)
    if args.output_dir:
        override_key = "output-dir" if "output-dir" in runner_args else "output_dir"
        runner_args[override_key] = args.output_dir

    for key in PATH_ARG_KEYS:
        variants = {
            key,
            key.replace("_", "-"),
            key.replace("-", "_"),
        }
        for variant in variants:
            if variant in runner_args and runner_args[variant] is not None:
                runner_args[variant] = str(resolve_path(runner_args[variant]))

    runner_path_value = experiment.get("runner_path")
    if not runner_path_value:
        raise ValueError("runner_path must be provided in the config.")
    runner_path = resolve_path(runner_path_value)
    conda_env = experiment.get("conda_env", "lazy_evict")
    env_overrides = {k: str(v) for k, v in experiment.get("env", {}).items()}

    gpus = determine_gpus(args, experiment)
    if not gpus:
        raise RuntimeError("No GPUs available; specify --gpus or update the config.")

    if args.num_shards is not None:
        total_shards = args.num_shards
    else:
        num_cfg = experiment.get("num_shards")
        if isinstance(num_cfg, str) and num_cfg.strip().lower() == "auto":
            total_shards = len(gpus)
        elif num_cfg is None:
            total_shards = len(gpus)
        else:
            total_shards = int(num_cfg)
    if total_shards < 1:
        raise ValueError("num_shards must be >= 1")

    log_dir = resolve_path(args.log_dir or experiment.get("log_dir", "logs/lazy_eviction_sparse_round_prefill"))
    method_output_dir_value = (
        args.method_output_dir
        or experiment.get("method_output_dir")
        or runner_args.get("output_dir")
        or runner_args.get("output-dir")
    )
    if not method_output_dir_value:
        raise ValueError("method_output_dir must be provided (config or --method-output-dir).")
    method_output_dir = resolve_path(method_output_dir_value)
    merged_dir_name = experiment.get("merged_dir_name")

    base_env = prepare_environment(env_overrides)
    runner_arg_list = format_runner_args(runner_args, total_shards)
    base_cmd = build_base_command(conda_env, runner_path, runner_arg_list)

    print(f"Loaded config from {args.config}")
    print(f"GPUs: {gpus}")
    print(f"Total shards: {total_shards}")
    print(f"Logs -> {log_dir}")
    print(f"Method output dir -> {method_output_dir}")

    run_shards(gpus, total_shards, base_cmd, base_env, log_dir, args.dry_run)

    if args.skip_merge:
        print("Skipping merge step (requested).")
        return

    run_merge(conda_env, method_output_dir, merged_dir_name, base_env, args.dry_run)


if __name__ == "__main__":
    main()
