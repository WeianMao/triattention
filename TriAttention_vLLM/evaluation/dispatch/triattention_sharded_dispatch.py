#!/usr/bin/env python3
"""Sharded launcher for TriAttention vLLM evaluation (adapted from R-KV dispatcher)."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO

import yaml

# Project roots
TRIATTENTION_ROOT = Path(__file__).resolve().parents[2]  # TriAttention_vLLM/
DC_ROOT = TRIATTENTION_ROOT.parent  # dc/
RKV_ROOT = DC_ROOT / "R-KV"

# Add paths for imports
if str(TRIATTENTION_ROOT) not in sys.path:
    sys.path.insert(0, str(TRIATTENTION_ROOT))
if str(RKV_ROOT) not in sys.path:
    sys.path.insert(0, str(RKV_ROOT))

from weian_development.process_utils import mask_process_command

DEFAULT_CONFIG = TRIATTENTION_ROOT / "evaluation" / "dispatch" / "configs" / "triattention_aime24.yaml"
MERGE_SCRIPT = TRIATTENTION_ROOT / "evaluation" / "merge" / "merge_shards.py"
MULTI_EVAL_SCRIPT = RKV_ROOT / "HuggingFace" / "evaluation" / "eval_math_multi.py"
PATH_ARG_KEYS = {"output_dir", "dataset_path", "model_path", "tokenizer_path", "sparse_stats_path"}
RUNNER_EXCLUDE_KEYS = {"num_samples_by_dataset"}


@dataclass
class ActiveShard:
    shard_id: int
    gpu: str
    process: subprocess.Popen
    log_handle: TextIO
    log_path: Path
    last_log_size: int
    last_log_update_ts: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="Path to YAML config file")
    parser.add_argument("--gpus", type=str, help="Comma/space separated GPU ids (overrides config)")
    parser.add_argument("--num-shards", type=int, dest="num_shards", help="Override total shard count")
    parser.add_argument("--log-dir", type=str, help="Override log directory")
    parser.add_argument("--output-dir", type=str, help="Override runner output_dir argument")
    parser.add_argument("--method-output-dir", type=str, help="Override merge target directory")
    parser.add_argument("--gpu-memory-threshold", type=int, help="Override GPU memory threshold for auto selection")
    parser.add_argument(
        "--stall-timeout-minutes",
        type=float,
        default=None,
        help="Fail fast if a shard log has no growth for this many minutes (<=0 disables).",
    )
    parser.add_argument("--skip-merge", action="store_true", help="Skip shard merge step")
    parser.add_argument(
        "--skip-existing",
        dest="skip_existing",
        action="store_true",
        default=True,
        help="Skip shards whose outputs already exist (default: enabled).",
    )
    parser.add_argument(
        "--no-skip-existing",
        dest="skip_existing",
        action="store_false",
        help="Force rerun shards even if outputs exist.",
    )
    parser.add_argument("--no-eval", action="store_true", help="Skip eval_math.py after merge")
    parser.add_argument("--eval-output-dir", type=str, help="Directory to write eval results")
    parser.add_argument("--dataset", type=str, default="aime24", help="Dataset name for eval script")
    parser.add_argument("--dry-run", action="store_true", help="Print what would run without launching processes")
    parser.add_argument(
        "--allow-legacy-v1",
        action="store_true",
        help="Allow legacy V1 runner config. Default behavior enforces current TriAttention runner.",
    )
    # TriAttention-specific args
    parser.add_argument(
        "--kv-budget",
        type=int,
        default=None,
        help="Override runner arg: KV cache budget.",
    )
    parser.add_argument(
        "--pruning-mode",
        type=str,
        default=None,
        choices=["per_head", "per_layer", "per_layer_per_head"],
        help="Override runner arg: pruning mode.",
    )
    parser.add_argument(
        "--divide-length",
        type=int,
        default=None,
        help="Override runner arg: compression trigger interval.",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=None,
        help="Override runner arg: recent token window size.",
    )
    parser.add_argument(
        "--enforce-eager",
        type=str,
        default=None,
        choices=["true", "false"],
        help="Override runner arg: enforce eager execution.",
    )
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


def compute_local_runs(total_runs: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    base = total_runs // num_shards
    extra = total_runs % num_shards
    start = shard_id * base + min(shard_id, extra)
    count = base + (1 if shard_id < extra else 0)
    return start, count


def shard_run_dir(base_dir: Path, shard_id: int) -> Path:
    return base_dir / f"shard{shard_id:02d}"


def run_paths(base_dir: Path, shard_id: int, run_id: int) -> tuple[Path, Path]:
    run_dir = shard_run_dir(base_dir, shard_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    run_path = run_dir / f"run{run_id:03d}.jsonl"
    meta_path = run_dir / f"run{run_id:03d}.meta.json"
    return run_path, meta_path


def run_completed(base_dir: Path, shard_id: int, run_id: int, expected_records: int) -> bool:
    run_path, meta_path = run_paths(base_dir, shard_id, run_id)
    if not run_path.exists() or run_path.stat().st_size == 0 or not meta_path.exists():
        return False
    try:
        with meta_path.open() as meta_fp:
            meta = json.load(meta_fp)
        status_ok = meta.get("status") == "complete"
        recorded = int(meta.get("records", -1))
    except Exception:
        return False
    if not status_ok:
        return False
    if expected_records > 0 and recorded >= 0 and recorded < expected_records:
        return False
    if expected_records <= 0:
        return True
    try:
        with run_path.open() as fp:
            lines = sum(1 for _ in fp)
        return lines >= expected_records
    except Exception:
        return False


def count_dataset_examples(dataset_path: Path, max_examples: int | None = None) -> int:
    count = 0
    with dataset_path.open("r", encoding="utf-8") as fp:
        for count, _ in enumerate(fp, start=1):
            if max_examples is not None and count >= max_examples:
                return max_examples
    return count


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
    threshold = args.gpu_memory_threshold or experiment.get("gpu_memory_threshold", 9000)
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


def resolve_path(value: str | Path, base_root: Path = DC_ROOT) -> Path:
    """Resolve path relative to project root."""
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    # Handle R-KV/ prefixed paths
    if parts and parts[0] == "R-KV":
        return (RKV_ROOT / Path(*parts[1:])).resolve() if len(parts) > 1 else RKV_ROOT.resolve()
    # Handle TriAttention_vLLM/ prefixed paths
    if parts and parts[0] == "TriAttention_vLLM":
        return (TRIATTENTION_ROOT / Path(*parts[1:])).resolve() if len(parts) > 1 else TRIATTENTION_ROOT.resolve()
    return (base_root / path).resolve()


def format_runner_args(args_dict: Dict[str, object], total_shards: int) -> List[str]:
    formatted: List[str] = []
    for key, value in args_dict.items():
        if value is None:
            continue
        if key in RUNNER_EXCLUDE_KEYS:
            continue
        flag = f"--{key.replace('_', '-')}"
        if key in PATH_ARG_KEYS:
            value = resolve_path(str(value))
        if isinstance(value, bool):
            # Runner uses str2bool type, so pass true/false as string value
            formatted.extend([flag, str(value).lower()])
        elif isinstance(value, (list, tuple)):
            for item in value:
                formatted.extend([flag, str(item)])
        else:
            formatted.extend([flag, str(value)])
    formatted.extend(["--num-shards", str(total_shards)])
    return formatted


def build_base_command(conda_env: str, runner_path: Path, runner_args: List[str]) -> List[str]:
    """Build shard launch command.

    Default path keeps existing conda-run behavior.
    Set TRIATTN_DISPATCH_PYTHON_BIN to bypass conda and launch with a direct
    interpreter path (useful when conda lock contention stalls multi-shard jobs).
    """
    direct_python = os.environ.get("TRIATTN_DISPATCH_PYTHON_BIN", "").strip()
    if direct_python:
        return [direct_python, str(runner_path)] + runner_args
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
    log_stamp: str,
) -> ActiveShard:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"triattention_shard{shard_id:02d}_{log_stamp}.log"
    shard_cmd = base_cmd + ["--shard-id", str(shard_id)]
    env = base_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_handle = log_path.open("w", buffering=1)
    print(f"[launch] shard {shard_id} -> GPU {gpu}, log {log_path}")
    process = subprocess.Popen(
        shard_cmd,
        cwd=str(TRIATTENTION_ROOT),
        stdout=log_handle,
        stderr=subprocess.STDOUT,
        env=env,
        start_new_session=True,
    )
    now = time.time()
    return ActiveShard(
        shard_id=shard_id,
        gpu=gpu,
        process=process,
        log_handle=log_handle,
        log_path=log_path,
        last_log_size=0,
        last_log_update_ts=now,
    )


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
    log_stamp: str,
    dry_run: bool,
    output_dir: Path,
    skip_existing: bool,
    num_samples: int,
    total_records: int,
    stall_timeout_minutes: float,
) -> None:
    if not gpus:
        raise ValueError("No GPUs available to schedule shards")
    shards_to_run: List[int]
    pending_runs: Dict[int, List[int]] = {}
    if skip_existing:
        shards_to_run = []
        for shard_id in range(total_shards):
            run_start, run_count = compute_local_runs(num_samples, total_shards, shard_id)
            if run_count == 0:
                print(f"[skip] shard {shard_id} has 0 assigned runs, no output required.")
                continue
            missing = []
            for run_id in range(run_start, run_start + run_count):
                if not run_completed(output_dir, shard_id, run_id, total_records):
                    missing.append(run_id)
            if not missing:
                print(f"[skip] shard {shard_id} has all assigned runs completed.")
                continue
            pending_runs[shard_id] = missing
            shards_to_run.append(shard_id)
        if not shards_to_run:
            print("All shard outputs already exist; skipping shard launch.")
            return
    else:
        shards_to_run = []
        for shard_id in range(total_shards):
            _, run_count = compute_local_runs(num_samples, total_shards, shard_id)
            if run_count == 0:
                print(f"[skip] shard {shard_id} has 0 assigned runs, no output required.")
                continue
            shards_to_run.append(shard_id)
    if dry_run:
        for shard_id in shards_to_run:
            run_start, run_count = compute_local_runs(num_samples, total_shards, shard_id)
            run_end = run_start + run_count - 1
            gpu = gpus[shard_id % len(gpus)]
            log_path = (log_dir / f"triattention_shard{shard_id:02d}_{log_stamp}.log").resolve()
            cmd_preview = base_cmd + ["--shard-id", str(shard_id)]
            runs_preview = pending_runs.get(shard_id)
            run_span = f"runs={run_start}-{run_end}"
            if runs_preview is not None:
                run_span = f"missing_runs={runs_preview}"
            print(
                f"[dry-run] shard {shard_id} -> GPU {gpu} ({run_span}, "
                f"records={total_records})\n"
                f"  log: {log_path}\n  cmd: {' '.join(cmd_preview)}"
            )
        return
    shard_queue: deque[int] = deque(shards_to_run)
    available: deque[str] = deque(gpus)
    active: Dict[str, ActiveShard] = {}
    stall_timeout_seconds = max(0.0, float(stall_timeout_minutes)) * 60.0
    try:
        while shard_queue or active:
            while shard_queue and available:
                gpu = available.popleft()
                shard_id = shard_queue.popleft()
                _, run_count = compute_local_runs(num_samples, total_shards, shard_id)
                if run_count == 0:
                    print(f"[skip] shard {shard_id} has 0 assigned runs, continuing.")
                    available.append(gpu)
                    continue
                active[gpu] = launch_shard(gpu, shard_id, base_cmd, base_env, log_dir, log_stamp)
            if not active:
                continue
            time.sleep(5)
            now = time.time()
            if stall_timeout_seconds > 0:
                for gpu, shard in active.items():
                    if shard.process.poll() is not None:
                        continue
                    try:
                        curr_size = int(shard.log_path.stat().st_size)
                    except OSError:
                        curr_size = shard.last_log_size
                    if curr_size > shard.last_log_size:
                        shard.last_log_size = curr_size
                        shard.last_log_update_ts = now
                        continue
                    if (now - shard.last_log_update_ts) >= stall_timeout_seconds:
                        terminate_active(active.values())
                        raise RuntimeError(
                            f"Shard {shard.shard_id} on GPU {gpu} stalled: "
                            f"no log growth for {stall_timeout_minutes:.1f} minutes "
                            f"(log={shard.log_path})"
                        )
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
    subprocess.check_call(cmd, cwd=str(TRIATTENTION_ROOT))


def _record_sample_idx(record: dict) -> int | None:
    value = record.get("sample_idx", record.get("idx"))
    return value if isinstance(value, int) else None


def _record_draw_idx(record: dict) -> int | None:
    value = record.get("draw_idx", 0)
    return value if isinstance(value, int) else None


def validate_merged_output_completeness(base_dir: Path, expected_num_samples: int | None) -> None:
    """Fail fast before evaluation when merged outputs are incomplete.

    HF eval script only warns on incomplete draw counts; dispatch should stop
    early to avoid publishing misleading metrics from half-finished shards.
    """
    if not expected_num_samples or expected_num_samples <= 0:
        return
    if not base_dir.exists():
        raise FileNotFoundError(f"merged base_dir not found: {base_dir}")
    jsonl_files = sorted(base_dir.glob("*.jsonl"))
    if not jsonl_files:
        raise FileNotFoundError(f"no merged jsonl found under {base_dir}")

    grouped_draws: dict[int, set[int]] = defaultdict(set)
    duplicate_pairs = 0
    missing_sample_idx = 0
    invalid_draw_idx = 0
    total_records = 0
    for path in jsonl_files:
        with path.open() as fp:
            for line in fp:
                line = line.strip()
                if not line:
                    continue
                total_records += 1
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    raise RuntimeError(f"invalid json line in merged file: {path}") from None
                if not isinstance(rec, dict):
                    continue
                sample_idx = _record_sample_idx(rec)
                if sample_idx is None:
                    missing_sample_idx += 1
                    continue
                draw_idx = _record_draw_idx(rec)
                if draw_idx is None or draw_idx < 0:
                    invalid_draw_idx += 1
                    continue
                draws = grouped_draws[sample_idx]
                before = len(draws)
                draws.add(draw_idx)
                if len(draws) == before:
                    duplicate_pairs += 1

    if missing_sample_idx or invalid_draw_idx:
        raise RuntimeError(
            "Merged outputs contain invalid indexing fields: "
            f"missing_sample_idx={missing_sample_idx} invalid_draw_idx={invalid_draw_idx} "
            f"base_dir={base_dir}"
        )
    if total_records <= 0 or not grouped_draws:
        raise RuntimeError(
            "Merged outputs contain no valid records before evaluation: "
            f"total_records={total_records} valid_questions={len(grouped_draws)} "
            f"base_dir={base_dir}"
        )

    bad_counts = [sid for sid, draws in grouped_draws.items() if len(draws) != expected_num_samples]
    if duplicate_pairs or bad_counts:
        preview = bad_counts[:10]
        raise RuntimeError(
            "Merged outputs incomplete or duplicate before evaluation: "
            f"questions={len(grouped_draws)} total_records={total_records} "
            f"expected_draws_per_question={expected_num_samples} "
            f"bad_question_count={len(bad_counts)} bad_question_preview={preview} "
            f"duplicate_pairs={duplicate_pairs} base_dir={base_dir}"
        )


def run_evaluation(base_dir: Path, dataset: str, exp_name: str, output_dir: Optional[Path], conda_env: str, dry_run: bool, num_samples: int | None = None) -> None:
    if not base_dir.exists():
        print(f"[eval] skip, base_dir not found: {base_dir}")
        return
    if not any(base_dir.glob("*.jsonl")):
        print(f"[eval] skip, no jsonl under {base_dir}")
        return
    if not dry_run:
        validate_merged_output_completeness(base_dir, num_samples)
    cmd = [
        "conda",
        "run",
        "-n",
        conda_env,
        "python",
        str(MULTI_EVAL_SCRIPT),
        "--base_dir",
        str(base_dir),
        "--dataset",
        dataset,
        "--exp_name",
        exp_name,
    ]
    if output_dir:
        cmd.extend(["--output_dir", str(output_dir)])
    if num_samples:
        cmd.extend(["--num_samples", str(num_samples)])
    if dry_run:
        print(f"[dry-run] eval command: {' '.join(cmd)}")
        return
    print(f"[eval] {' '.join(cmd)}")
    subprocess.check_call(cmd, cwd=str(TRIATTENTION_ROOT))


def main() -> None:
    mask_process_command("PD-L1_binder")
    args = parse_args()
    config = load_config(args.config)
    experiment = config.get("experiment", {})

    conda_env = experiment.get("conda_env", "rkv")
    eval_conda_env = experiment.get("eval_conda_env", conda_env)
    runner_path = resolve_path(experiment["runner_path"])
    allowed_current_runners = {
        "vllm_triattention_runner.py",
        "vllm_triattention_runtime_runner.py",
        "vllm_triattention_runtime_runner.py",
    }
    if (runner_path.name not in allowed_current_runners and not args.allow_legacy_v1):
        raise RuntimeError(
            "Refusing to run legacy V1 runner by default. "
            "Use the current TriAttention runner/config or pass --allow-legacy-v1 explicitly."
        )
    total_shards = args.num_shards or experiment.get("num_shards", 1)
    gpus = determine_gpus(args, experiment)
    log_dir = resolve_path(args.log_dir or experiment.get("log_dir", "TriAttention_vLLM/evaluation/logs"))
    log_stamp = time.strftime("%Y%m%d_%H%M%S")
    method_output_dir = resolve_path(args.method_output_dir or experiment.get("method_output_dir", "TriAttention_vLLM/evaluation/outputs"))
    merged_dir_name = experiment.get("merged_dir_name", "merged")
    eval_output_dir = resolve_path(args.eval_output_dir) if args.eval_output_dir else None

    runner_args = experiment.get("runner_args", {}).copy()
    if args.output_dir:
        runner_args["output_dir"] = args.output_dir
    if args.kv_budget is not None:
        runner_args["kv_budget"] = args.kv_budget
    if args.pruning_mode is not None:
        runner_args["pruning_mode"] = args.pruning_mode
    if args.divide_length is not None:
        runner_args["divide_length"] = args.divide_length
    if args.window_size is not None:
        runner_args["window_size"] = args.window_size
    if args.enforce_eager is not None:
        runner_args["enforce_eager"] = args.enforce_eager.lower() == "true"

    base_env = prepare_environment(experiment.get("env", {}))
    base_env.setdefault("VLLM_PROCESS_NAME_PREFIX", "PD-L1_binder")

    # TriAttention environment variables for legacy V1 backend only.
    # Current runner path must avoid V1 env coupling.
    if runner_path.name not in allowed_current_runners:
        if runner_args.get("sparse_stats_path"):
            base_env["TRIATTENTION_STATS_PATH"] = str(resolve_path(runner_args["sparse_stats_path"]))
        if runner_args.get("kv_budget"):
            base_env["TRIATTENTION_KV_BUDGET"] = str(runner_args["kv_budget"])
        if runner_args.get("divide_length"):
            base_env["TRIATTENTION_DIVIDE_LENGTH"] = str(runner_args["divide_length"])
        if runner_args.get("window_size"):
            base_env["TRIATTENTION_WINDOW_SIZE"] = str(runner_args["window_size"])
        if runner_args.get("pruning_mode"):
            base_env["TRIATTENTION_PRUNING_MODE"] = str(runner_args["pruning_mode"])
        # Suppress verbose logging in dispatch mode
        base_env.setdefault("TRIATTENTION_QUIET", "1")

    runner_args["output_dir"] = resolve_path(args.output_dir or runner_args.get("output_dir", method_output_dir / "shards"))
    runner_args["dataset_path"] = resolve_path(runner_args["dataset_path"])
    runner_args["model_path"] = resolve_path(runner_args["model_path"])
    if runner_args.get("sparse_stats_path"):
        runner_args["sparse_stats_path"] = resolve_path(runner_args["sparse_stats_path"])
    max_examples = runner_args.get("max_examples")
    try:
        max_examples = int(max_examples) if max_examples is not None else None
    except Exception:
        max_examples = None

    dataset_example_count = count_dataset_examples(runner_args["dataset_path"], max_examples)
    stall_timeout_minutes = (
        args.stall_timeout_minutes
        if args.stall_timeout_minutes is not None
        else float(experiment.get("stall_timeout_minutes", 0))
    )

    base_cmd = build_base_command(conda_env, runner_path, format_runner_args(runner_args, total_shards))
    num_samples = int(runner_args.get("num_samples", 64))

    run_shards(
        gpus,
        total_shards,
        base_cmd,
        base_env,
        log_dir,
        log_stamp,
        args.dry_run,
        runner_args["output_dir"],
        args.skip_existing,
        num_samples,
        total_records=dataset_example_count,
        stall_timeout_minutes=stall_timeout_minutes,
    )
    merge_outputs(runner_args["output_dir"], merged_dir_name, args.skip_merge, args.dry_run)
    merged_dir = runner_args["output_dir"].parent / merged_dir_name
    if not eval_output_dir:
        eval_output_dir = merged_dir.parent / "eval"
    if not args.no_eval:
        exp_name = experiment.get("name", merged_dir_name)
        run_evaluation(
            merged_dir,
            args.dataset,
            exp_name,
            eval_output_dir,
            eval_conda_env,
            args.dry_run,
            num_samples,
        )


if __name__ == "__main__":
    main()
