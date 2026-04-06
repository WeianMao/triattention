#!/usr/bin/env python3
"""Sharded launcher for multi-GPU inference dispatch."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, TextIO

RKV_ROOT = Path(__file__).resolve().parents[1]

import yaml


DEFAULT_CONFIG = RKV_ROOT / "configs" / "dispatch_default.yaml"
MERGE_SCRIPT = RKV_ROOT / "scripts" / "merge_shards.py"
MULTI_EVAL_SCRIPT = RKV_ROOT / "evaluation" / "eval_math_multi.py"
PATH_ARG_KEYS = {"output_dir", "dataset_path", "model_path", "tokenizer_path"}
RUNNER_EXCLUDE_KEYS = {"num_samples_by_dataset"}


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
        "--triattention-normalize-scores",
        dest="triattention_normalize_scores",
        action="store_true",
        help="Override runner arg: enable sparse score normalization for TriAttention.",
    )
    parser.add_argument(
        "--no-triattention-normalize-scores",
        dest="triattention_normalize_scores",
        action="store_false",
        help="Override runner arg: disable sparse score normalization for TriAttention.",
    )
    parser.set_defaults(triattention_normalize_scores=None)
    parser.add_argument(
        "--count-prompt-tokens",
        dest="count_prompt_tokens",
        action="store_true",
        help="Override runner arg: include prefill tokens in budget calculation (aligns with R-KV behavior).",
    )
    parser.add_argument(
        "--no-count-prompt-tokens",
        dest="count_prompt_tokens",
        action="store_false",
        help="Override runner arg: exclude prefill tokens from budget calculation.",
    )
    parser.set_defaults(count_prompt_tokens=None)
    parser.add_argument(
        "--attention-layer-compression",
        dest="attention_layer_compression",
        action="store_true",
        help="Override runner arg: use attention-layer compression.",
    )
    parser.add_argument(
        "--no-attention-layer-compression",
        dest="attention_layer_compression",
        action="store_false",
        help="Override runner arg: disable attention-layer compression.",
    )
    parser.set_defaults(attention_layer_compression=None)
    parser.add_argument(
        "--slack-budget-trigger",
        dest="slack_budget_trigger",
        action="store_true",
        help="Override runner arg: trigger at budget + divide_length before pruning.",
    )
    parser.add_argument(
        "--no-slack-budget-trigger",
        dest="slack_budget_trigger",
        action="store_false",
        help="Override runner arg: trigger at budget only (default).",
    )
    parser.set_defaults(slack_budget_trigger=None)
    parser.add_argument(
        "--divide-length",
        dest="divide_length",
        type=int,
        default=None,
        help="Override runner arg: compress every N decode steps.",
    )
    parser.add_argument(
        "--triattention-frequency-window",
        dest="triattention_frequency_window",
        type=int,
        default=None,
        help="Override runner arg: maximum offset length for sparse pruning frequency scoring.",
    )
    parser.add_argument(
        "--attn-implementation",
        dest="attn_implementation",
        type=str,
        default=None,
        choices=["eager", "flash_attention_2", "sdpa"],
        help="Override runner arg: attention implementation (eager for V100, flash_attention_2 for A100/H100).",
    )
    parser.add_argument(
        "--allow-prefill-compression",
        dest="allow_prefill_compression",
        action="store_true",
        help="Override runner arg: allow prefill tokens to be compressed.",
    )
    parser.add_argument(
        "--no-allow-prefill-compression",
        dest="allow_prefill_compression",
        action="store_false",
        help="Override runner arg: always preserve prefill tokens (default TriAttention behavior).",
    )
    parser.set_defaults(allow_prefill_compression=None)
    parser.add_argument(
        "--protect-prefill",
        dest="protect_prefill",
        action="store_true",
        help="Override runner arg: protect prefill tokens from compression (ablation for R-KV method).",
    )
    parser.add_argument(
        "--no-protect-prefill",
        dest="protect_prefill",
        action="store_false",
        help="Override runner arg: allow prefill tokens to compete for budget (default R-KV behavior).",
    )
    parser.set_defaults(protect_prefill=None)
    per_head_group = parser.add_mutually_exclusive_group()
    per_head_group.add_argument(
        "--per-head-pruning",
        dest="per_head_pruning",
        action="store_true",
        help="Enable per-KV-head independent pruning.",
    )
    per_head_group.add_argument(
        "--no-per-head-pruning",
        dest="per_head_pruning",
        action="store_false",
        help="Disable per-KV-head independent pruning.",
    )
    parser.set_defaults(per_head_pruning=None)
    layer_perhead_group = parser.add_mutually_exclusive_group()
    layer_perhead_group.add_argument(
        "--per-layer-perhead-pruning",
        dest="per_layer_perhead_pruning",
        action="store_true",
        help="Enable per-layer-per-head independent pruning (each (layer, KV head) selects independently).",
    )
    layer_perhead_group.add_argument(
        "--no-per-layer-perhead-pruning",
        dest="per_layer_perhead_pruning",
        action="store_false",
        help="Disable per-layer-per-head independent pruning.",
    )
    parser.set_defaults(per_layer_perhead_pruning=None)
    parser.add_argument(
        "--layer-perhead-aggregation",
        type=str,
        choices=["max", "mean"],
        default="max",
        help="Aggregation method for per-layer-perhead pruning: max (default) or mean.",
    )
    parser.add_argument(
        "--disable-mlr",
        dest="disable_mlr",
        action="store_true",
        help="Override runner arg: disable MLR term in TriAttention extra computation.",
    )
    parser.add_argument(
        "--no-disable-mlr",
        dest="disable_mlr",
        action="store_false",
        help="Override runner arg: enable MLR term in TriAttention extra computation (default).",
    )
    parser.set_defaults(disable_mlr=None)
    parser.add_argument(
        "--disable-trig",
        dest="disable_trig",
        action="store_true",
        help="Override runner arg: disable position-dependent term in TriAttention scoring.",
    )
    parser.add_argument(
        "--no-disable-trig",
        dest="disable_trig",
        action="store_false",
        help="Override runner arg: enable position-dependent term in TriAttention scoring (default).",
    )
    parser.set_defaults(disable_trig=None)
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


def compute_local_runs(num_samples: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    base = num_samples // num_shards
    extra = num_samples % num_shards
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


def questions_for_shard(total_questions: int, num_shards: int, shard_id: int) -> int:
    base = total_questions // num_shards
    extra = total_questions % num_shards
    return base + (1 if shard_id < extra else 0)


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


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "R-KV":
        path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return (RKV_ROOT / path).resolve()


def format_runner_args(args_dict: Dict[str, object], total_shards: int) -> List[str]:
    formatted: List[str] = []
    for key, value in args_dict.items():
        if value is None:
            continue
        if key in RUNNER_EXCLUDE_KEYS:
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
    log_stamp: str,
) -> ActiveShard:
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"shard{shard_id:02d}_{log_stamp}.log"
    shard_cmd = base_cmd + ["--shard_id", str(shard_id)]
    env = base_env.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    log_handle = log_path.open("w", buffering=1)
    print(f"[launch] shard {shard_id} -> GPU {gpu}, log {log_path}")
    process = subprocess.Popen(
        shard_cmd,
        cwd=str(RKV_ROOT),
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
    log_stamp: str,
    dry_run: bool,
    output_dir: Path,
    skip_existing: bool,
    num_samples: int,
    total_questions: int | None = None,
) -> None:
    if not gpus:
        raise ValueError("No GPUs available to schedule shards")
    shards_to_run: List[int]
    pending_runs: Dict[int, List[int]] = {}
    expected_records = total_questions if total_questions is not None else 0
    if skip_existing:
        shards_to_run = []
        for shard_id in range(total_shards):
            start_draw, local_count = compute_local_runs(num_samples, total_shards, shard_id)
            if local_count == 0:
                print(f"[skip] shard {shard_id} has 0 assigned runs, no output required.")
                continue
            missing = []
            for run_id in range(start_draw, start_draw + local_count):
                if not run_completed(output_dir, shard_id, run_id, expected_records):
                    missing.append(run_id)
            if not missing:
                print(f"[skip] shard {shard_id} has all {local_count} runs completed.")
                continue
            pending_runs[shard_id] = missing
            shards_to_run.append(shard_id)
        if not shards_to_run:
            print("All shard outputs already exist; skipping shard launch.")
            return
    else:
        shards_to_run = []
        for shard_id in range(total_shards):
            _, local_count = compute_local_runs(num_samples, total_shards, shard_id)
            if local_count == 0:
                print(f"[skip] shard {shard_id} has 0 assigned runs, no output required.")
                continue
            shards_to_run.append(shard_id)
    if dry_run:
        for shard_id in shards_to_run:
            start_draw, local_count = compute_local_runs(num_samples, total_shards, shard_id)
            gpu = gpus[shard_id % len(gpus)]
            log_path = (log_dir / f"shard{shard_id:02d}_{log_stamp}.log").resolve()
            cmd_preview = base_cmd + ["--shard_id", str(shard_id)]
            runs_preview = pending_runs.get(shard_id)
            run_span = f"runs={local_count}"
            if runs_preview is not None:
                run_span = f"missing_runs={runs_preview}"
            print(f"[dry-run] shard {shard_id} -> GPU {gpu} ({run_span})\n  log: {log_path}\n  cmd: {' '.join(cmd_preview)}")
        return
    shard_queue: deque[int] = deque(shards_to_run)
    available: deque[str] = deque(gpus)
    active: Dict[str, ActiveShard] = {}
    try:
        while shard_queue or active:
            while shard_queue and available:
                gpu = available.popleft()
                shard_id = shard_queue.popleft()
                _, local_count = compute_local_runs(num_samples, total_shards, shard_id)
                if local_count == 0:
                    print(f"[skip] shard {shard_id} has 0 assigned runs, continuing.")
                    available.append(gpu)
                    continue
                active[gpu] = launch_shard(gpu, shard_id, base_cmd, base_env, log_dir, log_stamp)
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
    subprocess.check_call(cmd, cwd=str(RKV_ROOT))


def run_evaluation(base_dir: Path, dataset: str, exp_name: str, output_dir: Optional[Path], conda_env: str, dry_run: bool, num_samples: int | None = None) -> None:
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
    subprocess.check_call(cmd, cwd=str(RKV_ROOT))


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    experiment = config.get("experiment", {})

    conda_env = experiment.get("conda_env", "triattention")
    runner_path = resolve_path(experiment["runner_path"])
    total_shards = args.num_shards or experiment.get("num_shards", 1)
    gpus = determine_gpus(args, experiment)
    log_dir = resolve_path(args.log_dir or experiment.get("log_dir", "logs/dispatch"))
    log_stamp = time.strftime("%Y%m%d_%H%M%S")
    method_output_dir = resolve_path(args.method_output_dir or experiment.get("method_output_dir", "outputs/dispatch"))
    merged_dir_name = experiment.get("merged_dir_name", "merged")
    eval_output_dir = resolve_path(args.eval_output_dir) if args.eval_output_dir else None

    runner_args = experiment.get("runner_args", {}).copy()
    if args.output_dir:
        runner_args["output_dir"] = args.output_dir
    if args.triattention_normalize_scores is not None:
        runner_args["triattention_normalize_scores"] = args.triattention_normalize_scores
    if args.count_prompt_tokens is not None:
        runner_args["count_prompt_tokens"] = args.count_prompt_tokens
    if args.attention_layer_compression is not None:
        runner_args["attention_layer_compression"] = args.attention_layer_compression
    if args.slack_budget_trigger is not None:
        runner_args["slack_budget_trigger"] = args.slack_budget_trigger
    if args.divide_length is not None:
        runner_args["divide_length"] = args.divide_length
    if args.triattention_frequency_window is not None:
        runner_args["triattention_frequency_window"] = args.triattention_frequency_window
    if args.disable_mlr is not None:
        runner_args["disable_mlr"] = args.disable_mlr
    if args.disable_trig is not None:
        runner_args["disable_trig"] = args.disable_trig
    if args.attn_implementation is not None:
        runner_args["attn_implementation"] = args.attn_implementation
    if args.per_head_pruning is not None:
        runner_args["per_head_pruning"] = args.per_head_pruning
    if args.per_layer_perhead_pruning is not None:
        runner_args["per_layer_perhead_pruning"] = args.per_layer_perhead_pruning
    if args.layer_perhead_aggregation:
        runner_args["layer_perhead_aggregation"] = args.layer_perhead_aggregation
    if args.allow_prefill_compression is not None:
        runner_args["allow_prefill_compression"] = args.allow_prefill_compression
    if args.protect_prefill is not None:
        runner_args["protect_prefill"] = args.protect_prefill
    base_env = prepare_environment(experiment.get("env", {}))

    runner_args["output_dir"] = resolve_path(args.output_dir or runner_args.get("output_dir", method_output_dir / "shards"))
    runner_args["dataset_path"] = resolve_path(runner_args["dataset_path"])
    runner_args["model_path"] = resolve_path(runner_args["model_path"])
    max_examples = runner_args.get("max_examples")
    try:
        max_examples = int(max_examples) if max_examples is not None else None
    except Exception:
        max_examples = None

    dataset_example_count = count_dataset_examples(runner_args["dataset_path"], max_examples)

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
        total_questions=dataset_example_count,
    )
    merge_outputs(runner_args["output_dir"], merged_dir_name, args.skip_merge, args.dry_run)
    merged_dir = runner_args["output_dir"].parent / merged_dir_name
    if not eval_output_dir:
        eval_output_dir = merged_dir.parent / "eval"
    if not args.no_eval:
        exp_name = experiment.get("name", merged_dir_name)
        run_evaluation(merged_dir, args.dataset, exp_name, eval_output_dir, conda_env, args.dry_run, num_samples)


if __name__ == "__main__":
    main()
