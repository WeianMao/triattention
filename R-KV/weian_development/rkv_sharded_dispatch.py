#!/usr/bin/env python3
"""Sharded launcher for R-KV HuggingFace AIME runs (non-intrusive copy of LazyEviction dispatcher)."""
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
if str(RKV_ROOT) not in sys.path:
    sys.path.insert(0, str(RKV_ROOT))

import yaml

from weian_development.process_utils import mask_process_command

DEFAULT_CONFIG = RKV_ROOT / "weian_script" / "configs" / "rkv_aime24_sharded.yaml"
MERGE_SCRIPT = RKV_ROOT / "weian_development" / "merge_rkv_shards.py"
MULTI_EVAL_SCRIPT = RKV_ROOT / "HuggingFace" / "evaluation" / "eval_math_multi.py"
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
        "--sparse-normalize-scores",
        dest="sparse_normalize_scores",
        action="store_true",
        help="Override runner arg: enable sparse score normalization for SpeckV.",
    )
    parser.add_argument(
        "--no-sparse-normalize-scores",
        dest="sparse_normalize_scores",
        action="store_false",
        help="Override runner arg: disable sparse score normalization for SpeckV.",
    )
    parser.set_defaults(sparse_normalize_scores=None)
    parser.add_argument(
        "--use-rank-aggregation",
        dest="use_rank_aggregation",
        action="store_true",
        help="Override runner arg: enable rank-based aggregation (min-pooling) instead of z-score + max-pooling.",
    )
    parser.add_argument(
        "--no-use-rank-aggregation",
        dest="use_rank_aggregation",
        action="store_false",
        help="Override runner arg: disable rank-based aggregation.",
    )
    parser.set_defaults(use_rank_aggregation=None)
    parser.add_argument(
        "--sparse-use-similarity",
        dest="sparse_use_similarity",
        action="store_true",
        help="Override runner arg: enable Similarity Deduplication in SpecKV.",
    )
    parser.add_argument(
        "--no-sparse-use-similarity",
        dest="sparse_use_similarity",
        action="store_false",
        help="Override runner arg: disable Similarity Deduplication in SpecKV.",
    )
    parser.set_defaults(sparse_use_similarity=None)
    parser.add_argument(
        "--sparse-similarity-mix-lambda",
        type=float,
        default=None,
        help="Override runner arg: frequency score weight for similarity scoring (e.g., 0.1, 0.3, 0.5, 0.7, 0.9).",
    )
    parser.add_argument(
        "--use-rank-similarity-combination",
        dest="use_rank_similarity_combination",
        action="store_true",
        help="Override runner arg: enable rank+similarity combination with normalized inverted rank.",
    )
    parser.add_argument(
        "--no-use-rank-similarity-combination",
        dest="use_rank_similarity_combination",
        action="store_false",
        help="Override runner arg: disable rank+similarity combination.",
    )
    parser.set_defaults(use_rank_similarity_combination=None)
    parser.add_argument(
        "--include-prefill-in-budget",
        dest="include_prefill_in_budget",
        action="store_true",
        help="Override runner arg: include prefill tokens in budget calculation (aligns with R-KV behavior).",
    )
    parser.add_argument(
        "--no-include-prefill-in-budget",
        dest="include_prefill_in_budget",
        action="store_false",
        help="Override runner arg: exclude prefill tokens from budget calculation.",
    )
    parser.set_defaults(include_prefill_in_budget=None)
    parser.add_argument(
        "--rkv-style-compression",
        dest="rkv_style_compression",
        action="store_true",
        help="Override runner arg: use R-KV style compression instead of SpeckV default.",
    )
    parser.add_argument(
        "--no-rkv-style-compression",
        dest="rkv_style_compression",
        action="store_false",
        help="Override runner arg: disable R-KV style compression.",
    )
    parser.set_defaults(rkv_style_compression=None)
    parser.add_argument(
        "--rkv-style-slack-trigger",
        dest="rkv_style_slack_trigger",
        action="store_true",
        help="Override runner arg: for R-KV style, trigger at budget + divide_length before pruning.",
    )
    parser.add_argument(
        "--no-rkv-style-slack-trigger",
        dest="rkv_style_slack_trigger",
        action="store_false",
        help="Override runner arg: for R-KV style, trigger at budget only (default).",
    )
    parser.set_defaults(rkv_style_slack_trigger=None)
    parser.add_argument(
        "--rkv-aligned-budget",
        dest="rkv_aligned_budget",
        action="store_true",
        help="Override runner arg: align budget calculation with R-KV (compress to exact budget).",
    )
    parser.add_argument(
        "--no-rkv-aligned-budget",
        dest="rkv_aligned_budget",
        action="store_false",
        help="Override runner arg: use original budget calculation (compress to budget - round_window).",
    )
    parser.set_defaults(rkv_aligned_budget=None)
    parser.add_argument(
        "--divide-length",
        dest="divide_length",
        type=int,
        default=None,
        help="Override runner arg: compress every N decode steps (like R-KV's divide_length).",
    )
    parser.add_argument(
        "--sparse-offset-max-length",
        dest="sparse_offset_max_length",
        type=int,
        default=None,
        help="Override runner arg: maximum offset length for sparse pruning frequency scoring.",
    )
    parser.add_argument(
        "--disable-top-n-high-freq",
        dest="disable_top_n_high_freq",
        type=int,
        default=None,
        help="Override runner arg: disable top-n high-frequency components in position-dependent scoring.",
    )
    parser.add_argument(
        "--simulate-bug-phase-offset",
        dest="simulate_bug_phase_offset",
        type=int,
        default=None,
        help="Override runner arg: simulate bug 896cbca6 phase offset by subtracting N×ω from phase (typical Δ≈156).",
    )
    parser.add_argument(
        "--simulate-attention-position-offset",
        dest="simulate_attention_position_offset",
        type=int,
        default=None,
        help="Override runner arg: simulate bug 896cbca6 attention position offset by adding offset to RoPE position_ids (typical Δ≈156).",
    )
    parser.add_argument(
        "--allow-prefill-compression",
        dest="allow_prefill_compression",
        action="store_true",
        help="Override runner arg: allow prefill tokens to be compressed (R-KV style).",
    )
    parser.add_argument(
        "--no-allow-prefill-compression",
        dest="allow_prefill_compression",
        action="store_false",
        help="Override runner arg: always preserve prefill tokens (default SpeckV behavior).",
    )
    parser.set_defaults(allow_prefill_compression=None)
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


def compute_local_samples(num_samples: int, num_shards: int, shard_id: int) -> tuple[int, int]:
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
            start_draw, local_count = compute_local_samples(num_samples, total_shards, shard_id)
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
            _, local_count = compute_local_samples(num_samples, total_shards, shard_id)
            if local_count == 0:
                print(f"[skip] shard {shard_id} has 0 assigned runs, no output required.")
                continue
            shards_to_run.append(shard_id)
    if dry_run:
        for shard_id in shards_to_run:
            start_draw, local_count = compute_local_samples(num_samples, total_shards, shard_id)
            gpu = gpus[shard_id % len(gpus)]
            log_path = (log_dir / f"rkv_aime24_shard{shard_id:02d}.log").resolve()
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
                _, local_count = compute_local_samples(num_samples, total_shards, shard_id)
                if local_count == 0:
                    print(f"[skip] shard {shard_id} has 0 assigned runs, continuing.")
                    available.append(gpu)
                    continue
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
    if args.sparse_normalize_scores is not None:
        runner_args["sparse_normalize_scores"] = args.sparse_normalize_scores
    if args.use_rank_aggregation is not None:
        runner_args["use_rank_aggregation"] = args.use_rank_aggregation
    if args.sparse_use_similarity is not None:
        runner_args["sparse_use_similarity"] = args.sparse_use_similarity
    if args.sparse_similarity_mix_lambda is not None:
        runner_args["sparse_similarity_mix_lambda"] = args.sparse_similarity_mix_lambda
    if args.use_rank_similarity_combination is not None:
        runner_args["use_rank_similarity_combination"] = args.use_rank_similarity_combination
    if args.include_prefill_in_budget is not None:
        runner_args["include_prefill_in_budget"] = args.include_prefill_in_budget
    if args.rkv_style_compression is not None:
        runner_args["rkv_style_compression"] = args.rkv_style_compression
    if args.rkv_style_slack_trigger is not None:
        runner_args["rkv_style_slack_trigger"] = args.rkv_style_slack_trigger
    if args.rkv_aligned_budget is not None:
        runner_args["rkv_aligned_budget"] = args.rkv_aligned_budget
    if args.divide_length is not None:
        runner_args["divide_length"] = args.divide_length
    if args.sparse_offset_max_length is not None:
        runner_args["sparse_offset_max_length"] = args.sparse_offset_max_length
    if args.disable_top_n_high_freq is not None:
        runner_args["disable_top_n_high_freq"] = args.disable_top_n_high_freq
    if args.simulate_bug_phase_offset is not None:
        runner_args["simulate_bug_phase_offset"] = args.simulate_bug_phase_offset
    if args.simulate_attention_position_offset is not None:
        runner_args["simulate_attention_position_offset"] = args.simulate_attention_position_offset
    if args.per_head_pruning is not None:
        runner_args["per_head_pruning"] = args.per_head_pruning
    if args.per_layer_perhead_pruning is not None:
        runner_args["per_layer_perhead_pruning"] = args.per_layer_perhead_pruning
    if args.allow_prefill_compression is not None:
        runner_args["allow_prefill_compression"] = args.allow_prefill_compression
    base_env = prepare_environment(experiment.get("env", {}))
    base_env.setdefault("VLLM_PROCESS_NAME_PREFIX", "PD-L1_binder")

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
