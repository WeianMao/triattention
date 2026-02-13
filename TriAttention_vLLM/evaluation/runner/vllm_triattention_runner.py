#!/usr/bin/env python3
"""Shard-aware vLLM runner for TriAttention evaluation.

This script replaces the HuggingFace R-KV runner with a vLLM-based implementation
using TriAttention KV compression. It maintains the same interface for compatibility
with the sharded dispatch framework.

Interface compatible with: R-KV/weian_development/rkv_sharded_eval.py
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch

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

# Dataset configuration
dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "aime25": ["question", "answer"],
    "math": ["problem", "answer"],
    "math500": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 8192,
    "aime24": 32768,
    "aime25": 32768,
    "math": 8192,
    "math500": 8192,
}

RUN_SEED_STRIDE = 1_000_000

# Prompt template (aligned with HF baseline)
DEFAULT_SYSTEM_PROMPT = ""
PROMPT_TEMPLATE = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def str2bool(value: str | bool) -> bool:
    """Parse boolean from string or bool."""
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to interpret boolean value '{value}'")


def resolve_path(path_like: str | Path) -> Path:
    """Resolve path relative to project root."""
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "R-KV":
        return (RKV_ROOT / Path(*parts[1:])).resolve() if len(parts) > 1 else RKV_ROOT.resolve()
    if parts and parts[0] == "TriAttention_vLLM":
        return (TRIATTENTION_ROOT / Path(*parts[1:])).resolve() if len(parts) > 1 else TRIATTENTION_ROOT.resolve()
    return (DC_ROOT / path).resolve()


def compute_local_records(total_records: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    """Compute start index and count of records for a shard."""
    base = total_records // num_shards
    extra = total_records % num_shards
    start = shard_id * base + min(shard_id, extra)
    count = base + (1 if shard_id < extra else 0)
    return start, count


def shard_run_dir(base_dir: Path, shard_id: int) -> Path:
    """Get the output directory for a shard."""
    return base_dir / f"shard{shard_id:02d}"


def run_artifacts(base_dir: Path, shard_id: int, run_id: int) -> dict[str, Path]:
    """Get all artifact paths for a run."""
    run_dir = shard_run_dir(base_dir, shard_id)
    run_dir.mkdir(parents=True, exist_ok=True)
    stem = run_dir / f"run{run_id:03d}"
    return {
        "run": stem.with_suffix(".jsonl"),
        "tmp": stem.with_suffix(".jsonl.tmp"),
        "meta": stem.with_suffix(".meta.json"),
        "meta_tmp": stem.with_suffix(".meta.json.tmp"),
    }


def run_is_complete(run_path: Path, meta_path: Path, expected_records: int) -> bool:
    """Check if a run has completed successfully."""
    if not run_path.exists() or run_path.stat().st_size == 0 or not meta_path.exists():
        return False
    try:
        with meta_path.open() as fp:
            meta = json.load(fp)
    except Exception:
        return False
    if meta.get("status") != "complete":
        return False
    recorded = meta.get("records")
    if expected_records > 0 and isinstance(recorded, int) and recorded < expected_records:
        return False
    if expected_records <= 0:
        return True
    try:
        with run_path.open() as fp:
            lines = sum(1 for _ in fp)
        return lines >= expected_records
    except Exception:
        return False


def _record_sample_idx(record: dict) -> int | None:
    """Extract sample index from a record."""
    value = record.get("sample_idx")
    if isinstance(value, int):
        return value
    value = record.get("index")
    if isinstance(value, int):
        return value
    return None


def load_existing_sample_indices(path: Path) -> set[int]:
    """Load existing sample indices from a partial run file."""
    indices: set[int] = set()
    if not path.exists():
        return indices
    try:
        with path.open() as fp:
            for line in fp:
                stripped = line.strip()
                if not stripped:
                    continue
                try:
                    record = json.loads(stripped)
                except json.JSONDecodeError:
                    break
                sample_idx = _record_sample_idx(record)
                if sample_idx is not None:
                    indices.add(sample_idx)
    except Exception:
        return set()
    return indices


def write_run_meta(meta_path: Path, run_id: int, shard_id: int, records: int) -> None:
    """Write completion metadata file."""
    meta_tmp = meta_path.with_suffix(".meta.json.tmp")
    meta = {
        "status": "complete",
        "records": records,
        "run_id": run_id,
        "shard_id": shard_id,
    }
    with meta_tmp.open("w") as fp:
        json.dump(meta, fp)
    meta_tmp.replace(meta_path)


def extract_question_from_record(record: dict, fallback_keys: List[str] = None) -> str:
    """Extract question from a dataset record."""
    if "question" in record:
        return record["question"]
    if "problem" in record:
        return record["problem"]
    if fallback_keys:
        for key in fallback_keys:
            if key in record:
                return record[key]
    raise KeyError(f"Cannot find question in record: {list(record.keys())}")


def build_prompt(question: str, use_chat_template: bool = False, system_prompt: str = "") -> str:
    """Build prompt from question."""
    # Use simple template (aligned with HF baseline)
    return PROMPT_TEMPLATE.format(question=question)


def load_dataset(
    path: Path,
    dataset_name: str,
    max_examples: int | None = None,
) -> tuple[List[str], List[dict]]:
    """Load dataset and build prompts."""
    prompts: List[str] = []
    test_data: List[dict] = []
    fallback_keys: List[str] = []
    if dataset_name in dataset2key and dataset2key[dataset_name]:
        fallback_keys.append(dataset2key[dataset_name][0])

    with path.open() as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question = extract_question_from_record(example, fallback_keys=fallback_keys)
            example["question"] = question
            prompt = build_prompt(question)
            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)
            if max_examples and len(test_data) >= max_examples:
                break
    return prompts, test_data


def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="vLLM TriAttention Sharded Runner")

    # Core parameters (compatible with rkv_sharded_eval.py)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset-path", dest="dataset_path", type=str, required=True)
    parser.add_argument("--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--model-path", dest="model_path", type=str, required=True)
    parser.add_argument("--max-length", dest="max_length", type=int, default=-1)
    parser.add_argument("--load-dtype", dest="load_dtype", type=str, default="bfloat16",
                        choices=["bfloat16", "float16"])

    # Sharding parameters
    parser.add_argument("--shard-id", dest="shard_id", type=int, required=True)
    parser.add_argument("--num-shards", dest="num_shards", type=int, required=True)

    # Sampling parameters
    parser.add_argument("--num-samples", dest="num_samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top-p", dest="top_p", type=float, default=0.95)
    parser.add_argument("--top-k", dest="top_k", type=int, default=-1)
    parser.add_argument("--max-examples", dest="max_examples", type=int, default=None)

    # TriAttention compression parameters
    parser.add_argument("--kv-budget", dest="kv_budget", type=int, default=2048)
    parser.add_argument("--window-size", dest="window_size", type=int, default=128)
    parser.add_argument("--divide-length", dest="divide_length", type=int, default=128)
    parser.add_argument("--sparse-round-window", dest="sparse_round_window", type=int, default=32)
    parser.add_argument("--sparse-stats-path", dest="sparse_stats_path", type=str, default=None)
    parser.add_argument("--sparse-offset-max-length", dest="sparse_offset_max_length", type=int, default=65536)
    parser.add_argument("--sparse-score-aggregation", dest="sparse_score_aggregation", type=str,
                        default="mean", choices=["mean", "max"])
    parser.add_argument("--pruning-mode", dest="pruning_mode", type=str, default="per_head",
                        choices=["per_head", "per_layer", "per_layer_per_head"])
    parser.add_argument("--sparse-normalize-scores", dest="sparse_normalize_scores", type=str2bool, default=True)
    parser.add_argument("--include-prefill-in-budget", dest="include_prefill_in_budget", type=str2bool, default=True)
    parser.add_argument("--protect-prefill", dest="protect_prefill", type=str2bool, default=True)
    parser.add_argument("--disable-mlr", dest="disable_mlr", type=str2bool, default=False)
    parser.add_argument("--disable-trig", dest="disable_trig", type=str2bool, default=False)
    parser.add_argument("--disable-top-n-high-freq", dest="disable_top_n_high_freq", type=int, default=0)

    # vLLM-specific parameters
    parser.add_argument("--tensor-parallel-size", dest="tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument("--disable-compression", dest="disable_compression", type=str2bool, default=False,
                        help="Run without compression (fullkv baseline)")

    return parser.parse_args()


def setup_triattention_config(args: argparse.Namespace):
    """Create TriAttention configuration from arguments."""
    from triattention import TriAttentionConfig

    stats_path = resolve_path(args.sparse_stats_path) if args.sparse_stats_path else None

    config = TriAttentionConfig(
        stats_path=stats_path,
        kv_budget=args.kv_budget,
        divide_length=args.divide_length,
        sparse_round_window=args.sparse_round_window,
        pruning_mode=args.pruning_mode,
        score_aggregation=args.sparse_score_aggregation,
        offset_max_length=args.sparse_offset_max_length,
        sparse_normalize_scores=args.sparse_normalize_scores,
        window_size=args.window_size,
        include_prefill_in_budget=args.include_prefill_in_budget,
        protect_prefill=args.protect_prefill,
        disable_mlr=args.disable_mlr,
        disable_trig=args.disable_trig,
        disable_top_n_high_freq=args.disable_top_n_high_freq,
    )
    return config


def setup_vllm_engine(args: argparse.Namespace, tri_config=None):
    """Initialize vLLM engine with optional TriAttention compression.

    Uses the official inheritance approach via setup_triattention() for V0 API,
    or environment variables for V1 API (when VLLM_ATTENTION_BACKEND is set).

    Configuration is passed via:
    1. setup_triattention(config) - registers config globally
    2. Environment variables - for V1 backend to pick up configuration
    """
    from vllm import LLM, SamplingParams

    # Determine max_model_len
    max_model_len = args.max_length if args.max_length > 0 else 32768

    tri_wrapper = None

    # Setup TriAttention configuration if compression is enabled
    if tri_config is not None and not args.disable_compression:
        from triattention.backends import setup_triattention
        from triattention.vllm_integration import TriAttentionWrapper

        # Register config globally for backend to access
        setup_triattention(tri_config)

        # Create wrapper for request state management
        tri_wrapper = TriAttentionWrapper(tri_config)

        # Set environment variables for V1 backend (if used)
        # These are read by v1_backend.py's _load_config_from_env()
        if tri_config.stats_path:
            os.environ["TRIATTENTION_STATS_PATH"] = str(tri_config.stats_path)
        os.environ["TRIATTENTION_KV_BUDGET"] = str(tri_config.kv_budget)
        os.environ["TRIATTENTION_DIVIDE_LENGTH"] = str(tri_config.divide_length)
        os.environ["TRIATTENTION_WINDOW_SIZE"] = str(tri_config.window_size)
        os.environ["TRIATTENTION_PRUNING_MODE"] = str(tri_config.pruning_mode)

        print(f"[TriAttention] Configuration registered: kv_budget={tri_config.kv_budget}, "
              f"divide_length={tri_config.divide_length}, pruning_mode={tri_config.pruning_mode}")
    else:
        print("[TriAttention] Compression disabled (fullkv mode)")

    # Initialize vLLM engine
    # Note: enforce_eager=True is required for TriAttention (CUDA graphs not compatible)
    llm_kwargs = dict(
        model=args.model_path,
        dtype=args.load_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
        max_model_len=max_model_len,
        enforce_eager=True,
    )
    # When compression is enabled, tell vLLM to use the CUSTOM backend
    # (registered by triattention plugin to point to TriAttentionBackend)
    if tri_config is not None and not args.disable_compression:
        llm_kwargs["attention_backend"] = "CUSTOM"
    llm = LLM(**llm_kwargs)

    return llm, tri_wrapper


def main(args: argparse.Namespace) -> None:
    """Main evaluation loop."""
    mask_process_command("PD-L1_binder")

    # Determine dataset name from path
    args.dataset_name = Path(args.dataset_path).name.split(".")[0]
    if (not args.max_length) or args.max_length <= 0:
        if args.dataset_name in dataset2max_length:
            args.max_length = dataset2max_length[args.dataset_name]

    num_samples = args.num_samples
    run_ids = list(range(num_samples))
    output_root = Path(args.output_dir)

    # Load dataset
    prompts, test_data = load_dataset(
        Path(args.dataset_path),
        args.dataset_name,
        max_examples=args.max_examples,
    )
    total_records = len(test_data)
    if total_records == 0:
        print("[ERROR] No records in dataset")
        return

    # Compute shard range
    start_record, local_records = compute_local_records(total_records, args.num_shards, args.shard_id)
    if local_records == 0:
        print(f"[INFO] Shard {args.shard_id} has no records assigned")
        return

    record_end = start_record + local_records - 1
    local_prompts = prompts[start_record : start_record + local_records]
    local_data = test_data[start_record : start_record + local_records]
    local_sample_indices = {item["index"] for item in local_data}

    print(f"[INFO] Shard {args.shard_id}/{args.num_shards}: records {start_record}-{record_end} ({local_records} total)")

    # Setup TriAttention config
    tri_config = None
    if not args.disable_compression and args.sparse_stats_path:
        tri_config = setup_triattention_config(args)
        print(f"[INFO] TriAttention config: kv_budget={tri_config.kv_budget}, "
              f"divide_length={tri_config.divide_length}, pruning_mode={tri_config.pruning_mode}")

    # Initialize vLLM engine
    print("[INFO] Initializing vLLM engine...")
    llm, tri_wrapper = setup_vllm_engine(args, tri_config)

    # Create sampling params
    from vllm import SamplingParams
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_length,
        n=1,
    )

    # Run inference for each sample
    for run_id in run_ids:
        artifacts = run_artifacts(output_root, args.shard_id, run_id)
        if run_is_complete(artifacts["run"], artifacts["meta"], local_records):
            print(f"[SKIP] run {run_id} already complete")
            continue

        # Load existing progress for resume
        existing_path = artifacts["tmp"] if artifacts["tmp"].exists() else artifacts["run"] if artifacts["run"].exists() else None
        existing_indices = load_existing_sample_indices(existing_path) if existing_path else set()
        completed = len(existing_indices & local_sample_indices)

        if completed >= local_records and local_records > 0:
            if existing_path == artifacts["tmp"]:
                existing_path.replace(artifacts["run"])
            if not artifacts["meta"].exists():
                write_run_meta(artifacts["meta"], run_id, args.shard_id, local_records)
            continue

        if artifacts["meta_tmp"].exists():
            artifacts["meta_tmp"].unlink()

        out_path = existing_path or artifacts["tmp"]
        if completed:
            sys.stderr.write(f"[resume] shard={args.shard_id} run={run_id} done={completed}/{local_records}\n")
            sys.stderr.flush()

        with out_path.open("a") as fout:
            progress = completed
            for local_idx, prompt in enumerate(local_prompts):
                sample_idx = local_data[local_idx]["index"]
                if sample_idx in existing_indices:
                    continue

                record_id = int(local_data[local_idx].get("id", sample_idx))
                seed_value = args.seed + run_id * RUN_SEED_STRIDE + sample_idx
                set_seed(seed_value)

                # Register request for state management
                request_id = f"req_{sample_idx}_{run_id}"
                if tri_wrapper is not None:
                    tri_wrapper.register_request(request_id)

                progress += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sys.stderr.write(
                    f"[progress {timestamp}] shard={args.shard_id} run={run_id + 1}/{num_samples} "
                    f"record={progress}/{local_records} "
                    f"range={start_record}-{record_end} sample_idx={sample_idx}\n"
                )
                sys.stderr.flush()

                # Generate using vLLM
                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                generated_text = outputs[0].outputs[0].text

                # Count tokens
                prefill_length = len(outputs[0].prompt_token_ids)
                output_tokens = len(outputs[0].outputs[0].token_ids)
                total_tokens = prefill_length + output_tokens

                # Build output record (compatible with rkv_sharded_eval.py format)
                record = dict(local_data[local_idx])
                record["prompt"] = prompt
                record["output"] = generated_text
                record["prefill_tokens"] = prefill_length
                record["output_tokens"] = output_tokens
                record["total_tokens"] = total_tokens
                record["sample_idx"] = sample_idx
                record["draw_idx"] = run_id
                record["backend"] = "vllm_triattention"
                record["kv_budget"] = args.kv_budget
                record["pruning_mode"] = args.pruning_mode

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

                # Unregister request to clean up state
                if tri_wrapper is not None:
                    tri_wrapper.unregister_request(request_id)

            torch.cuda.empty_cache()

        write_run_meta(artifacts["meta"], run_id, args.shard_id, local_records)
        if out_path == artifacts["tmp"]:
            artifacts["tmp"].replace(artifacts["run"])

    torch.cuda.empty_cache()
    print(f"[DONE] Shard {args.shard_id} completed all runs")


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    main(args)
