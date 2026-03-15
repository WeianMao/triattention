#!/usr/bin/env python3
"""Shard-aware vLLM runner for TriAttention evaluation.

This runner is the current/default TriAttention integration runner:
- uses vLLM configurable extension points (worker_cls/scheduler_cls);
- does not rely on V1 custom attention backend path.
"""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import torch
from transformers import AutoTokenizer

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
from weian_development.speckv.prompt_utils import (
    DEFAULT_SYSTEM_PROMPT,
    build_prompt as build_hf_prompt,
    extract_question_from_record as extract_hf_question_from_record,
)

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
RUNTIME_ENV_PREFIX = "TRIATTN_RUNTIME_"


def set_seed(seed: int) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)
    try:
        import numpy as np  # type: ignore
        np.random.seed(seed)
    except Exception:
        # NumPy is optional in some runtime envs; torch+python RNG seeding is still useful.
        pass
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


def compute_local_runs(total_runs: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    """Compute start run_id and run count for this shard.

    This matches HF sharded strategy: all shards see full question set, while
    draw/run ids are partitioned across shards.
    """
    base = total_runs // num_shards
    extra = total_runs % num_shards
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


def load_dataset(
    path: Path,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    *,
    use_chat_template: bool,
    system_prompt: str,
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
            question = extract_hf_question_from_record(
                example,
                fallback_keys=fallback_keys,
            )
            example["question"] = question
            prompt = build_hf_prompt(
                tokenizer,
                question,
                use_chat_template=use_chat_template,
                system_prompt=system_prompt,
            )
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

    # Core parameters
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
    parser.add_argument(
        "--use-chat-template",
        dest="use_chat_template",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--chat-system-prompt",
        dest="chat_system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
    )

    # TriAttention compression parameters
    parser.add_argument("--kv-budget", dest="kv_budget", type=int, default=2048)
    parser.add_argument("--divide-length", dest="divide_length", type=int, default=128)
    parser.add_argument("--protect-prefill", dest="protect_prefill", type=str2bool, default=True)
    parser.add_argument("--enable-kv-usage-trigger", dest="enable_kv_usage_trigger", type=str2bool, default=False)
    parser.add_argument("--kv-usage-trigger", dest="kv_usage_trigger", type=float, default=0.98)
    parser.add_argument("--kv-usage-release", dest="kv_usage_release", type=float, default=0.90)
    parser.add_argument("--window-size", dest="window_size", type=int, default=128)
    parser.add_argument("--sparse-stats-path", dest="sparse_stats_path", type=str, default=None)
    parser.add_argument(
        "--sparse-score-aggregation",
        dest="sparse_score_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
    )
    parser.add_argument(
        "--pruning-mode",
        dest="pruning_mode",
        type=str,
        default="per_head",
        choices=["per_head", "per_layer", "per_layer_per_head"],
    )
    parser.add_argument(
        "--sparse-normalize-scores",
        dest="sparse_normalize_scores",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--include-prefill-in-budget",
        dest="include_prefill_in_budget",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--per-head-selection-semantics",
        dest="per_head_selection_semantics",
        type=str,
        default="hf_aligned_global_per_head",
        choices=["legacy_layer_local", "hf_aligned_global_per_head"],
    )
    parser.add_argument(
        "--layer-perhead-aggregation",
        dest="layer_perhead_aggregation",
        type=str,
        default="max",
        choices=["max", "mean"],
    )
    parser.add_argument(
        "--per-layer-aggregation",
        dest="per_layer_aggregation",
        type=str,
        default="max",
        choices=["max", "mean", "pure_mean"],
    )
    parser.add_argument(
        "--allow-per-layer-mode",
        dest="allow_per_layer_mode",
        type=str2bool,
        default=False,
        help="Explicitly allow pruning_mode=per_layer (disabled by default to avoid accidental use).",
    )
    parser.add_argument("--disable-mlr", dest="disable_mlr", type=str2bool, default=False)
    parser.add_argument("--disable-trig", dest="disable_trig", type=str2bool, default=False)
    parser.add_argument(
        "--disable-top-n-high-freq",
        dest="disable_top_n_high_freq",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--enable-experimental-kv-compaction",
        dest="enable_experimental_kv_compaction",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--enable-experimental-block-reclaim",
        dest="enable_experimental_block_reclaim",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--require-triton-scoring",
        dest="require_triton_scoring",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--require-physical-reclaim",
        dest="require_physical_reclaim",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--fail-on-effective-len-regression",
        dest="fail_on_effective_len_regression",
        type=str2bool,
        default=True,
    )
    parser.add_argument(
        "--effective-len-regression-ratio",
        dest="effective_len_regression_ratio",
        type=float,
        default=0.9,
    )
    parser.add_argument(
        "--effective-len-guard-divide-multiples",
        dest="effective_len_guard_divide_multiples",
        type=int,
        default=2,
    )
    parser.add_argument(
        "--score-chunk-max-tokens",
        dest="score_chunk_max_tokens",
        type=int,
        default=4096,
    )
    parser.add_argument("--log-decisions", dest="log_decisions", type=str2bool, default=True)
    parser.add_argument(
        "--enforce-eager",
        dest="enforce_eager",
        type=str2bool,
        default=False,
    )

    # vLLM-specific parameters
    parser.add_argument("--tensor-parallel-size", dest="tensor_parallel_size", type=int, default=1)
    parser.add_argument("--gpu-memory-utilization", dest="gpu_memory_utilization", type=float, default=0.9)
    parser.add_argument(
        "--prefill-auto-chunk",
        dest="prefill_auto_chunk",
        type=str2bool,
        default=False,
        help=(
            "Enable vLLM chunked prefill for long prompts. "
            "When enabled, requests longer than the effective chunk trigger are split."
        ),
    )
    parser.add_argument(
        "--prefill-chunk-threshold",
        dest="prefill_chunk_threshold",
        type=int,
        default=2048,
        help=(
            "Chunk trigger threshold for prefill optimization. "
            "Current vLLM integration uses max_num_batched_tokens as both trigger and chunk size."
        ),
    )
    parser.add_argument(
        "--prefill-chunk-size",
        dest="prefill_chunk_size",
        type=int,
        default=2048,
        help=(
            "Chunk size for prefill optimization. "
            "Must match prefill_chunk_threshold in current runtime integration."
        ),
    )
    parser.add_argument("--disable-compression", dest="disable_compression", type=str2bool, default=False,
                        help="Run without TriAttention scheduler/worker injection (fullkv baseline).")
    parser.add_argument(
        "--force-runtime-integration",
        dest="force_runtime_integration",
        type=str2bool,
        default=False,
        help="Use TriAttention worker/scheduler even when compression is disabled (diagnostic).",
    )
    parser.add_argument(
        "--force-runtime-worker",
        dest="force_runtime_worker",
        type=str2bool,
        default=False,
        help="Use only TriAttention worker injection as a diagnostic (works with disable_compression=true).",
    )
    parser.add_argument(
        "--force-runtime-scheduler",
        dest="force_runtime_scheduler",
        type=str2bool,
        default=False,
        help="Use only TriAttention scheduler injection as a diagnostic (works with disable_compression=true).",
    )

    return parser.parse_args()


def _apply_runtime_env(args: argparse.Namespace) -> None:
    env_values = {
        "KV_BUDGET": str(args.kv_budget),
        "DIVIDE_LENGTH": str(args.divide_length),
        "PROTECT_PREFILL": str(args.protect_prefill).lower(),
        "DISABLE_COMPRESSION": str(args.disable_compression).lower(),
        "ENABLE_KV_USAGE_TRIGGER": str(args.enable_kv_usage_trigger).lower(),
        "KV_USAGE_TRIGGER": str(args.kv_usage_trigger),
        "KV_USAGE_RELEASE": str(args.kv_usage_release),
        "ENABLE_EXPERIMENTAL_KV_COMPACTION": str(
            args.enable_experimental_kv_compaction
        ).lower(),
        "ENABLE_EXPERIMENTAL_BLOCK_RECLAIM": str(
            args.enable_experimental_block_reclaim
        ).lower(),
        "REQUIRE_TRITON_SCORING": str(args.require_triton_scoring).lower(),
        "REQUIRE_PHYSICAL_RECLAIM": str(args.require_physical_reclaim).lower(),
        "FAIL_ON_EFFECTIVE_LEN_REGRESSION": str(
            args.fail_on_effective_len_regression
        ).lower(),
        "EFFECTIVE_LEN_REGRESSION_RATIO": str(args.effective_len_regression_ratio),
        "EFFECTIVE_LEN_GUARD_DIVIDE_MULTIPLES": str(
            args.effective_len_guard_divide_multiples
        ),
        "SCORE_CHUNK_MAX_TOKENS": str(args.score_chunk_max_tokens),
        "LOG_DECISIONS": str(args.log_decisions).lower(),
        "WINDOW_SIZE": str(args.window_size),
        "PRUNING_MODE": str(args.pruning_mode),
        "SPARSE_SCORE_AGGREGATION": str(args.sparse_score_aggregation),
        "SPARSE_NORMALIZE_SCORES": str(args.sparse_normalize_scores).lower(),
        "INCLUDE_PREFILL_IN_BUDGET": str(args.include_prefill_in_budget).lower(),
        "PER_HEAD_SELECTION_SEMANTICS": str(args.per_head_selection_semantics),
        "LAYER_PERHEAD_AGGREGATION": str(args.layer_perhead_aggregation),
        "PER_LAYER_AGGREGATION": str(args.per_layer_aggregation),
        "ALLOW_PER_LAYER_MODE": str(args.allow_per_layer_mode).lower(),
        "DISABLE_MLR": str(args.disable_mlr).lower(),
        "DISABLE_TRIG": str(args.disable_trig).lower(),
        "DISABLE_TOP_N_HIGH_FREQ": str(args.disable_top_n_high_freq),
    }
    for key, value in env_values.items():
        os.environ[RUNTIME_ENV_PREFIX + key] = value
    if args.sparse_stats_path:
        sparse_stats_path = str(resolve_path(args.sparse_stats_path))
        os.environ[RUNTIME_ENV_PREFIX + "SPARSE_STATS_PATH"] = sparse_stats_path
    else:
        os.environ.pop(RUNTIME_ENV_PREFIX + "SPARSE_STATS_PATH", None)


def setup_vllm_engine(args: argparse.Namespace):
    """Initialize vLLM engine with optional TriAttention injection."""
    from vllm import LLM

    max_model_len = args.max_length if args.max_length > 0 else 32768
    load_dtype = args.load_dtype
    if load_dtype == "bfloat16" and torch.cuda.is_available():
        major, _minor = torch.cuda.get_device_capability()
        if major < 8:
            print(
                "[TriAttention] GPU capability < 8.0, "
                "fallback dtype bfloat16 -> float16"
            )
            load_dtype = "float16"
    llm_kwargs = dict(
        model=args.model_path,
        dtype=load_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
        max_model_len=max_model_len,
        enforce_eager=bool(args.enforce_eager),
    )
    if os.environ.get("TRIATTN_DEBUG_FORCE_SYNC_SCHED", "0") == "1":
        llm_kwargs["async_scheduling"] = False
    if bool(getattr(args, "prefill_auto_chunk", False)):
        chunk_threshold = int(getattr(args, "prefill_chunk_threshold", 0))
        chunk_size = int(getattr(args, "prefill_chunk_size", 0))
        if chunk_threshold <= 0 or chunk_size <= 0:
            raise ValueError(
                "prefill_chunk_threshold and prefill_chunk_size must both be > 0 "
                "when prefill_auto_chunk is enabled"
            )
        if chunk_threshold != chunk_size:
            raise ValueError(
                "Current vLLM integration maps prefill trigger and chunk size to a "
                "single engine knob (max_num_batched_tokens). Set "
                "prefill_chunk_threshold == prefill_chunk_size."
            )
        llm_kwargs["enable_chunked_prefill"] = True
        llm_kwargs["max_num_batched_tokens"] = int(chunk_size)

    force_runtime_integration = bool(
        getattr(args, "force_runtime_integration", False)
    )
    force_runtime_worker = bool(
        getattr(args, "force_runtime_worker", False)
    )
    force_runtime_scheduler = bool(
        getattr(args, "force_runtime_scheduler", False)
    )
    force_any_runtime = bool(force_runtime_integration or force_runtime_worker or force_runtime_scheduler)
    use_runtime_integration = (not args.disable_compression) or force_any_runtime
    if use_runtime_integration:
        if not args.enable_experimental_kv_compaction and not args.disable_compression:
            raise RuntimeError(
                "TriAttention strict mode requires "
                "enable_experimental_kv_compaction=true"
            )
        if not args.require_triton_scoring and not args.disable_compression:
            raise RuntimeError(
                "TriAttention strict mode requires require_triton_scoring=true"
            )
        if (
            args.require_physical_reclaim
            and not args.enable_experimental_block_reclaim
            and not args.disable_compression
        ):
            raise RuntimeError(
                "TriAttention strict mode requires "
                "enable_experimental_block_reclaim=true when "
                "require_physical_reclaim=true"
            )
        _apply_runtime_env(args)
        want_runtime_worker = (not args.disable_compression) or force_runtime_integration or force_runtime_worker
        want_runtime_scheduler = (not args.disable_compression) or force_runtime_integration or force_runtime_scheduler
        # Performance-first integration path: keep native vLLM class identities
        # and patch only the minimal methods in-place.
        from triattention_runtime.integration_monkeypatch import install_vllm_integration_monkeypatches

        install_vllm_integration_monkeypatches(
            patch_worker=bool(want_runtime_worker),
            patch_scheduler=bool(want_runtime_scheduler),
        )
        print(
            "[TriAttention] enabled: "
            f"budget={args.kv_budget}, divide={args.divide_length}, "
            f"protect_prefill={args.protect_prefill}, "
            f"experimental_compaction={args.enable_experimental_kv_compaction}, "
            f"experimental_block_reclaim={args.enable_experimental_block_reclaim}, "
            f"require_triton_scoring={args.require_triton_scoring}, "
            f"require_physical_reclaim={args.require_physical_reclaim}, "
            f"fail_on_effective_len_regression={args.fail_on_effective_len_regression}, "
            f"enforce_eager={args.enforce_eager}, "
            f"prefill_auto_chunk={args.prefill_auto_chunk}, "
            f"prefill_chunk_threshold={args.prefill_chunk_threshold}, "
            f"prefill_chunk_size={args.prefill_chunk_size}, "
            f"disable_compression={args.disable_compression}, "
            f"force_runtime_integration={force_runtime_integration}, "
            f"force_runtime_worker={force_runtime_worker}, "
            f"force_runtime_scheduler={force_runtime_scheduler}, "
            f"inject_worker={want_runtime_worker}, "
            f"inject_scheduler={want_runtime_scheduler}, "
            "integration_mode=monkeypatch"
        )
    else:
        print("[TriAttention] disabled (fullkv mode)")

    llm = LLM(**llm_kwargs)
    return llm


def main(args: argparse.Namespace) -> None:
    """Main evaluation loop."""
    mask_process_command("PD-L1_binder")

    args.dataset_name = Path(args.dataset_path).name.split(".")[0]
    if (not args.max_length) or args.max_length <= 0:
        if args.dataset_name in dataset2max_length:
            args.max_length = dataset2max_length[args.dataset_name]

    output_root = Path(args.output_dir)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        use_fast=True,
        padding_side="left",
        trust_remote_code=True,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    prompts, test_data = load_dataset(
        Path(args.dataset_path),
        args.dataset_name,
        tokenizer,
        use_chat_template=args.use_chat_template,
        system_prompt=args.chat_system_prompt,
        max_examples=args.max_examples,
    )
    total_records = len(test_data)
    if total_records == 0:
        print("[ERROR] No records in dataset")
        return

    num_samples = int(args.num_samples)
    start_draw, local_draws = compute_local_runs(
        total_runs=num_samples,
        num_shards=args.num_shards,
        shard_id=args.shard_id,
    )
    if local_draws == 0:
        print(f"[INFO] Shard {args.shard_id} has no runs assigned")
        return
    run_ids = list(range(start_draw, start_draw + local_draws))

    local_records = total_records
    start_record = 0
    record_end = total_records - 1
    local_prompts = prompts
    local_data = test_data
    local_sample_indices = {item["index"] for item in local_data}

    draw_end = start_draw + local_draws - 1
    print(
        f"[INFO] Shard {args.shard_id}/{args.num_shards}: "
        f"draws {start_draw}-{draw_end} ({local_draws} total), "
        f"records {start_record}-{record_end} ({local_records} total)"
    )

    print("[INFO] Initializing vLLM engine (TriAttention)...")
    llm = setup_vllm_engine(args)
    llm_tokenizer = llm.get_tokenizer()
    eos_token_id = getattr(llm_tokenizer, "eos_token_id", None)

    from vllm import SamplingParams

    for run_id in run_ids:
        artifacts = run_artifacts(output_root, args.shard_id, run_id)
        if run_is_complete(artifacts["run"], artifacts["meta"], local_records):
            print(f"[SKIP] run {run_id} already complete")
            continue

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

                seed_value = args.seed + run_id * RUN_SEED_STRIDE + sample_idx
                set_seed(seed_value)
                prefill_length = len(
                    tokenizer(
                        prompt,
                        add_special_tokens=True,
                    )["input_ids"]
                )
                max_new_tokens = max(1, args.max_length - prefill_length)
                # HF Transformers path implicitly uses generation_config defaults.
                # In particular, top_k defaults to 50 when do_sample=True, unless explicitly set.
                # Our HF-strict configs should pass top_k=50; if it is left negative, warn loudly.
                if args.top_k < 0:
                    sys.stderr.write(
                        f"[WARN] top_k={args.top_k} disables top-k; HF default is 50. "
                        "If this is HF-strict alignment, set --top-k 50.\n"
                    )
                    sys.stderr.flush()
                sampling_params = SamplingParams(
                    temperature=args.temperature,
                    top_p=args.top_p,
                    top_k=args.top_k if args.top_k > 0 else -1,
                    max_tokens=max_new_tokens,
                    n=1,
                    seed=seed_value,
                    stop_token_ids=[eos_token_id] if eos_token_id is not None else None,
                )

                progress += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                sys.stderr.write(
                    f"[progress {timestamp}] shard={args.shard_id} run={run_id + 1}/{num_samples} "
                    f"record={progress}/{local_records} "
                    f"range={start_record}-{record_end} sample_idx={sample_idx}\n"
                )
                sys.stderr.flush()

                outputs = llm.generate([prompt], sampling_params, use_tqdm=False)
                generated_text = outputs[0].outputs[0].text

                prefill_length = len(outputs[0].prompt_token_ids)
                output_tokens = len(outputs[0].outputs[0].token_ids)
                total_tokens = prefill_length + output_tokens

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
                record["divide_length"] = args.divide_length
                record["protect_prefill"] = args.protect_prefill
                record["enable_experimental_kv_compaction"] = args.enable_experimental_kv_compaction
                record["enable_experimental_block_reclaim"] = (
                    args.enable_experimental_block_reclaim
                )
                record["require_triton_scoring"] = args.require_triton_scoring

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()

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
