"""Shard-aware AIME runner for R-KV HuggingFace backend (non-intrusive copy of run_math)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

RKV_ROOT = Path(__file__).resolve().parents[1]
HF_RKV_ROOT = RKV_ROOT / "HuggingFace"
for path in (HF_RKV_ROOT, RKV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from weian_development.process_utils import mask_process_command
from weian_development.rkv_cache_utils import reset_model_cache
from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3
from rkv.compression.speckv import apply_speckv_generate_patch
from weian_development.speckv.prompt_utils import (
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
    build_prompt,
    extract_question_from_record,
)
from weian_development.speckv.stats_utils import normalize_dtype_name
try:
    from weian_development.rkv_debug.qk_capture import (
        activate_capture,
        deactivate_capture,
        capture_requested_for_sample,
        patch_llama_attention_for_capture,
    )
except Exception:  # pragma: no cover - fail open
    def activate_capture(*args, **kwargs):
        return

    def deactivate_capture():
        return

    def capture_requested_for_sample(*args, **kwargs):
        return False

    def patch_llama_attention_for_capture():
        return False

dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "aime25": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 8192,
    "aime24": 32768,
    "aime25": 32768,
    "math": 8192,
}

RUN_SEED_STRIDE = 1_000_000


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


def str2bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    value = value.strip().lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Unable to interpret boolean value '{value}'")


def resolve_torch_dtype(name: str):
    normalized = name.lower()
    if normalized == "bfloat16":
        return torch.bfloat16
    if normalized == "float16":
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def resolve_under_rkv(path_like: str | Path) -> Path:
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    parts = path.parts
    if parts and parts[0] == "R-KV":
        path = Path(*parts[1:]) if len(parts) > 1 else Path(".")
    return (RKV_ROOT / path).resolve()


def compute_local_runs(num_samples: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    base = num_samples // num_shards
    extra = num_samples % num_shards
    start = shard_id * base + min(shard_id, extra)
    count = base + (1 if shard_id < extra else 0)
    return start, count


def shard_run_dir(base_dir: Path, shard_id: int) -> Path:
    return base_dir / f"shard{shard_id:02d}"


def run_artifacts(base_dir: Path, shard_id: int, run_id: int) -> dict[str, Path]:
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


def load_dataset(
    path: Path,
    dataset_name: str,
    tokenizer: AutoTokenizer,
    *,
    use_chat_template: bool,
    system_prompt: str,
    max_examples: int | None = None,
) -> tuple[List[str], List[dict]]:
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
            prompt = build_prompt(
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", "--dataset-path", dest="dataset_path", type=str, required=True)
    parser.add_argument("--output_dir", "--output-dir", dest="output_dir", type=str, required=True)
    parser.add_argument("--model_path", "--model-path", dest="model_path", type=str, required=True)
    parser.add_argument("--max_length", "--max-length", dest="max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", "--eval-batch-size", dest="eval_batch_size", type=int, default=1)
    parser.add_argument("--load_dtype", "--load-dtype", dest="load_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--attn_implementation",
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["rkv", "fullkv", "snapkv", "streamingllm", "h2o", "speckv"],
    )
    parser.add_argument("--kv_budget", "--kv-budget", dest="kv_budget", type=int, default=None)
    parser.add_argument("--window_size", "--window-size", dest="window_size", type=int, default=8)
    parser.add_argument("--first_tokens", "--first-tokens", dest="first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", "--mix-lambda", dest="mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", "--retain-ratio", dest="retain_ratio", type=float, default=0.2)
    parser.add_argument("--update_kv", "--update-kv", dest="update_kv", type=str2bool, default=True)
    parser.add_argument("--fp32_topk", "--fp32-topk", dest="fp32_topk", type=str2bool, default=False)
    parser.add_argument(
        "--reset_cache_each_batch",
        "--reset-cache-each-batch",
        dest="reset_cache_each_batch",
        type=str2bool,
        default=False,
    )
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )
    parser.add_argument(
        "--divide_method",
        "--divide-method",
        type=str,
        default="step_length",
        choices=["newline", "step_length"],
    )
    parser.add_argument("--divide_length", "--divide-length", dest="divide_length", type=int, default=128)
    parser.add_argument(
        "--compression_content",
        "--compression-content",
        type=str,
        default="all",
        choices=["think", "all"],
        help="whether to compress the whole model output or only the think part",
    )
    parser.add_argument("--shard_id", type=int, required=True)
    parser.add_argument("--num_shards", type=int, required=True)
    parser.add_argument("--output_name", type=str, default=None)
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument(
        "--sparse_stats_path",
        type=str,
        default=None,
        help="Stats file for SpeckV sparse round pruning (required when method=speckv).",
    )
    parser.add_argument(
        "--sparse_round_window",
        type=int,
        default=None,
        help="Round window for sparse pruning (defaults to window_size when unset).",
    )
    parser.add_argument(
        "--sparse_offset_max_length",
        type=int,
        default=65536,
        help="Maximum offset length for sparse pruning frequency scoring.",
    )
    parser.add_argument(
        "--disable_top_n_high_freq",
        type=int,
        default=0,
        help="Disable top-n high-frequency components in position-dependent scoring (0=disabled).",
    )
    parser.add_argument(
        "--simulate_bug_phase_offset",
        type=int,
        default=0,
        help="Simulate bug 896cbca6 phase offset: subtract N×ω from phase (0=disabled, typical Δ≈156).",
    )
    parser.add_argument(
        "--sparse_score_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Aggregation strategy for sparse round pruning scores.",
    )
    parser.add_argument(
        "--sparse_normalize_scores",
        type=str2bool,
        default=False,
        help="Normalize per-head sparse scores before aggregation.",
    )
    parser.add_argument(
        "--use_rank_aggregation",
        type=str2bool,
        default=False,
        help="Use rank-based aggregation (min-pooling) instead of z-score + max-pooling.",
    )
    parser.add_argument(
        "--sparse_use_similarity",
        type=str2bool,
        default=False,
        help="Enable Similarity Deduplication in SpecKV (combines frequency with R-KV similarity).",
    )
    parser.add_argument(
        "--sparse_similarity_mix_lambda",
        type=float,
        default=0.1,
        help="Mix lambda for similarity scoring: final = freq * lambda - sim * (1-lambda). Needs search: 0.1, 0.3, 0.5, 0.7, 0.9.",
    )
    parser.add_argument(
        "--use_rank_similarity_combination",
        type=str2bool,
        default=False,
        help="Enable rank+similarity combination with normalized inverted rank direction.",
    )
    parser.add_argument(
        "--sparse_seed",
        type=int,
        default=0,
        help="Seed used by sparse pruner for noise / head shuffling.",
    )
    parser.add_argument(
        "--sparse_head_limit",
        type=int,
        default=None,
        help="Optional head limit for sparse stats (None keeps all sampled heads).",
    )
    parser.add_argument(
        "--per_head_pruning",
        type=str2bool,
        default=False,
        help="Enable per-KV-head independent pruning (each head selects tokens independently). Default: False",
    )
    parser.add_argument(
        "--use_chat_template",
        type=str2bool,
        default=False,
        help="Wrap prompts with tokenizer.apply_chat_template when using sparse pruning.",
    )
    parser.add_argument(
        "--chat_system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when --use_chat_template is enabled for sparse pruning.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on number of dataset examples for quick smoke tests.",
    )
    # Alignment args for fair R-KV comparison
    parser.add_argument(
        "--include_prefill_in_budget",
        "--include-prefill-in-budget",
        dest="include_prefill_in_budget",
        type=str2bool,
        default=False,
        help="Include prefill tokens in budget calculation (aligns with R-KV behavior).",
    )
    parser.add_argument(
        "--rkv_style_compression",
        "--rkv-style-compression",
        dest="rkv_style_compression",
        type=str2bool,
        default=False,
        help="Use R-KV style attention-layer compression instead of generate wrapper.",
    )
    parser.add_argument(
        "--rkv_style_slack_trigger",
        "--rkv-style-slack-trigger",
        dest="rkv_style_slack_trigger",
        type=str2bool,
        default=False,
        help="For R-KV style compression, trigger pruning at budget + divide_length (like generate wrapper).",
    )
    parser.add_argument(
        "--rkv_aligned_budget",
        "--rkv-aligned-budget",
        dest="rkv_aligned_budget",
        type=str2bool,
        default=False,
        help="Align budget calculation with R-KV: compress to exact budget instead of budget - round_window.",
    )
    parser.add_argument(
        "--allow_prefill_compression",
        "--allow-prefill-compression",
        dest="allow_prefill_compression",
        type=str2bool,
        default=False,
        help="Allow prefill tokens to be compressed (R-KV style). When False, prefill is always preserved.",
    )
    # Note: --divide_length is already defined above (line 238) for R-KV, reused for SpeckV alignment
    return parser.parse_args()


def configure_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def main(args: argparse.Namespace) -> None:
    mask_process_command("PD-L1_binder")
    args.dataset_name = Path(args.dataset_path).name.split(".")[0]
    if (not args.max_length) or args.max_length <= 0:
        if args.dataset_name in dataset2max_length:
            args.max_length = dataset2max_length[args.dataset_name]
    if args.eval_batch_size != 1:
        raise ValueError("eval_batch_size must be 1 for current R-KV sharded runner.")

    total_samples = args.num_samples
    start_draw, local_samples = compute_local_runs(total_samples, args.num_shards, args.shard_id)
    if local_samples == 0:
        return

    run_ids = list(range(start_draw, start_draw + local_samples))
    output_root = Path(args.output_dir)

    method_lower = args.method.lower() if args.method else ""
    if method_lower == "speckv":
        # SpeckV paper/baseline uses plain prompt (no chat); do not allow chat mode.
        if args.kv_budget is None:
            raise ValueError("kv_budget must be provided for speckv.")
        if bool(args.use_chat_template):
            raise ValueError("SpeckV uses the plain R-KV prompt; use_chat_template must be False.")

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )
    tokenizer = configure_tokenizer(tokenizer)

    prompt_use_chat = False
    prompts, test_data = load_dataset(
        Path(args.dataset_path),
        args.dataset_name,
        tokenizer,
        use_chat_template=prompt_use_chat,
        system_prompt=args.chat_system_prompt,
        max_examples=args.max_examples,
    )
    expected_records = len(test_data)
    if expected_records == 0:
        return

    method_name = method_lower if method_lower else None
    speckv_method_config: Dict[str, object] = {}
    if method_lower == "speckv":
        if args.kv_budget is None:
            raise ValueError("kv_budget must be provided for speckv.")
        speckv_method_config = {
            "kv_budget": args.kv_budget,
            "window_size": args.window_size,
            "sparse_stats_path": args.sparse_stats_path,
            "sparse_round_window": args.sparse_round_window or args.window_size,
            "sparse_offset_max_length": args.sparse_offset_max_length,
            "disable_top_n_high_freq": args.disable_top_n_high_freq,
            "sparse_score_aggregation": args.sparse_score_aggregation,
            "sparse_head_limit": args.sparse_head_limit,
            "sparse_seed": args.sparse_seed,
            "sparse_normalize_scores": args.sparse_normalize_scores,
            "use_rank_aggregation": args.use_rank_aggregation,
            "sparse_use_similarity": args.sparse_use_similarity,
            "sparse_similarity_mix_lambda": args.sparse_similarity_mix_lambda,
            "use_rank_similarity_combination": args.use_rank_similarity_combination,
        }

    method_config = {"budget": args.kv_budget, "window_size": args.window_size}
    if method_name in {"rkv", "snapkv"}:
        method_config.update(
            {
                "mix_lambda": args.mix_lambda,
                "retain_ratio": args.retain_ratio,
                "retain_direction": args.retain_direction,
                "first_tokens": args.first_tokens,
                "fp32_topk": args.fp32_topk,
            }
        )
    elif method_name == "streamingllm":
        method_config.update({"first_tokens": args.first_tokens})
    if method_lower == "speckv":
        method_config = speckv_method_config

    compression_config = {
        "method": method_name,
        "method_config": method_config,
        "compression": None,
        "update_kv": args.update_kv,
    }
    model_config = {
        "divide_method": args.divide_method,
        "divide_length": args.divide_length,
        "compression_content": args.compression_content,
    }

    if method_name and method_name not in {"fullkv", "speckv"}:
        if "llama" in args.model_path.lower():
            replace_llama(compression_config)
        elif "qwen3" in args.model_path.lower():
            replace_qwen3(compression_config)
        elif "qwen" in args.model_path.lower():
            replace_qwen2(compression_config)
        else:
            raise ValueError(f"Unsupported model: {args.model_path}")

    dtype = resolve_torch_dtype(args.load_dtype)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto",
        use_cache=True,
        attn_implementation=args.attn_implementation,
    )
    model.eval()
    model.config.update(model_config)

    capture_root = os.environ.get("RKV_QK_CAPTURE_DIR")
    capture_root_path = Path(capture_root).expanduser() if capture_root else None
    capture_model_info = {
        "model_path": args.model_path,
        "dataset_path": args.dataset_path,
        "kv_budget": args.kv_budget,
        "window_size": args.window_size,
        "method": method_name,
        "attn_implementation": args.attn_implementation,
        "load_dtype": args.load_dtype,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    if capture_root_path:
        patched = patch_llama_attention_for_capture()
        if not patched:
            sys.stderr.write("[qk_capture] failed to patch LlamaAttention for capture; proceeding without QK dumps.\n")

    if method_name and method_name not in {"fullkv", "speckv"}:
        model.newline_token_ids = [
            tokenizer.encode("\n")[-1],
            tokenizer.encode(".\n")[-1],
            tokenizer.encode(")\n")[-1],
            tokenizer.encode("\n\n")[-1],
            tokenizer.encode(".\n\n")[-1],
            tokenizer.encode(")\n\n")[-1],
        ]
        model.after_think_token_ids = [
            tokenizer.encode("</think>")[-1],
        ]
    elif method_lower == "speckv":
        if args.sparse_stats_path is None:
            raise ValueError("sparse_stats_path must be provided for speckv.")
        stats_path = resolve_under_rkv(args.sparse_stats_path)
        if not stats_path.exists():
            raise FileNotFoundError(f"SpeckV stats file not found: {stats_path}")
        round_window = args.sparse_round_window if args.sparse_round_window and args.sparse_round_window > 0 else args.window_size
        metadata_expectations = {
            "prompt_template": PROMPT_TEMPLATE,
            "use_chat_template": prompt_use_chat,
            "system_prompt": args.chat_system_prompt if prompt_use_chat else "",
            "attn_implementation": args.attn_implementation,
            "dtype": normalize_dtype_name(dtype),
            "kv_budget": int(args.kv_budget),
        }
        if args.rkv_style_compression:
            # Use R-KV style attention-layer compression
            from weian_development.speckv.speckv_rkv_style import apply_speckv_rkv_style_patch
            apply_speckv_rkv_style_patch(
                model,
                stats_path=stats_path,
                model_path=Path(args.model_path),
                kv_budget=int(args.kv_budget),
                offset_max_length=args.sparse_offset_max_length,
                score_aggregation=args.sparse_score_aggregation,
                sparse_seed=args.sparse_seed,
                head_limit=args.sparse_head_limit,
                metadata_expectations=metadata_expectations,
                normalize_scores=args.sparse_normalize_scores,
                use_rank_aggregation=args.use_rank_aggregation,
                include_prefill_in_budget=args.include_prefill_in_budget,
                divide_length=args.divide_length,
                use_slack_trigger=args.rkv_style_slack_trigger,
            )
        else:
            # Use original generate wrapper implementation
            apply_speckv_generate_patch(
                model,
                stats_path=stats_path,
                model_path=Path(args.model_path),
                kv_budget=int(args.kv_budget),
                round_window=round_window,
                offset_max_length=args.sparse_offset_max_length,
                score_aggregation=args.sparse_score_aggregation,
                sparse_seed=args.sparse_seed,
                head_limit=args.sparse_head_limit,
                metadata_expectations=metadata_expectations,
                normalize_scores=args.sparse_normalize_scores,
                use_rank_aggregation=args.use_rank_aggregation,
                sparse_use_similarity=args.sparse_use_similarity,
                sparse_similarity_mix_lambda=args.sparse_similarity_mix_lambda,
                use_rank_similarity_combination=args.use_rank_similarity_combination,
                include_prefill_in_budget=args.include_prefill_in_budget,
                per_head_pruning=args.per_head_pruning,
                rkv_aligned_budget=args.rkv_aligned_budget,
                divide_length=args.divide_length,
                allow_prefill_compression=args.allow_prefill_compression,
                disable_top_n_high_freq=args.disable_top_n_high_freq,
                simulate_bug_phase_offset=args.simulate_bug_phase_offset,
            )

    for run_id in run_ids:
        artifacts = run_artifacts(output_root, args.shard_id, run_id)
        if run_is_complete(artifacts["run"], artifacts["meta"], expected_records):
            continue
        for stale in ("meta", "meta_tmp", "tmp"):
            path = artifacts[stale]
            if path.exists():
                path.unlink()

        with artifacts["tmp"].open("w") as fout:
            for local_idx, prompt in enumerate(prompts):
                tokenized_prompts = tokenizer(
                    [prompt],
                    padding="longest",
                    return_tensors="pt",
                    add_special_tokens=True,
                ).to("cuda")
                prefill_length = int(tokenized_prompts["attention_mask"].sum().item())
                sample_idx = test_data[local_idx]["index"]
                record_id = int(test_data[local_idx].get("id", sample_idx))
                seed_value = args.seed + run_id * RUN_SEED_STRIDE + sample_idx
                set_seed(seed_value)

                if args.reset_cache_each_batch:
                    reset_model_cache(model)

                if capture_root_path and capture_requested_for_sample(record_id):
                    activate_capture(
                        capture_root_path,
                        shard_id=args.shard_id,
                        run_id=run_id,
                        sample_id=record_id,
                        prefill_length=prefill_length,
                        model_info=capture_model_info,
                    )
                else:
                    deactivate_capture()

                output = model.generate(
                    **tokenized_prompts,
                    max_length=args.max_length,
                    do_sample=True,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    num_beams=1,
                    num_return_sequences=1,
                )

                total_tokens = int((output[0] != tokenizer.pad_token_id).sum().item())
                output_tokens = total_tokens - prefill_length
                decoded = tokenizer.decode(
                    output[0][prefill_length:], skip_special_tokens=True
                )

                record = dict(test_data[local_idx])
                record["prompt"] = prompt
                record["output"] = decoded
                record["prefill_tokens"] = prefill_length
                record["output_tokens"] = output_tokens
                record["total_tokens"] = total_tokens
                record["sample_idx"] = sample_idx
                record["draw_idx"] = run_id

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                deactivate_capture()
            torch.cuda.empty_cache()

        meta_tmp = artifacts["meta_tmp"]
        meta = {
            "status": "complete",
            "records": expected_records,
            "run_id": run_id,
            "shard_id": args.shard_id,
        }
        with meta_tmp.open("w") as fp:
            json.dump(meta, fp)
        meta_tmp.replace(artifacts["meta"])
        artifacts["tmp"].replace(artifacts["run"])
    torch.cuda.empty_cache()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    main(args)
