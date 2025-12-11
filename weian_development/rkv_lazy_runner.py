"""Shard-aware RKV runner for LazyEviction with fairness switches (prompt/decoding/budget/length)."""
from __future__ import annotations

import argparse
import json
import os
import random
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

PROJECT_ROOT = Path(__file__).resolve().parents[1]
for path in (PROJECT_ROOT,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from weian_development.process_utils import mask_process_command
from weian_development.rkv_cache_utils import reset_model_cache
from weian_development.speckv.prompt_utils import (
    DEFAULT_SYSTEM_PROMPT,
    PROMPT_TEMPLATE,
    build_prompt,
    extract_question_from_record,
)
from weian_development.speckv.stats_utils import normalize_dtype_name

# HF compatibility: older Transformers builds may not expose eager_attention_forward.
try:
    from transformers.models import llama as hf_llama_mod
    if not hasattr(hf_llama_mod.modeling_llama, "eager_attention_forward"):
        def _rkv_placeholder_eager_attention_forward(*args, **kwargs):
            raise NotImplementedError("eager_attention_forward is unavailable in this transformers build")
        hf_llama_mod.modeling_llama.eager_attention_forward = _rkv_placeholder_eager_attention_forward
except Exception:
    pass

# HF compatibility: some transformers versions do not expose ALL_ATTENTION_FUNCTIONS.
try:
    import transformers.modeling_utils as hf_modeling_utils
    if not hasattr(hf_modeling_utils, "ALL_ATTENTION_FUNCTIONS"):
        from transformers.models.llama.modeling_llama import _flash_attention_forward
        import torch
        from torch.nn.functional import scaled_dot_product_attention

        def _flash_attention_wrapper(self, query_states, key_states, value_states, attention_mask, dropout=0.0, scaling=None, **kwargs):
            q_len = query_states.shape[2]
            attn_output = _flash_attention_forward(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attention_mask,
                q_len,
                dropout=dropout,
                sliding_window=getattr(self, "sliding_window", None),
                use_top_left_mask=getattr(self, "_flash_attn_uses_top_left_mask", False),
                is_causal=getattr(self, "is_causal", True),
                **kwargs,
            )
            return attn_output, None

        def _sdpa_wrapper(self, query_states, key_states, value_states, attention_mask, dropout=0.0, scaling=None, **kwargs):
            attn_output = scaled_dot_product_attention(
                query_states.transpose(1, 2),
                key_states.transpose(1, 2),
                value_states.transpose(1, 2),
                attn_mask=attention_mask,
                dropout_p=dropout if self.training else 0.0,
                is_causal=getattr(self, "is_causal", True),
            )
            return attn_output, None

        hf_modeling_utils.ALL_ATTENTION_FUNCTIONS = {
            "flash_attention_2": _flash_attention_wrapper,
            "sdpa": _sdpa_wrapper,
            "eager": _sdpa_wrapper,
        }
except Exception:
    pass
from weian_development.rkv_lazy.monkeypatch import replace_llama, replace_qwen2, replace_qwen3

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


def shard_span(total: int, num_shards: int, shard_id: int) -> tuple[int, int]:
    base = total // num_shards
    extra = total % num_shards
    start = shard_id * base + min(shard_id, extra)
    count = base + (1 if shard_id < extra else 0)
    return start, count


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
    prompt_style: str,
    system_prompt: str,
    max_examples: int | None = None,
) -> tuple[List[str], List[dict]]:
    prompts: List[str] = []
    test_data: List[dict] = []
    fallback_keys: List[str] = []
    if dataset_name in dataset2key and dataset2key[dataset_name]:
        fallback_keys.append(dataset2key[dataset_name][0])

    use_chat_template = prompt_style == "chat"
    use_lazy_chat = prompt_style == "lazy_chat"

    with path.open() as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question = extract_question_from_record(example, fallback_keys=fallback_keys)
            example["question"] = question
            if use_lazy_chat:
                prompt = (
                    f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
                    f"<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{{}}.\n"
                    f"{question}<|im_end|>\n"
                    f"<|im_start|>assistant\n<think>\n"
                )
            else:
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
    parser.add_argument("--tokenizer_path", "--tokenizer-path", dest="tokenizer_path", type=str, required=True)
    parser.add_argument("--max_length", "--max-length", dest="max_length", type=int, default=-1)
    parser.add_argument("--max_new_tokens", "--max-new-tokens", dest="max_new_tokens", type=int, default=16384)
    parser.add_argument(
        "--length_mode",
        type=str,
        default="max_new_tokens",
        choices=["max_new_tokens", "legacy_total"],
        help="Whether to cap by max_new_tokens (prefill excluded) or total max_length (legacy).",
    )
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
    parser.add_argument("--num_samples", type=int, default=1)
    parser.add_argument("--do_sample", type=str2bool, default=False)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument(
        "--aggregation",
        type=str,
        default="pass_at_1",
        choices=["pass_at_1", "majority"],
        help="Post-hoc aggregation mode when num_samples>1 (recorded as metadata only).",
    )
    parser.add_argument(
        "--prompt_style",
        type=str,
        default="lazy_chat",
        choices=["plain", "chat", "lazy_chat"],
        help="Plain prompt, tokenizer chat template, or LazyEviction chat prompt with <think>.",
    )
    parser.add_argument(
        "--chat_system_prompt",
        type=str,
        default=DEFAULT_SYSTEM_PROMPT,
        help="System prompt used when --prompt_style chat is enabled.",
    )
    parser.add_argument(
        "--trust_remote_code",
        type=str2bool,
        default=True,
        help="Forward to AutoModelForCausalLM.from_pretrained.",
    )
    parser.add_argument(
        "--count_prefill_in_budget",
        type=str2bool,
        default=False,
        help="If False, expand per-layer kv_budget by prefill length to avoid charging prefills.",
    )
    parser.add_argument(
        "--count_padding_in_budget",
        type=str2bool,
        default=False,
        help="Recorded in metadata; padding is excluded by construction when False.",
    )
    parser.add_argument(
        "--max_examples",
        type=int,
        default=None,
        help="Optional cap on number of dataset examples for quick smoke tests.",
    )
    return parser.parse_args()


def configure_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer


def set_layer_budget(model, budget: int | None) -> None:
    if budget is None:
        return
    layers = getattr(getattr(model, "model", None), "layers", None)
    if not layers:
        return
    for layer in layers:
        attn = getattr(layer, "self_attn", None)
        kv_cluster = getattr(attn, "kv_cluster", None)
        if kv_cluster and hasattr(kv_cluster, "budget"):
            kv_cluster.budget = int(budget)


def build_generation_kwargs(args: argparse.Namespace) -> Dict[str, object]:
    gen_kwargs: Dict[str, object] = {
        "do_sample": bool(args.do_sample),
        "temperature": float(args.temperature),
        "top_p": float(args.top_p),
        "num_beams": 1,
        "num_return_sequences": 1,
    }
    if args.length_mode == "max_new_tokens":
        gen_kwargs["max_new_tokens"] = int(args.max_new_tokens)
    else:
        gen_kwargs["max_length"] = int(args.max_length)
    return gen_kwargs


def main(args: argparse.Namespace) -> None:
    mask_process_command("PD-L1_binder")
    args.dataset_name = Path(args.dataset_path).name.split(".")[0]
    if (not args.max_length) or args.max_length <= 0:
        if args.dataset_name in dataset2max_length:
            args.max_length = dataset2max_length[args.dataset_name]
    if args.eval_batch_size != 1:
        raise ValueError("eval_batch_size must be 1 for current R-KV runner.")

    output_root = Path(args.output_dir)

    method_lower = args.method.lower() if args.method else ""
    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path, use_fast=True, padding_side="left"
    )
    tokenizer = configure_tokenizer(tokenizer)

    prompts, test_data = load_dataset(
        Path(args.dataset_path),
        args.dataset_name,
        tokenizer,
        prompt_style=args.prompt_style,
        system_prompt=args.chat_system_prompt,
        max_examples=args.max_examples,
    )
    total_questions = len(test_data)
    start_q, shard_questions = shard_span(total_questions, args.num_shards, args.shard_id)
    if shard_questions == 0:
        return
    prompts = prompts[start_q : start_q + shard_questions]
    test_data = test_data[start_q : start_q + shard_questions]
    expected_records = len(test_data)

    method_name = method_lower if method_lower else None
    if method_name == "speckv":
        raise ValueError("SpeckV not wired for RKV lazy runner; use dedicated script.")

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

    if method_name and method_name not in {"fullkv"}:
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
        device_map="cuda",
        use_cache=True,
        attn_implementation=args.attn_implementation,
        trust_remote_code=bool(args.trust_remote_code),
    )
    model.eval()
    model.config.update(model_config)

    budget_mode = {
        "count_prefill_in_budget": bool(args.count_prefill_in_budget),
        "count_padding_in_budget": bool(args.count_padding_in_budget),
    }
    gen_kwargs_base = build_generation_kwargs(args)

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

    run_id = args.shard_id
    run_ids = [run_id]
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
                record_id = test_data[local_idx].get("id", sample_idx)
                seed_value = args.seed + run_id * RUN_SEED_STRIDE + sample_idx
                set_seed(seed_value)

                if args.reset_cache_each_batch:
                    reset_model_cache(model)

                effective_budget = args.kv_budget
                if args.kv_budget is not None and args.count_prefill_in_budget:
                    effective_budget = max(args.window_size + 1, args.kv_budget - prefill_length)
                set_layer_budget(model, effective_budget)

                gen_kwargs = dict(gen_kwargs_base)
                if args.length_mode == "legacy_total":
                    gen_kwargs["max_length"] = min(
                        args.max_length if args.max_length > 0 else gen_kwargs_base.get("max_length", 32768),
                        args.max_length,
                    )
                output = model.generate(
                    **tokenized_prompts,
                    **gen_kwargs,
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
                record["kv_budget_requested"] = args.kv_budget
                record["kv_budget_effective"] = effective_budget
                record["budget_mode"] = budget_mode
                record["prompt_style"] = args.prompt_style
                record["do_sample"] = bool(args.do_sample)
                record["temperature"] = float(args.temperature)
                record["top_p"] = float(args.top_p)
                record["aggregation"] = args.aggregation
                record["length_mode"] = args.length_mode
                record["max_new_tokens"] = args.max_new_tokens
                record["max_length_total"] = args.max_length
                record["trust_remote_code"] = bool(args.trust_remote_code)

                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
            torch.cuda.empty_cache()

        meta_tmp = artifacts["meta_tmp"]
        meta = {
            "status": "complete",
            "records": expected_records,
            "run_id": run_id,
            "shard_id": args.shard_id,
            "kv_budget": args.kv_budget,
            "budget_mode": budget_mode,
            "prompt_style": args.prompt_style,
            "do_sample": bool(args.do_sample),
            "aggregation": args.aggregation,
            "length_mode": args.length_mode,
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
