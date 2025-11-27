"""Shard-aware AIME runner for R-KV HuggingFace backend (non-intrusive copy of run_math)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation.logits_process import (
    InfNanRemoveLogitsProcessor,
    LogitsProcessorList,
    TemperatureLogitsWarper,
    TopKLogitsWarper,
    TopPLogitsWarper,
)
from weian_development.hf_offline_runner_sparse.sparse_round_pruner_prefill_keep import (
    SparsePruningConfig,
    SparseRoundPruner,
)

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = REPO_ROOT / "R-KV"
HF_RKV_ROOT = REPO_ROOT / "R-KV" / "HuggingFace"
for path in (REPO_ROOT, MODULE_ROOT, HF_RKV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from weian_development.process_utils import mask_process_command
from weian_development.rkv_cache_utils import reset_model_cache
from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3

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

DEFAULT_CHAT_SYSTEM = "You are a helpful assistant."
RUN_SEED_STRIDE = 1_000_000


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


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


def load_dataset(path: Path, dataset_name: str, shard_id: int, num_shards: int) -> List[dict]:
    prompts: List[str] = []
    test_data: List[dict] = []
    with path.open() as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[dataset_name][0]
            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)
            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)
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
        choices=["rkv", "fullkv", "snapkv", "streamingllm", "h2o", "sparse_round_prefill_keep", "speckv"],
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
        help="Stats file for sparse round pruning (required when method=sparse_round_prefill_keep).",
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
        "--sparse_score_aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Aggregation strategy for sparse round pruning scores.",
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
        "--use_chat_template",
        type=str2bool,
        default=True,
        help="Wrap prompts with tokenizer.apply_chat_template when using sparse pruning (SpeckV baseline uses chat).",
    )
    parser.add_argument(
        "--chat_system_prompt",
        type=str,
        default=DEFAULT_CHAT_SYSTEM,
        help="System prompt used when --use_chat_template is enabled for sparse pruning.",
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


def compute_available_new_tokens(max_length: int, prompt_length: int, model, tokenizer) -> int:
    candidate_context_lengths: List[int] = []
    for attr in (
        "max_position_embeddings",
        "max_sequence_length",
        "n_positions",
        "max_seq_len",
        "window",
        "context_length",
    ):
        value = getattr(model.config, attr, None)
        if isinstance(value, int) and value > 0:
            candidate_context_lengths.append(value)

    tokenizer_limit = getattr(tokenizer, "model_max_length", None)
    if isinstance(tokenizer_limit, int) and 0 < tokenizer_limit < 10**7:
        candidate_context_lengths.append(tokenizer_limit)

    context_cap = max(candidate_context_lengths) if candidate_context_lengths else None
    if max_length and max_length > 0:
        context_cap = max_length if context_cap is None else min(context_cap, max_length)
    if context_cap is None:
        return 1
    available = context_cap - prompt_length
    return max(1, available)


def build_sampling_components(
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
) -> tuple[LogitsProcessorList, LogitsProcessorList]:
    processors = LogitsProcessorList([InfNanRemoveLogitsProcessor()])
    warpers = LogitsProcessorList()
    if not do_sample:
        return processors, warpers
    if temperature and temperature != 1.0:
        warpers.append(TemperatureLogitsWarper(temperature))
    if top_k and top_k > 0:
        warpers.append(TopKLogitsWarper(top_k))
    if 0 < top_p < 1.0:
        warpers.append(TopPLogitsWarper(top_p))
    return processors, warpers


def sample_next_token(
    logits: torch.Tensor,
    sequence: torch.Tensor,
    do_sample: bool,
    processors: LogitsProcessorList,
    warpers: LogitsProcessorList,
) -> torch.Tensor:
    processed = processors(sequence, logits)
    if not do_sample:
        return torch.argmax(processed, dim=-1, keepdim=True)
    warped = warpers(sequence, processed)
    probs = F.softmax(warped, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1)
    return next_token


def speckv_generate_sequence(
    model: AutoModelForCausalLM,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int],
    do_sample: bool,
    temperature: float,
    top_p: float,
    top_k: int,
    pruner: SparseRoundPruner,
) -> torch.Tensor:
    if max_new_tokens <= 0:
        return input_ids.clone()

    processors, warpers = build_sampling_components(do_sample, temperature, top_p, top_k)
    sequence = input_ids.clone()

    with torch.inference_mode():
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )

    logits = outputs.logits[:, -1, :]
    past_key_values = outputs.past_key_values
    if isinstance(past_key_values, Cache):
        pkv_tuple = past_key_values.to_legacy_cache()
    elif isinstance(past_key_values, tuple):
        pkv_tuple = past_key_values
        past_key_values = DynamicCache.from_legacy_cache(pkv_tuple)
    else:
        pkv_tuple = past_key_values

    pruner.attach_initial_cache(pkv_tuple)
    pkv_tuple = pruner.enforce_max_limit(pkv_tuple)
    pkv_tuple = pruner.ensure_capacity(pkv_tuple)
    cache_for_model = DynamicCache.from_legacy_cache(pkv_tuple) if isinstance(pkv_tuple, tuple) else pkv_tuple

    generated: List[int] = []

    for _ in range(max_new_tokens):
        next_token = sample_next_token(logits, sequence, do_sample, processors, warpers)
        token_id = int(next_token.item())
        generated.append(token_id)
        next_token_tensor = torch.tensor([[token_id]], device=input_ids.device, dtype=torch.long)
        sequence = torch.cat([sequence, next_token_tensor], dim=1)

        if eos_token_id is not None and token_id == eos_token_id:
            break

        with torch.inference_mode():
            position_ids = torch.tensor(
                [[pruner.absolute_position]],
                device=next_token_tensor.device,
                dtype=torch.long,
            )
            outputs = model(
                input_ids=next_token_tensor,
                attention_mask=None,
                past_key_values=cache_for_model,
                use_cache=True,
                return_dict=True,
                position_ids=position_ids,
                cache_position=position_ids,
            )

        past_key_values = outputs.past_key_values
        if isinstance(past_key_values, Cache):
            pkv_tuple = past_key_values.to_legacy_cache()
        elif isinstance(past_key_values, tuple):
            pkv_tuple = past_key_values
            past_key_values = DynamicCache.from_legacy_cache(pkv_tuple)
        else:
            pkv_tuple = past_key_values

        pruner.on_token_appended()
        if pruner.should_start_next_round():
            pkv_tuple = pruner.start_next_round(pkv_tuple)

        cache_for_model = DynamicCache.from_legacy_cache(pkv_tuple) if isinstance(pkv_tuple, tuple) else pkv_tuple
        logits = outputs.logits[:, -1, :]

    if generated:
        gen_tensor = torch.tensor(generated, device=input_ids.device, dtype=torch.long).unsqueeze(0)
        full_sequence = torch.cat([input_ids, gen_tensor], dim=1)
    else:
        full_sequence = input_ids.clone()

    return full_sequence


def build_sparse_pruner_config(args: argparse.Namespace, device: torch.device, max_keys: int, model_path: Path) -> SparsePruningConfig:
    stats_path = args.sparse_stats_path
    if not stats_path:
        raise ValueError("sparse_stats_path is required for sparse_round_prefill_keep.")
    resolved_stats = Path(stats_path)
    if not resolved_stats.is_absolute():
        resolved_stats = (REPO_ROOT / resolved_stats).resolve()
    if not resolved_stats.exists():
        raise FileNotFoundError(f"Sparse stats file not found: {resolved_stats}")

    round_window = args.sparse_round_window if args.sparse_round_window and args.sparse_round_window > 0 else args.window_size
    return SparsePruningConfig(
        stats_path=resolved_stats,
        model_path=model_path,
        device=device,
        dtype=torch.float32,
        max_keys=max_keys,
        round_window=round_window,
        offset_max_length=args.sparse_offset_max_length,
        score_aggregation=args.sparse_score_aggregation,
        seed=args.sparse_seed,
        head_limit=args.sparse_head_limit,
    )


def format_prompt(question: str, tokenizer: AutoTokenizer, use_chat_template: bool, system_prompt: str) -> str:
    base_prompt = prompt_template.format(question=question, answer="")
    if not use_chat_template:
        return base_prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": base_prompt},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


def run_sparse_round_generation(
    args: argparse.Namespace,
    tokenizer: AutoTokenizer,
    model,
    question: str,
    max_length: int,
    pruner_cfg: SparsePruningConfig,
) -> tuple[str, int, int, str]:
    prompt = format_prompt(question, tokenizer, args.use_chat_template, args.chat_system_prompt)
    tokenized_prompts = tokenizer(
        [prompt],
        padding="longest",
        return_tensors="pt",
        add_special_tokens=True,
    ).to(model.device)
    prompt_length = int(tokenized_prompts["attention_mask"].sum().item())
    max_new_tokens = compute_available_new_tokens(max_length, prompt_length, model, tokenizer)

    pruner = SparseRoundPruner(pruner_cfg)
    sequence = speckv_generate_sequence(
        model=model,
        input_ids=tokenized_prompts["input_ids"],
        attention_mask=tokenized_prompts["attention_mask"],
        max_new_tokens=max_new_tokens,
        eos_token_id=tokenizer.eos_token_id,
        do_sample=True,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=0,
        pruner=pruner,
    )

    total_tokens = int(sequence[0].numel())
    output_tokens = total_tokens - prompt_length
    decoded = tokenizer.decode(sequence[0][prompt_length:], skip_special_tokens=True)
    return decoded, prompt_length, output_tokens, prompt


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
    use_sparse_round = method_lower in {"sparse_round_prefill_keep", "speckv"}
    if use_sparse_round and not args.use_chat_template:
        raise ValueError("SpeckV/sparse_round_prefill_keep requires chat template to align with R-KV baseline.")

    if use_sparse_round and args.kv_budget is None:
        raise ValueError("kv_budget must be provided for sparse_round_prefill_keep.")

    prompts, test_data = load_dataset(Path(args.dataset_path), args.dataset_name, args.shard_id, args.num_shards)
    if args.max_examples and args.max_examples > 0:
        prompts = prompts[: args.max_examples]
        test_data = test_data[: args.max_examples]
    expected_records = len(test_data)
    if expected_records == 0:
        return

    if use_sparse_round:
        tokenizer = configure_tokenizer(
            AutoTokenizer.from_pretrained(args.model_path, use_fast=True, padding_side="left")
        )
        dtype = resolve_torch_dtype(args.load_dtype)
        model = AutoModelForCausalLM.from_pretrained(
            args.model_path,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
            use_cache=True,
            attn_implementation=args.attn_implementation,
        )
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        pruner_cfg = build_sparse_pruner_config(
            args,
            device=device,
            max_keys=int(args.kv_budget),
            model_path=Path(args.model_path),
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
                for example in test_data:
                    question = example["question"]
                    sample_idx = example["index"]
                    seed_value = args.seed + run_id * RUN_SEED_STRIDE + sample_idx
                    set_seed(seed_value)
                    if args.reset_cache_each_batch:
                        reset_model_cache(model)

                    decoded, prefill_tokens, output_tokens, rendered_prompt = run_sparse_round_generation(
                        args,
                        tokenizer,
                        model,
                        question,
                        args.max_length,
                        pruner_cfg,
                    )
                    total_tokens = prefill_tokens + output_tokens

                    record = dict(example)
                    record["prompt"] = rendered_prompt
                    record["output"] = decoded
                    record["prefill_tokens"] = prefill_tokens
                    record["output_tokens"] = output_tokens
                    record["total_tokens"] = total_tokens
                    record["sample_idx"] = sample_idx
                    record["draw_idx"] = run_id

                    fout.write(json.dumps(record, ensure_ascii=False) + "\n")
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
        return

    method_config = {"budget": args.kv_budget, "window_size": args.window_size}
    if args.method in {"rkv", "snapkv"}:
        method_config.update(
            {
                "mix_lambda": args.mix_lambda,
                "retain_ratio": args.retain_ratio,
                "retain_direction": args.retain_direction,
                "first_tokens": args.first_tokens,
                "fp32_topk": args.fp32_topk,
            }
        )
    elif args.method == "streamingllm":
        method_config.update({"first_tokens": args.first_tokens})

    compression_config = {
        "method": args.method,
        "method_config": method_config,
        "compression": None,
        "update_kv": args.update_kv,
    }
    model_config = {
        "divide_method": args.divide_method,
        "divide_length": args.divide_length,
        "compression_content": args.compression_content,
    }

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, use_fast=True, padding_side="left"
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    if args.method and args.method.lower() != "fullkv":
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

    if args.method and args.method.lower() != "fullkv":
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
                seed_value = args.seed + run_id * RUN_SEED_STRIDE + sample_idx
                set_seed(seed_value)

                if args.reset_cache_each_batch:
                    reset_model_cache(model)

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
