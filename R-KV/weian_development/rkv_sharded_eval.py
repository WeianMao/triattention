"""Shard-aware AIME runner for R-KV HuggingFace backend (non-intrusive copy of run_math)."""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import List

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[2]
MODULE_ROOT = REPO_ROOT / "R-KV"
HF_RKV_ROOT = REPO_ROOT / "R-KV" / "HuggingFace"
for path in (REPO_ROOT, MODULE_ROOT, HF_RKV_ROOT):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from weian_development.process_utils import mask_process_command
from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3

dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 8192,
    "aime24": 16384,
    "math": 8192,
}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


def load_dataset(path: Path, dataset_name: str, shard_id: int, num_shards: int) -> List[dict]:
    prompts: List[str] = []
    test_data: List[dict] = []
    with path.open() as f:
        for index, line in enumerate(f):
            if index % num_shards != shard_id:
                continue
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
    parser.add_argument(
        "--attn_implementation",
        "--attn-implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )
    parser.add_argument("--method", type=str, default=None, choices=["rkv", "fullkv", "snapkv", "streamingllm", "h2o"])
    parser.add_argument("--kv_budget", "--kv-budget", dest="kv_budget", type=int, default=None)
    parser.add_argument("--window_size", "--window-size", dest="window_size", type=int, default=8)
    parser.add_argument("--first_tokens", "--first-tokens", dest="first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", "--mix-lambda", dest="mix_lambda", type=float, default=0.07)
    parser.add_argument("--retain_ratio", "--retain-ratio", dest="retain_ratio", type=float, default=0.2)
    parser.add_argument("--update_kv", "--update-kv", dest="update_kv", type=bool, default=True)
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
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    mask_process_command("PD-L1_binder")
    args.dataset_name = Path(args.dataset_path).name.split(".")[0]
    if args.max_length == -1:
        args.max_length = dataset2max_length[args.dataset_name]

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_name = args.output_name or f"shard{args.shard_id:02d}.jsonl"
    save_path = output_dir / output_name

    prompts, test_data = load_dataset(Path(args.dataset_path), args.dataset_name, args.shard_id, args.num_shards)

    compression_config = {
        "method": args.method,
        "method_config": {
            "budget": args.kv_budget,
            "window_size": args.window_size,
            "mix_lambda": args.mix_lambda,
            "retain_ratio": args.retain_ratio,
            "retain_direction": args.retain_direction,
            "first_tokens": args.first_tokens,
        },
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

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
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

    with save_path.open("w") as fout:
        for i in range(0, len(prompts), args.eval_batch_size):
            batch_prompts = prompts[i : i + args.eval_batch_size]
            tokenized_prompts = tokenizer(
                batch_prompts,
                padding="longest",
                return_tensors="pt",
                add_special_tokens=True,
            ).to("cuda")

            prefill_lengths = tokenized_prompts["attention_mask"].sum(dim=1).tolist()

            output = model.generate(
                **tokenized_prompts,
                max_length=args.max_length,
                do_sample=False,
                num_beams=1,
            )

            batch_token_stats = []
            for j in range(output.size(0)):
                total_tokens = int((output[j] != tokenizer.pad_token_id).sum().item())
                prefill = prefill_lengths[j]
                output_tokens = total_tokens - prefill
                sample_idx = test_data[i + j]["index"]
                batch_token_stats.append(
                    {
                        "sample_idx": sample_idx,
                        "prefill_tokens": prefill,
                        "output_tokens": output_tokens,
                        "total_tokens": total_tokens,
                    }
                )

            batch_outputs = tokenizer.batch_decode(
                [output[j][prefill_lengths[j] :] for j in range(output.size(0))],
                skip_special_tokens=True,
            )

            torch.cuda.empty_cache()

            for j in range(len(batch_outputs)):
                data_idx = i + j
                sample_idx = batch_token_stats[j]["sample_idx"]
                record = test_data[data_idx]
                record["prompt"] = batch_prompts[j]
                record["output"] = batch_outputs[j]
                record["prefill_tokens"] = batch_token_stats[j]["prefill_tokens"]
                record["output_tokens"] = batch_token_stats[j]["output_tokens"]
                record["total_tokens"] = batch_token_stats[j]["total_tokens"]
                record["sample_idx"] = sample_idx
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)
    main(args)
