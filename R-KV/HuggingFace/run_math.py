import json
import random
import argparse
import sys
from pathlib import Path

import numpy as np
from tqdm import tqdm

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from rkv.monkeypatch import replace_llama, replace_qwen2, replace_qwen3

RKV_ROOT = Path(__file__).resolve().parents[1]
if str(RKV_ROOT) not in sys.path:
    sys.path.insert(0, str(RKV_ROOT))

from weian_development.process_utils import mask_process_command
from weian_development.rkv_cache_utils import reset_model_cache

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


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


def str2bool(value):
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


def main(args):
    fout = open(args.save_path, "w")

    prompts = []
    test_data = []

    with open(args.dataset_path) as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[args.dataset_name][0]

            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)

            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)

    for sample_idx, prompt in enumerate(tqdm(prompts)):
        tokenized_prompts = tokenizer(
            [prompt],
            padding="longest",
            return_tensors="pt",
            add_special_tokens=True,
        ).to("cuda")
        prefill_length = int(tokenized_prompts["attention_mask"].sum().item())

        for draw_idx in range(args.num_samples):
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

            record = dict(test_data[sample_idx])
            record["prompt"] = prompt
            record["output"] = decoded
            record["prefill_tokens"] = prefill_length
            record["output_tokens"] = output_tokens
            record["total_tokens"] = total_tokens
            record["sample_idx"] = sample_idx
            record["draw_idx"] = draw_idx

            fout.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.cuda.empty_cache()

    fout.close()


def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)
    parser.add_argument("--load_dtype", type=str, default="bfloat16", choices=["bfloat16", "float16"])
    parser.add_argument(
        "--attn_implementation",
        type=str,
        default="flash_attention_2",
        choices=["flash_attention_2", "sdpa", "eager"],
    )

    # method config
    parser.add_argument(
        "--method",
        type=str,
        default=None,
        choices=["rkv", "fullkv", "snapkv", "streamingllm", "h2o"],
    )
    parser.add_argument("--kv_budget", type=int, default=None)
    parser.add_argument("--window_size", type=int, default=8)
    parser.add_argument("--first_tokens", type=int, default=4)
    parser.add_argument("--mix_lambda", type=float, default=0.1)
    parser.add_argument("--retain_ratio", type=float, default=0.2)
    parser.add_argument("--update_kv", type=str2bool, default=True)
    parser.add_argument("--fp32_topk", type=str2bool, default=False)
    parser.add_argument("--reset_cache_each_batch", type=str2bool, default=False)
    parser.add_argument(
        "--retain_direction", type=str, default="last", choices=["last", "first"]
    )

    # model config
    parser.add_argument(
        "--divide_method",
        type=str,
        default="step_length",
        choices=["newline", "step_length"],
    )
    parser.add_argument("--divide_length", type=int, default=128)
    parser.add_argument(
        "--compression_content",
        type=str,
        default="all",
        choices=["think", "all"],
        help="whether to compress the whole model output or only the think part",
    )
    parser.add_argument("--num_samples", type=int, default=64)
    parser.add_argument("--temperature", type=float, default=0.6)
    parser.add_argument("--top_p", type=float, default=0.95)

    return parser.parse_args()


if __name__ == "__main__":
    mask_process_command("PD-L1_binder")
    args = parse_arguments()
    set_seed(args.seed)

    args.dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    if args.dataset_name in dataset2max_length:
        args.max_length = dataset2max_length[args.dataset_name]
    if args.eval_batch_size != 1:
        raise ValueError("eval_batch_size must be 1 for current R-KV HuggingFace path.")

    # ====== build compression config ======
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
        "update_kv": args.update_kv
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

    # apply monkey patch
    if args.method.lower() != "fullkv":
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

    if args.method.lower() != "fullkv":
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

    main(args)
