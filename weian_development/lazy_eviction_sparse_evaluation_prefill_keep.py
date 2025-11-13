import os
import json
import random
from copy import deepcopy
from pathlib import Path
from time import time
from typing import List, Optional, Sequence, Tuple

import argparse
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import sys

ROOT = Path(__file__).resolve().parents[1]
LAZY_DIR = ROOT / "LazyEviction"
DEFAULT_SPARSE_STATS = Path("weian_development/hf_offline_runner_sparse/stats/distill_qwen7b_qid9001_trace00_stats.pt")
SPARSE_METHOD = "sparse_round_prefill"

if str(LAZY_DIR) not in sys.path:
    sys.path.insert(0, str(LAZY_DIR))

from eval.utils import generate_completions
from data_processing.process_utils import *
from data_processing.answer_extraction import *
from eval.eval_script import *
from model.temp_cacheobs import TempCache
from weian_development.hf_offline_runner_sparse.example_offline_hf_serialized import (
    run_sparse_generation,
)
from weian_development.hf_offline_runner_sparse.sparse_round_pruner_prefill_keep import (
    SparsePruningConfig,
    SparseRoundPruner,
)


def resolve_lazy_path(path_str: str) -> str:
    path = Path(path_str)
    if not path.is_absolute():
        path = LAZY_DIR / path
    return str(path)

def set_random_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def read_data(path):
    if path.endswith("json"):
        data = json.load(open(path, "r"))
    elif path.endswith("jsonl"):
        data = []
        with open(path, "r") as file:
            for line in file:
                line = json.loads(line)
                data.append(line)
    else:
        raise NotImplementedError()
    return data

def is_sparse_method(method: Optional[str]) -> bool:
    return bool(method and method.lower() == SPARSE_METHOD)

def resolve_project_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    return path

def build_sparse_pruner_config(args, device: torch.device) -> SparsePruningConfig:
    stats_path = args.sparse_stats_path or DEFAULT_SPARSE_STATS
    stats_path = resolve_project_path(stats_path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Sparse stats file not found: {stats_path}")
    round_window = args.sparse_round_window
    if round_window is None or round_window <= 0:
        round_window = args.decoding_recent_size
    return SparsePruningConfig(
        stats_path=stats_path,
        model_path=resolve_project_path(args.model_path),
        device=device,
        dtype=torch.float32,
        max_keys=args.max_kv_capacity,
        round_window=round_window,
        offset_max_length=args.sparse_offset_max_length,
        score_aggregation=args.sparse_score_aggregation,
        seed=args.sparse_seed,
        head_limit=args.sparse_head_limit,
    )

def configure_tokenizer(tokenizer: AutoTokenizer) -> AutoTokenizer:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id
    return tokenizer

def compute_available_new_tokens(args, model, tokenizer, prompt_length: int) -> int:
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

    if candidate_context_lengths:
        model_context_length = max(candidate_context_lengths)
    else:
        model_context_length = prompt_length + max(1, args.max_new_tokens if args.max_new_tokens > 0 else 32768)
    available_context = max(1, model_context_length - prompt_length)
    requested = args.max_new_tokens
    if requested is None or requested <= 0:
        return available_context
    return max(1, min(requested, available_context))

def run_lazy_method(args, tokenizer, prompts: Sequence[str]) -> Tuple[List[str], float]:
    print("Loading model and tokenizer for LazyEviction baseline...", flush=True)

    tokenizer = configure_tokenizer(tokenizer)

    if args.model_type == "llama3":
        from model.monkeypatch import replace_llama
        replace_llama(args.method.lower())
    elif args.model_type in {"qwen", "qwen3"}:
        from model.monkeypatch import replace_qwen
        replace_qwen(args.method.lower())

    print(f"Working on max_kv_capacity {args.max_kv_capacity}")

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        trust_remote_code=True,
        device_map="auto",
        low_cpu_mem_usage=True,
        use_cache=args.use_cache,
        attn_implementation=args.attn_implementation,
    )

    TempCache.alpha = args.alpha
    if args.method not in ["FullKV"]:
        layers = len(model.model.layers)
        for i in range(layers):
            model.model.layers[i].self_attn.config.max_kv_capacity = args.max_kv_capacity
            model.model.layers[i].self_attn.config.decoding_recent_size = args.decoding_recent_size
            if args.method and args.method.lower() == "window_lazy":
                model.model.layers[i].self_attn.config.obs_size = args.decoding_recent_size

    stop_id_sequences: List[List[int]] = []
    if tokenizer.eos_token_id is not None:
        stop_id_sequences = [[tokenizer.eos_token_id]]

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    start_time = time()
    outputs, _ = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=args.temperature,
        top_p=1.0,
        batch_size=args.eval_batch_size,
        stop_id_sequences=stop_id_sequences if stop_id_sequences else None,
        end_of_generation_id_sequence=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None,
    )
    if torch.cuda.is_available():
        torch.cuda.synchronize()
    total_time = time() - start_time
    return outputs, total_time

def run_sparse_method(args, tokenizer, prompts: Sequence[str]) -> Tuple[List[str], float]:
    if args.max_kv_capacity <= 0:
        raise ValueError("SparseRound requires --max_kv_capacity to be a positive integer.")
    print("Loading model for SparseRound evaluation...", flush=True)

    tokenizer = configure_tokenizer(tokenizer)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    dtype = torch.float16 if device.type == "cuda" else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        trust_remote_code=True,
        torch_dtype=dtype,
        attn_implementation=args.attn_implementation,
        low_cpu_mem_usage=True,
        use_cache=True,
    )
    model.to(device)
    model.eval()

    if args.eval_batch_size != 1:
        print("SparseRound currently enforces eval_batch_size=1; overriding requested batch size.", flush=True)

    pruner_config = build_sparse_pruner_config(args, device)
    do_sample = args.temperature > 0.0

    outputs: List[str] = []
    if device.type == "cuda":
        torch.cuda.synchronize()
    start_time = time()
    for prompt in prompts:
        prompt_inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = prompt_inputs["input_ids"].to(device)
        prompt_length = int(input_ids.shape[-1])
        max_new_tokens = compute_available_new_tokens(args, model, tokenizer, prompt_length)
        pruner = SparseRoundPruner(pruner_config)
        sequence, _ = run_sparse_generation(
            model=model,
            input_ids=input_ids,
            max_new_tokens=max_new_tokens,
            eos_token_id=tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=args.temperature,
            top_p=1.0,
            top_k=0,
            store_logprobs=False,
            pruner=pruner,
        )
        completion_ids = sequence[:, prompt_length:]
        completion_text = tokenizer.batch_decode(
            completion_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        outputs.append(completion_text.strip())
    if device.type == "cuda":
        torch.cuda.synchronize()
    total_time = time() - start_time
    return outputs, total_time

def infer(args, test_data, answer_extraction_fn):
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, trust_remote_code=True)

    prompts = []
    num = 0
    for example in test_data:
        num+=1
        prompt = ""
        for mess in example['messages']:
            if mess['role'] == 'user':
                if args.model_type == 'llama3':
                    prompt += f"{tokenizer.bos_token}" + "<|user|>\n\nPlease reason step by step, and put your final answer within \\boxed{}.\n" + f"{mess['content']}\n{tokenizer.eos_token}<|assistant|><think>\n"
                elif args.model_type == 'qwen':
                    prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n" + f"{mess['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                elif args.model_type == 'qwen3':
                    prompt += "<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n<|im_start|>user\nPlease reason step by step, and put your final answer within \\boxed{}.\n" + f"{mess['content']}<|im_end|>\n<|im_start|>assistant\n<think>\n"
                else:
                    raise NotImplementedError()
            elif mess['role'] == 'assistant':
                prompt += mess['content'].rstrip()
            prompt = prompt.lstrip()
        example['prompt'] = prompt
        prompts.append(prompt)

    if is_sparse_method(args.method):
        model_outputs, total_time = run_sparse_method(args, tokenizer, prompts)
    else:
        model_outputs, total_time = run_lazy_method(args, tokenizer, prompts)

    cot_lengths = []
    for model_completion in model_outputs:
        cot = model_completion.split('\n\nThe final answer is:')[0]
        cot_length = tokenizer(cot, return_tensors="pt")['input_ids'].shape[1]
        cot_lengths.append(cot_length)

    predictions = [eval(answer_extraction_fn)(item['messages'][-2]['content'], output, task='cot') for item, output in tqdm(zip(test_data, model_outputs), desc="extract answer", total=len(model_outputs))]
    assert len(model_outputs) > 0, f"{len(model_outputs)}"
    print("predictions:", predictions)
    results = []
    for example, output, pred, cot_length in zip(test_data, model_outputs, predictions, cot_lengths):
        item = deepcopy(example)
        item.update({
            'model_output': output,
            'prediction': pred,
            'cot_length': cot_length,
        })
        results.append(item)
    return results, total_time


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, default="outputs/Qwen2.5-7B-Instruct/gsm8k/", help="default to `model_path`_predictions")
    parser.add_argument("--model-path", type=str, default="/your_model_path/Qwen2.5-7B-Instruct")
    parser.add_argument("--tokenizer-path", type=str, default="/your_model_path/Qwen2.5-7B-Instruct")
    parser.add_argument("--model-size", type=str, choices=['1.5b', '4b', '3b', '7b', '13b', '32b', '33b', '34b', '70b'], default="7b")
    parser.add_argument("--model-type", type=str, choices=['llama3', 'qwen', 'qwen3'], default="qwen")
    parser.add_argument("--benchmark", type=str, choices=['gsm8k', 'math','aime'], default="gsm8k")
    parser.add_argument("--data-type", type=str, choices=['train', 'test', 'test_for_vis'], default="test")

    parser.add_argument("--max_num_examples", type=int, default=100000000000000, help="maximum number of examples to evaluate.")
    parser.add_argument("--max_new_tokens", type=int, default=512)
    parser.add_argument("--eval_batch_size", type=int, default=16, help="batch size for evaluation.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_shards", type=int, default=1, help="total number of shards for data parallel evaluation")
    parser.add_argument("--shard_id", type=int, default=0, help="0-indexed shard identifier for the current process")

    # for KV compression
    parser.add_argument("--use_cache", type=bool, default=True, help="")
    parser.add_argument("--attn_implementation", type=str,  default="eager", choices=["flash_attention_2", "sdpa", "eager"])
    parser.add_argument("--method", type=str,  default=SPARSE_METHOD)
    parser.add_argument("--max_kv_capacity", type=int, default=-1, help="")
    parser.add_argument("--decoding_recent_size", type=int, default=128, help="")
    parser.add_argument("--alpha", type=float, default=0.0001, help="alpha for Lazy")
    parser.add_argument("--sparse_stats_path", type=str, default=str(DEFAULT_SPARSE_STATS), help="Path to sparse stats for SparseRound method")
    parser.add_argument("--sparse_offset_max_length", type=int, default=65536, help="Maximum offset length for sparse scoring")
    parser.add_argument("--sparse_score_aggregation", type=str, default="mean", choices=["mean", "max"], help="Aggregation for sparse round scores")
    parser.add_argument("--sparse_head_limit", type=int, default=None, help="Optional head limit for SparseRound stats")
    parser.add_argument("--sparse_seed", type=int, default=0, help="Random seed for SparseRound pruner noise")
    parser.add_argument("--sparse_round_window", type=int, default=None, help="Round window override for SparseRound method")

    args, unparsed_args = parser.parse_known_args()

    if args.num_shards < 1:
        raise ValueError(f"num_shards must be >= 1, got {args.num_shards}")
    if not 0 <= args.shard_id < args.num_shards:
        raise ValueError(f"shard_id must be in [0, {args.num_shards - 1}], got {args.shard_id}")

    # os.environ['CUDA_VISIBLE_DEVICES'] = "0"

    print(f"Evaluating {args.model_path}", flush=True)
    print(f"Max new tokens: {args.max_new_tokens}, eval batch size: {args.eval_batch_size}, temperature: {args.temperature}, seed: {args.seed}\n", flush=True)
    args.output_dir = os.path.join(args.output_dir, f"{args.model_size}/", f"Original/{args.data_type}/", f"{args.method}/")
    if args.num_shards > 1:
        args.output_dir = os.path.join(args.output_dir, f"shard_{args.shard_id:02d}")
    config_path = LAZY_DIR / "configs" / f"{args.benchmark}_{args.data_type}.json"
    test_conf = read_data(str(config_path))
    for src, info in test_conf.items():
        fname = os.path.join(args.output_dir, "test_data", "test.jsonl")
        input_dir = os.path.dirname(fname)
        os.makedirs(input_dir, exist_ok=True)
        metric_path = os.path.join(args.output_dir, "samples", "metrics.json")
        if os.path.exists(metric_path) and read_data(metric_path)['n_samples'] > 0:
            continue   
        with open(fname, "w") as file:
            data = read_data(resolve_lazy_path(info['test_path']))  
            for i, sample in enumerate(tqdm(data, desc=f'processing {src}')):
                fn = eval(info['process_fn']) 
                sample['id'] = sample.get('id', f"{src}-{i}")
                for j, item in enumerate(fn(sample)):
                    item['dataset'] = src
                    item['id'] = f"{src}-test-{i}-{j}"
                    assert 'answer' in item
                    print(json.dumps(item), file=file, flush=True)

            output_dir = os.path.join(args.output_dir, f"{args.max_kv_capacity}", "samples")
            
            os.makedirs(output_dir, exist_ok=True)

        set_random_seed(args.seed)

        print("Loading data...")
        test_data = []  
        
        with open(os.path.join(input_dir, f"test.jsonl")) as fin:
            for line in fin:
                example = json.loads(line)
                messages = example['messages']
                assert messages[-1]['role'] == 'assistant'
                example['reference'] = example.get('reference', '') or [mess['content'] for mess in messages if
                                                                        mess['role'] == 'assistant']
                for mess in messages:
                    if mess['role'] == 'assistant':
                        mess['content'] = ''
                example['messages'] = messages
                test_data.append(example)

        if args.max_num_examples and len(test_data) > args.max_num_examples:
            test_data = random.sample(test_data, args.max_num_examples)

        total_examples = len(test_data)
        if args.num_shards > 1:
            test_data = [example for idx, example in enumerate(test_data) if idx % args.num_shards == args.shard_id]
            print(f"Shard {args.shard_id + 1}/{args.num_shards}: {len(test_data)} / {total_examples} samples", flush=True)
            if not test_data:
                print("Shard has no samples, skipping inference.", flush=True)
                continue

        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)

        results, total_time = infer(args, test_data, info['answer_extraction_fn'])

        print("Finished inference...")

        os.environ['TOKENIZERS_PARALLELISM'] = "false"

        invalid_outputs = []
        labels = []
        for item in results:
            if len(item['prediction']) == 0:
                invalid_outputs.append({'prompt': item['prompt'], 'output':  item['model_output'], 'answer': item['prediction']})
                res = False
                extract_ans = None
            else:
                extract_ans = item['prediction']
                res = eval_math(item)
            labels.append(res)

        for item, label in zip(results, labels):
            item['accuracy'] = label

        print("Calculating accuracy...")
        acc = 0
        for item in results:
            acc += item['accuracy']
        print("output acc = {:.5f}".format(acc / len(results) * 100), flush=True)

        avg_cot_length = sum(item['cot_length'] for item in results) / len(results)
        print("output avg_cot_length = {:.5f}".format(avg_cot_length), flush=True)

        print("number of invalid outputs: {}".format(len(invalid_outputs)), flush=True)

        pred_fname = "predictions.jsonl"
        for item in results:
            with open(os.path.join(output_dir, pred_fname), 'a+', encoding='utf-8') as fout:
                line = json.dumps(item, ensure_ascii=False)
                fout.write(line + '\n')

        metric_fname = "metrics.json"
        with open(os.path.join(output_dir, metric_fname), "w") as fout:
            json.dump({
                "n_samples": len(results),
                "accuracy": sum(item['accuracy'] for item in results) / len(results),
                "avg_cot_length": avg_cot_length,
                'sample_latency': total_time / len(test_data),
            }, fout, indent=4)
