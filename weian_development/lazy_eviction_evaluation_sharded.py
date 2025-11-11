import os
import json
import torch
import random
import argparse
import numpy as np
from tqdm import tqdm
from time import time
from copy import deepcopy
from pathlib import Path
from peft import PeftModel
from transformers import AutoTokenizer, AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[1]
LAZY_DIR = ROOT / "LazyEviction"

from eval.utils import generate_completions
from data_processing.process_utils import *
from data_processing.answer_extraction import *
from eval.eval_script import *
from model.temp_cacheobs import TempCache


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

    print("Loading model and tokenizer...")

    tokenizer = AutoTokenizer.from_pretrained(
        args.tokenizer_path,
        trust_remote_code=True,
    )

    # replace KV compression module
    if args.model_type == "llama3":
        from model.monkeypatch import replace_llama 
        replace_llama(args.method.lower())
    elif args.model_type == "qwen":
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
        attn_implementation=args.attn_implementation
    )

    # KV config
    TempCache.alpha = args.alpha
    if args.method not in ["FullKV"]:
        layers = len(model.model.layers)
        for i in range(layers):
            model.model.layers[i].self_attn.config.max_kv_capacity = args.max_kv_capacity
            model.model.layers[i].self_attn.config.decoding_recent_size = args.decoding_recent_size
            if args.method.lower() == "window_lazy":
                model.model.layers[i].self_attn.config.obs_size = args.decoding_recent_size

    # set padding side to left for batch generation
    tokenizer.padding_side = "left"
    # set pad token to eos token if pad token is not set (as is the case for llama models)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

    stop_id_sequences = []
    if tokenizer.eos_token_id is not None:
        stop_id_sequences = [[tokenizer.eos_token_id]]

    torch.cuda.synchronize()
    start_time = time()
    outputs, finish_completion = generate_completions(
        model=model,
        tokenizer=tokenizer,
        prompts=prompts,
        max_new_tokens=args.max_new_tokens,
        do_sample=False,
        temperature=args.temperature,
        top_p=1.0,
        batch_size=args.eval_batch_size,
        stop_id_sequences=stop_id_sequences if stop_id_sequences else None,
        end_of_generation_id_sequence=[tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else None
    )
    torch.cuda.synchronize()
    total_time = time() - start_time
    ## 所有的输出
    model_outputs = outputs
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
    parser.add_argument("--model-type", type=str, choices=['llama3', 'qwen'], default="qwen")
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
    parser.add_argument("--method", type=str,  default=None)
    parser.add_argument("--max_kv_capacity", type=int, default=-1, help="")
    parser.add_argument("--decoding_recent_size", type=int, default=128, help="")
    parser.add_argument("--alpha", type=float, default=0.0001, help="alpha for Lazy")

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
