import os
import argparse
import time
import json
import csv
from tqdm import tqdm

from evaluate import evaluate
from utils import set_seed, load_jsonl, save_jsonl, construct_prompt
from parser import *
from trajectory import *
from data_loader import load_data, load_data_vanilla
from python_executor import PythonExecutor
from model_utils import load_hf_lm_and_tokenizer, generate_completions

import pdb

def parse_args():
    parser = argparse.ArgumentParser()
    # 实验相关参数
    parser.add_argument("--exp_name", default="QwQ-32B-Preview", type=str)
    # prompt 类型，比如 cot, pal 等
    parser.add_argument("--prompt_type", default="cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    # 输出目录
    parser.add_argument("--output_dir", default="./output", type=str)
    # 原代码中只处理单个文件，此处新增参数 base_dir，作为存放各个 deepseek 文件夹的根目录
    parser.add_argument("--base_dir", default="./results", type=str,
                        help="根目录，包含 deepseek-r1-distill-llama-8b_* 文件夹")
    # 停止词列表
    parser.add_argument("--stop_words", default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："], type=list)
    args = parser.parse_args()
    return args

def prepare_data(data_name, args):
    # 使用 load_data_vanilla 加载当前 JSON 文件
    examples = load_data_vanilla(args.input_path)

    # 注意：下面两行示例代码（复制部分）可以根据实际需要修改
    # 如若有样本去重、过滤等操作，可在此处实现

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    os.makedirs(f"{output_dir}/{args.exp_name}/{data_name}", exist_ok=True)

    # processed_samples 预留，如果需要对已有样本做去重等操作
    processed_samples = []
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples

def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True

def main(data_name, args):
    """
    对单个 JSON 文件进行处理和评测
    data_name 为数学数据集名称（如：gsm8k, math 等）
    此处同时利用 args.input_path 指定 JSON 文件
    同时假设 args.size 与 args.method 已在外部设定
    """
    examples, processed_samples = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # 初始化 python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for cnt, example in tqdm(enumerate(examples), total=len(examples)):
        # 对于不同数据集，可能 answer 字段名称不同
        if args.exp_name.lower().find("omni-math") != -1:
            example['solution'] = example['answer']
        else:
            try:
                example['solution'] = example['solution']
            except:
                example['solution'] = example['answer']
        
        idx = example.get("idx", cnt)

        try:
            example["question"] = example['question']
        except:
            example["question"] = example['problem']

        gt_cot, gt_ans = parse_ground_truth(example, data_name)
        example["gt_ans"] = gt_ans

        sample = {
            "idx": idx,
            "question": example["question"],
            "gt_cot": gt_cot,
            "gt": gt_ans,
        }

        for key in [
            "level", "type", "unit", "solution_type", "choices", "solution",
            "ques_type", "ans_type", "answer_type", "dataset", "subfield",
            "filed", "theorem", "answer", "domain", "difficulty", "source",
        ]:
            if key in example:
                sample[key] = example[key]
        samples.append(sample)

    codes = []
    for i in range(len(examples)):
        code = examples[i]['output']
        for stop_word in args.stop_words:
            if stop_word in code:
                code = code.split(stop_word)[0].strip()
        codes.append(code)

    results = [ run_execute(executor, code, args.prompt_type, data_name) for code in codes ]

    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i]
        result = results[i]
        preds = [result[0]]
        reports = [result[1]]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in ["A", "B", "C", "D", "E"]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                preds[j] = "".join([c for c in preds[j] if c in ["A", "B", "C", "D", "E"]])
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    all_samples.extend(processed_samples)
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # 修改输出文件名，包含规模（size）和方法（method）以免覆盖
    out_dir = os.path.join(args.output_dir, args.exp_name, data_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(out_dir, f"{args.size}-{args.method}_math_eval.jsonl")
    save_jsonl(all_samples, out_file)

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json

def main_all(args):
    """
    遍历 base_dir 下所有 deepseek 文件夹、数据集目录及 JSON 文件，
    对每个 JSON 文件调用 main() 进行评测，
    并将各结果按“规模-方法”（行）和“数学数据集”（列）汇总到 CSV 文件中。
    """
    results_table = {}  # 键： "size-method"，值： {data_name: acc}
    datasets_set = set()
    base_dir = args.base_dir

    folders = [
        "deepseek-r1-distill-llama-8b_128",
        "deepseek-r1-distill-llama-70b_128",
        "deepseek-r1-distill-qwen-1.5b_128",
        "deepseek-r1-distill-qwen-7b_128",
        "deepseek-r1-distill-qwen-14b_128",
        "deepseek-r1-distill-qwen-32b_128",
    ]
    # 遍历类似 deepseek-r1-distill-llama-8b_* 的文件夹
    # for deepseek_folder in os.listdir(base_dir):
    for deepseek_folder in folders:
        deepseek_path = os.path.join(base_dir, deepseek_folder)
        if not os.path.isdir(deepseek_path):
            continue
        # 从文件夹名称中提取规模（例如 "deepseek-r1-distill-llama-8b_128" -> "128"）
        if "_" in deepseek_folder:
            size = deepseek_folder.split("_")[-1]
        else:
            size = deepseek_folder
        # 遍历该规模文件夹下的每个数据集文件夹或文件
        for dataset_item in os.listdir(deepseek_path):
            dataset_path = os.path.join(deepseek_path, dataset_item)
            # 若为目录，则 dataset_item 为数据集名称
            if os.path.isdir(dataset_path):
                dataset = dataset_item
                datasets_set.add(dataset)
                # 在数据集目录下递归查找所有 JSON 或 JSONL 文件
                json_files = []
                # for root, dirs, files in os.walk(dataset_path):
                #     for file in files:
                #         if file.endswith(".json") or file.endswith(".jsonl"):
                #             json_files.append(os.path.join(root, file))
                files = ["FullKV.json"]
                for file in files:
                    if file.endswith(".json") or file.endswith(".jsonl"):
                        json_files.append(os.path.join(dataset_path, file))
            else:
                # 若为文件，则将该文件作为一个数据集（data_name 取文件名去掉后缀）
                if dataset_item.endswith(".json") or dataset_item.endswith(".jsonl"):
                    dataset = os.path.splitext(dataset_item)[0]
                    datasets_set.add(dataset)
                    json_files = [dataset_path]
                else:
                    continue

            # 对于当前数据集目录下的每个 JSON 文件，推断方法名称
            for json_file in json_files:
                # 计算相对路径，若 JSON 文件不在数据集目录根部，则第一级目录名作为方法名
                rel_path = os.path.relpath(json_file, dataset_path)
                parts = rel_path.split(os.sep)
                if len(parts) > 1:
                    method = parts[0]
                else:
                    method = os.path.splitext(parts[0])[0]
                row_label = f"{deepseek_folder}-{size}-{method}"
                # 为保证 main() 中输出文件名唯一，将规模和方法传入 args
                args.size = size
                args.method = method
                # 指定当前 JSON 文件作为输入
                args.input_path = json_file
                print(f"Processing: size={size}, dataset={dataset}, method={method}")
                try:
                    result_json = main(dataset, args)
                except Exception as e:
                    print(f"Error processing {json_file}: {e}")
                    continue
                acc = result_json.get("acc", None)
                if row_label not in results_table:
                    results_table[row_label] = {}
                results_table[row_label][dataset] = acc

            # 构造 CSV 表格：行：size-method，列：各数据集
            datasets_list = sorted(list(datasets_set))
            output_csv = os.path.join(args.output_dir, "all_results.csv")
            with open(output_csv, "w", newline="") as csvfile:
                writer = csv.writer(csvfile)
                header = ["Size-Method"] + datasets_list
                writer.writerow(header)
                for row_label in sorted(results_table.keys()):
                    row = [row_label]
                    for ds in datasets_list:
                        row.append(results_table[row_label].get(ds, ""))
                    writer.writerow(row)
            print(f"All results saved to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    # 调用 main_all 遍历所有 JSON 文件并汇总结果
    main_all(args)
