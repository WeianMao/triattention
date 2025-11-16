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
    # 指定评估的 JSON 文件所在的文件夹（非单个 JSON 文件）
    parser.add_argument("--base_dir", default="./results", type=str,
                        help="包含待评估 JSON/JSONL 文件的文件夹")
    # 输出目录
    parser.add_argument("--output_dir", default="./output", type=str)
    # 停止词列表
    parser.add_argument("--stop_words", default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："], type=list)
    parser.add_argument("--dataset", default=None, type=str)
    args = parser.parse_args()
    return args

def prepare_data(data_name, args):
    # 使用 load_data_vanilla 加载当前 JSON 文件
    examples = load_data_vanilla(args.input_path)

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    os.makedirs(f"{output_dir}/{args.exp_name}/{data_name}", exist_ok=True)

    # 若有样本去重、过滤等操作，可在此处实现
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
    data_name 为数学数据集名称（此处用 JSON 文件名去除后缀）
    """
    examples, processed_samples = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # 初始化 python executor，根据 prompt_type 判断答案获取方式
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
    # 这里用 size-method 作为文件名的一部分，可根据需要调整
    out_file = os.path.join(out_dir, f"{getattr(args, 'size', 'default')}-{getattr(args, 'method', 'default')}_math_eval.jsonl")
    save_jsonl(all_samples, out_file)

    with open(out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w") as f:
        json.dump(result_json, f, indent=4)
    return result_json

def main_all(args):
    """
    遍历指定文件夹下的所有 JSON/JSONL 文件，对每个文件调用 main() 进行评测，
    并将各结果汇总到 CSV 文件中。
    """
    # 只读取 base_dir 目录下的所有文件（不遍历子目录）
    json_files = []
    for file in os.listdir(args.base_dir):
        filepath = os.path.join(args.base_dir, file)
        if os.path.isfile(filepath) and (file.endswith(".json") or file.endswith(".jsonl")):
            json_files.append(filepath)

    if not json_files:
        print("在文件夹中未找到任何 JSON/JSONL 文件。")
        return

    results_table = {}  # 键为文件名（去除后缀），值为评测得到的 acc
    for json_file in json_files:
        # 使用文件名（不含扩展名）作为数据集名称
        dataset = args.dataset
        args.input_path = json_file
        # 可选：设置一些默认值用于输出文件名，便于区分不同文件的结果
        args.size = "default"
        args.method = dataset
        print(f"Processing: dataset={dataset} from file {json_file}")
        # try:
        result_json = main(dataset, args)
        # except Exception as e:
        #     print(f"Error processing {json_file}: {e}")
        #     continue
        acc = result_json.get("acc", None)
        results_table[json_file] = acc


    # 构造 CSV 表格，第一列为数据集名称，第二列为准确率
    output_csv = os.path.join(args.output_dir, "all_results.csv")
    os.makedirs(args.output_dir, exist_ok=True)
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        header = ["Dataset", "Accuracy"]
        writer.writerow(header)
        for dataset, acc in sorted(results_table.items()):
            writer.writerow([dataset, acc])
    print(f"所有评测结果已保存至 {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    # 调用 main_all 遍历指定文件夹下所有 JSON/JSONL 文件并汇总结果
    main_all(args)
