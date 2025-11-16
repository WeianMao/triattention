import os
import argparse
import time
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
    parser.add_argument("--exp_name", default="QwQ-32B-Preview", type=str)
    parser.add_argument("--data_name", default="gsm8k,math", type=str)
    parser.add_argument("--data_dir", default="./data", type=str)
    parser.add_argument("--output_dir", default="./output", type=str)
    parser.add_argument("--input_path", default="./results/GSM8K_test_QwQ-32B-Preview.jsonl", type=str)
    parser.add_argument("--stop_words", default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："], type=list)
    parser.add_argument("--prompt_type", default="cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    args = parser.parse_args()
    return args


def prepare_data(data_name, args):
    examples = load_data_vanilla(args.input_path)
    
    # select start and end
    # examples = examples[args.start : len(examples) if args.end == -1 else args.end]

    # get out_file name
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        output_dir = f"outputs/{output_dir}"
    os.makedirs(f"{output_dir}/{args.exp_name}/{data_name}", exist_ok=True)

    # load all processed samples
    processed_samples = []

    # dedepulicate
    processed_samples = {sample["idx"]: sample for sample in processed_samples}
    processed_idxs = list(processed_samples.keys())
    processed_samples = list(processed_samples.values())
    examples = [example for example in examples if example["idx"] not in processed_idxs]
    return examples, processed_samples


def setup(args):
    # infer & eval
    data_list = args.data_name.split(",")
    results = []
    for data_name in data_list:
        results.append(main(data_name, args))

    # add "avg" result to data_list and results
    data_list.append("avg")
    results.append(
        {
            "acc": sum([result["acc"] for result in results]) / len(results),
        }
    )

    # print all results
    pad = max([len(data_name) for data_name in data_list])
    print("\t".join(data_name.ljust(pad, " ") for data_name in data_list))
    print("\t".join([f"{result['acc']:.1f}".ljust(pad, " ") for result in results]))


def is_multi_choice(answer):
    for c in answer:
        if c not in ["A", "B", "C", "D", "E"]:
            return False
    return True


def main(data_name, args):
    examples, processed_samples = prepare_data(data_name, args)
    print("=" * 50)
    print("data:", data_name, " ,remain samples:", len(examples))
    if len(examples) > 0:
        print(examples[0])

    # init python executor
    if "pal" in args.prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)

    samples = []
    for cnt, example in tqdm(enumerate(examples), total=len(examples)):

        if args.data_name == "omni-math":
            example['solution'] = example['answer']
        else:
            try:
                example['solution'] = example['solution']
            except:
                example['solution'] = example['answer']
        
        if 'idx' in example.keys():
            idx = example["idx"]
        else:
            idx = cnt

        # parse question and answer
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

        # add remain fields
        for key in [
            "level",
            "type",
            "unit",
            "solution_type",
            "choices",
            "solution",
            "ques_type",
            "ans_type",
            "answer_type",
            "dataset",
            "subfield",
            "filed",
            "theorem",
            "answer",
            "domain",
            "difficulty",
            "source",
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

    results = [
        run_execute(executor, code, args.prompt_type, data_name) for code in codes
    ]

    # put results back to examples
    all_samples = []
    for i, sample in enumerate(samples):
        code = codes[i]
        result = results[i]
        preds = [result[0]]
        reports = [result[1]]
        for j in range(len(preds)):
            if sample["gt"] in ["A", "B", "C", "D", "E"] and preds[j] not in [
                "A",
                "B",
                "C",
                "D",
                "E",
            ]:
                preds[j] = choice_answer_clean(code[j])
            elif is_multi_choice(sample["gt"]) and not is_multi_choice(preds[j]):
                # remove any non-choice char
                preds[j] = "".join(
                    [c for c in preds[j] if c in ["A", "B", "C", "D", "E"]]
                )
        sample.update({"code": code, "pred": preds, "report": reports})
        all_samples.append(sample)

    # add processed samples
    all_samples.extend(processed_samples)
    # pdb.set_trace()
    all_samples, result_json = evaluate(
        samples=all_samples,
        data_name=data_name,
        prompt_type=args.prompt_type,
        execute=True,
    )

    # save outputs
    out_file = f"{args.output_dir}/{args.exp_name}/{data_name}/math_eval.jsonl"
    save_jsonl(all_samples, out_file)


    with open(
        out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json"), "w"
    ) as f:
        json.dump(result_json, f, indent=4)
    return result_json


if __name__ == "__main__":
    args = parse_args()
    setup(args)