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
    # Experiment related parameters
    parser.add_argument("--exp_name", default="QwQ-32B-Preview", type=str)
    # Prompt type, such as cot, pal, etc.
    parser.add_argument("--prompt_type", default="cot", type=str)
    parser.add_argument("--split", default="test", type=str)
    # Output directory
    parser.add_argument("--output_dir", default="./output", type=str)
    # Base directory containing the deepseek folders
    parser.add_argument("--base_dir", default="./results", type=str,
                        help="Base directory containing the deepseek-r1-distill-llama-8b_* folders")
    # Stop words list
    parser.add_argument("--stop_words", default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："], type=list)
    args = parser.parse_args()
    return args

def prepare_data(data_name, args):
    # Load the current JSON file using load_data_vanilla
    examples = load_data_vanilla(args.input_path)

    return examples

def collect_token_lengths(examples):
    prefill_tokens_lengths = []
    output_tokens_lengths = []
    total_tokens_lengths = []

    for example in examples:
        prefill_tokens = example.get("prefill_tokens", 0)
        output_tokens = example.get("output_tokens", 0)
        total_tokens = example.get("total_tokens", 0)

        prefill_tokens_lengths.append(prefill_tokens)
        output_tokens_lengths.append(output_tokens)
        total_tokens_lengths.append(total_tokens)

    return prefill_tokens_lengths, output_tokens_lengths, total_tokens_lengths

def save_token_lengths_to_csv(token_lengths, output_csv):
    header = []
    rows = []

    for model, model_data in token_lengths.items():
        row = [model]
        for dataset, token_data in model_data.items():
            prefill_tokens, output_tokens, total_tokens = token_data
            row.extend([prefill_tokens, output_tokens, total_tokens])
        rows.append(row)

    # Write to CSV file
    with open(output_csv, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Writing the header
        header = ["Model"]  # First column is the model name
        for dataset in token_lengths[list(token_lengths.keys())[0]].keys():
            header.extend([f"{dataset}-pre", f"{dataset}-out", f"{dataset}-tot])

        writer.writerow(header)

        for row in rows:
            writer.writerow(row)

def main(data_name, args):
    """
    Process a single JSON file for token lengths.
    """
    examples = prepare_data(data_name, args)
    
    # Collect token lengths
    prefill_tokens_lengths, output_tokens_lengths, total_tokens_lengths = collect_token_lengths(examples)

    return prefill_tokens_lengths, output_tokens_lengths, total_tokens_lengths

def main_all(args):
    """
    Traverse all deepseek folders and datasets under base_dir, process each JSON file and summarize results.
    """
    base_dir = args.base_dir

    folders = [
        "deepseek-r1-distill-llama-8b_128",
        "deepseek-r1-distill-llama-70b_128",
        "deepseek-r1-distill-qwen-1.5b_128",
        "deepseek-r1-distill-qwen-7b_128",
        "deepseek-r1-distill-qwen-14b_128",
        "deepseek-r1-distill-qwen-32b_128",
    ]

    token_lengths = {}

    for deepseek_folder in folders:
        deepseek_path = os.path.join(base_dir, deepseek_folder)
        if not os.path.isdir(deepseek_path):
            continue
        # Extract size and method from folder name
        if "_" in deepseek_folder:
            size = deepseek_folder.split("_")[-1]
        else:
            size = deepseek_folder

        for dataset_item in os.listdir(deepseek_path):
            dataset_path = os.path.join(deepseek_path, dataset_item, "FullKV.json")
            if dataset_path.endswith(".json") or dataset_path.endswith(".jsonl"):
                dataset = os.path.splitext(dataset_item)[0]
                args.input_path = dataset_path

                print(f"Processing: model={deepseek_folder}, dataset={dataset}")

                try:
                    prefill_tokens, output_tokens, total_tokens = main(dataset, args)
                    # Store the token lengths for this model and dataset
                    if deepseek_folder not in token_lengths:
                        token_lengths[deepseek_folder] = {}
                    token_lengths[deepseek_folder][dataset] = (sum(prefill_tokens)//len(prefill_tokens), sum(output_tokens)//len(output_tokens), sum(total_tokens)//len(total_tokens))
                except Exception as e:
                    print(f"Error processing {dataset}: {e}")
                    continue

    # import pdb
    # pdb.set_trace()
    # Save the token lengths to CSV
    output_csv = os.path.join(args.output_dir, "token_lengths.csv")
    save_token_lengths_to_csv(token_lengths, output_csv)
    print(f"Token lengths saved to {output_csv}")

if __name__ == "__main__":
    args = parse_args()
    main_all(args)
