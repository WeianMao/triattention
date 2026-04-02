# Adapted from ToRA (https://github.com/microsoft/ToRA)
# and DeepSeek-Math (https://github.com/deepseek-ai/DeepSeek-Math)
# Licensed under MIT License

import json
import os
from evaluation.utils import load_jsonl, lower_keys
from datasets import load_dataset, Dataset, concatenate_datasets


# HuggingFace dataset sources for auto-download
HF_DATASET_SOURCES = {
    "aime24": {
        "hf_path": "HuggingFaceH4/aime_2024",
        "field_map": {"problem": "question"},
    },
    "aime25": {
        "hf_path": "MathArena/aime_2025",
        "field_map": {"problem": "question"},
    },
    "math500": {
        "hf_path": "HuggingFaceH4/MATH-500",
        "field_map": {},
    },
}


def _auto_download(data_name, save_path):
    """Download dataset from HuggingFace and save as JSONL."""
    if data_name not in HF_DATASET_SOURCES:
        return False
    source = HF_DATASET_SOURCES[data_name]
    print(f"Dataset '{data_name}' not found locally. Downloading from {source['hf_path']}...")
    dataset = load_dataset(source["hf_path"], split="test")
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    field_map = source["field_map"]
    with open(save_path, "w") as f:
        for example in dataset:
            row = dict(example)
            for old_key, new_key in field_map.items():
                if old_key in row:
                    row[new_key] = row.pop(old_key)
            f.write(json.dumps(row) + "\n")
    print(f"Saved {len(dataset)} examples to {save_path}")
    return True


def load_data_vanilla(path):
    if not os.path.exists(path):
        # Try auto-download based on filename
        basename = os.path.splitext(os.path.basename(path))[0]
        if not _auto_download(basename, path):
            raise FileNotFoundError(f"Dataset not found: {path}")
    examples = list(load_jsonl(path))
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]
    return examples


def load_data(data_name, split, data_dir="./data"):
    data_file = f"{data_dir}/{data_name}/{split}.jsonl"
    if not os.path.exists(data_file):
        # Try auto-download from HuggingFace for known datasets
        _auto_download(data_name, data_file)
    if os.path.exists(data_file):
        examples = list(load_jsonl(data_file))
    else:
        if data_name == "math":
            dataset = load_dataset(
                "competition_math",
                split=split,
                name="main",
                cache_dir=f"{data_dir}/temp",
            )
        elif data_name == "gsm8k":
            dataset = load_dataset(data_name, split=split)
        elif data_name == "svamp":
            # evaluate on training set + test set
            dataset = load_dataset("ChilleD/SVAMP", split="train")
            dataset = concatenate_datasets(
                [dataset, load_dataset("ChilleD/SVAMP", split="test")]
            )
        elif data_name == "asdiv":
            dataset = load_dataset("EleutherAI/asdiv", split="validation")
            dataset = dataset.filter(
                lambda x: ";" not in x["answer"]
            )  # remove multi-answer examples
        elif data_name == "mawps":
            examples = []
            # four sub-tasks
            for data_name in ["singleeq", "singleop", "addsub", "multiarith"]:
                sub_examples = list(load_jsonl(f"{data_dir}/mawps/{data_name}.jsonl"))
                for example in sub_examples:
                    example["type"] = data_name
                examples.extend(sub_examples)
            dataset = Dataset.from_list(examples)
        elif data_name == "mmlu_stem":
            dataset = load_dataset("hails/mmlu_no_train", "all", split="test")
            # only keep stem subjects
            stem_subjects = [
                "abstract_algebra",
                "astronomy",
                "college_biology",
                "college_chemistry",
                "college_computer_science",
                "college_mathematics",
                "college_physics",
                "computer_security",
                "conceptual_physics",
                "electrical_engineering",
                "elementary_mathematics",
                "high_school_biology",
                "high_school_chemistry",
                "high_school_computer_science",
                "high_school_mathematics",
                "high_school_physics",
                "high_school_statistics",
                "machine_learning",
            ]
            dataset = dataset.rename_column("subject", "type")
            dataset = dataset.filter(lambda x: x["type"] in stem_subjects)
        elif data_name == "carp_en":
            dataset = load_jsonl(f"{data_dir}/carp_en/test.jsonl")
        else:
            raise NotImplementedError(data_name)

        examples = list(dataset)
        examples = [lower_keys(example) for example in examples]
        dataset = Dataset.from_list(examples)
        os.makedirs(f"{data_dir}/{data_name}", exist_ok=True)
        dataset.to_json(data_file)

    # add 'idx' in the first column
    if "idx" not in examples[0]:
        examples = [{"idx": i, **example} for i, example in enumerate(examples)]

    # dedepulicate & sort
    examples = sorted(examples, key=lambda x: x["idx"])
    return examples
