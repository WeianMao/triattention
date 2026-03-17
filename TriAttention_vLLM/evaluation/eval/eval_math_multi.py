import os
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List

from tqdm import tqdm

from evaluate import evaluate
from utils import save_jsonl
from python_executor import PythonExecutor
from parser import run_execute


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate multi-sample math generations (pass@k-style).")
    parser.add_argument("--exp_name", default="multi_eval", type=str)
    parser.add_argument("--prompt_type", default="cot", type=str)
    parser.add_argument(
        "--base_dir",
        default="./data",
        type=str,
        help="Folder containing JSON/JSONL files to be evaluated (merged outputs).",
    )
    parser.add_argument("--output_dir", default="./output_multi", type=str)
    parser.add_argument(
        "--stop_words",
        default=["</s>", "<|im_end|>", "<|endoftext|>", "\n题目："],
        type=list,
    )
    parser.add_argument("--dataset", default=None, type=str)
    parser.add_argument("--num_samples", type=int, default=None, help="Expected draws per question (optional).")
    return parser.parse_args()


def strip_stop_words(text: str, stop_words: List[str]) -> str:
    for stop_word in stop_words:
        if stop_word in text:
            text = text.split(stop_word)[0].strip()
    return text


def load_outputs(base_dir: Path) -> List[Dict]:
    items: List[Dict] = []
    for file in base_dir.iterdir():
        if not file.is_file():
            continue
        if not (file.suffix in {".json", ".jsonl"}):
            continue
        with file.open() as fp:
            for line in fp:
                try:
                    items.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
    if not items:
        raise FileNotFoundError(f"No JSON/JSONL files found under {base_dir}")
    return items


def group_by_sample(items: List[Dict]) -> Dict[int, Dict]:
    grouped: Dict[int, Dict] = defaultdict(lambda: {"meta": {}, "preds": {}})
    for item in items:
        sample_idx = item.get("sample_idx", item.get("idx"))
        if sample_idx is None:
            continue
        draw_idx = item.get("draw_idx", 0)
        grouped[sample_idx]["meta"] = grouped[sample_idx]["meta"] or item
        grouped[sample_idx]["preds"][draw_idx] = item.get("output", item.get("generation", ""))
    return grouped


def prepare_samples(grouped: Dict[int, Dict], data_name: str, stop_words: List[str], prompt_type: str) -> List[Dict]:
    # Use dataset loader shape to align auxiliary fields (level/type/etc.) if present.
    samples: List[Dict] = []
    if "pal" in prompt_type:
        executor = PythonExecutor(get_answer_expr="solution()")
    else:
        executor = PythonExecutor(get_answer_from_stdout=True)
    for sample_idx in sorted(grouped.keys()):
        info = grouped[sample_idx]
        meta = info["meta"]
        preds_map = info["preds"]
        preds_text = [preds_map[k] for k in sorted(preds_map.keys())]

        sample = {"idx": sample_idx}
        # Carry over common fields if present
        for key in ["question", "problem", "answer", "solution", "type", "level", "dataset", "choices"]:
            if key in meta:
                sample[key] = meta[key]
        if "question" not in sample and "problem" in sample:
            sample["question"] = sample["problem"]

        codes = [strip_stop_words(text or "", stop_words) for text in preds_text]
        preds = []
        reports = []
        for code in codes:
            result = run_execute(executor, code, prompt_type, data_name)
            preds.append(result[0])
            reports.append(result[1])

        sample["pred"] = preds
        sample["report"] = reports
        sample["code"] = codes
        samples.append(sample)
    return samples


def main():
    args = parse_args()
    base_dir = Path(args.base_dir)
    items = load_outputs(base_dir)
    if args.dataset:
        data_name = args.dataset
    else:
        # fall back to parent directory name
        data_name = base_dir.name

    grouped = group_by_sample(items)
    if args.num_samples:
        mismatched = [idx for idx, info in grouped.items() if len(info["preds"]) != args.num_samples]
        if mismatched:
            print(f"[warn] {len(mismatched)} samples have counts != expected {args.num_samples}")
    samples = prepare_samples(grouped, data_name, args.stop_words, args.prompt_type)

    # Evaluate (pass@1 is column-mean of multiple preds per sample)
    all_samples, result_json = evaluate(
        data_name=data_name,
        prompt_type=args.prompt_type,
        samples=samples,
        execute=True,
    )

    out_dir = os.path.join(args.output_dir, args.exp_name, data_name)
    os.makedirs(out_dir, exist_ok=True)
    out_file = os.path.join(
        out_dir,
        f"{getattr(args, 'size', 'default')}-{getattr(args, 'method', 'default')}_math_multi_eval.jsonl",
    )
    save_jsonl(all_samples, out_file)

    metrics_path = out_file.replace(".jsonl", f"_{args.prompt_type}_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(result_json, f, indent=4)

    pred_counts = [len(info["preds"]) for info in grouped.values()]
    summary = {
        "num_samples_total": len(items),
        "num_questions": len(grouped),
        "num_preds_per_question_expected": args.num_samples,
        "num_preds_per_question_min": min(pred_counts) if pred_counts else 0,
        "num_preds_per_question_max": max(pred_counts) if pred_counts else 0,
    }
    summary.update(result_json)
    print(summary)


if __name__ == "__main__":
    main()
