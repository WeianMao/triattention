#!/usr/bin/env python3
"""Compare HuggingFace SpeckV and vLLM TriAttention benchmark results."""

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np


ExtractAnswerFn = Callable[[str, str], str]
ParseGroundTruthFn = Callable[[Dict, str], Tuple[str, str]]
MathEqualFn = Callable[..., bool]


def _record_key(item: Dict) -> str:
    """Build a stable key for cross-run alignment."""
    if item.get("id") is not None:
        return f"id:{item['id']}"
    if item.get("index") is not None:
        return f"index:{item['index']}"
    question = item.get("question")
    if isinstance(question, str) and question:
        return f"question:{question}"
    return f"fallback:{json.dumps(item, sort_keys=True, ensure_ascii=False)}"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare HuggingFace and vLLM reasoning benchmark results"
    )
    parser.add_argument("--hf-results", type=str, required=True, help="Path to HuggingFace results JSONL")
    parser.add_argument("--vllm-results", type=str, required=True, help="Path to vLLM results JSONL")
    parser.add_argument("--output-report", type=str, default="comparison_report.txt", help="Path to save report")
    parser.add_argument("--dataset-name", type=str, default="aime24", help="Dataset name for parser/grader")
    parser.add_argument(
        "--strict-math-eval",
        dest="strict_math_eval",
        action="store_true",
        default=True,
        help="Use evaluation/eval parser+grader for answer matching (default: enabled).",
    )
    parser.add_argument(
        "--no-strict-math-eval",
        dest="strict_math_eval",
        action="store_false",
        help="Disable parser+grader and use simple extraction fallback.",
    )
    parser.add_argument("--detailed", action="store_true", help="Generate detailed per-question comparison")
    return parser.parse_args()


def _load_eval_helpers(
    strict_enabled: bool,
) -> Tuple[Optional[ExtractAnswerFn], Optional[ParseGroundTruthFn], Optional[MathEqualFn], Optional[str]]:
    """Load parser/grader helpers with graceful fallback."""
    if not strict_enabled:
        return None, None, None, "strict math eval disabled by flag"

    triattention_root = Path(__file__).resolve().parents[2]
    eval_dir = triattention_root / "evaluation" / "eval"
    if str(eval_dir) not in sys.path:
        sys.path.insert(0, str(eval_dir))

    try:
        from parser import extract_answer as eval_extract_answer  # type: ignore
        from parser import parse_ground_truth  # type: ignore
        from grader import math_equal  # type: ignore
    except Exception as exc:
        return None, None, None, f"failed to import parser/grader: {exc}"
    return eval_extract_answer, parse_ground_truth, math_equal, None


def load_results(results_path: str) -> List[Dict]:
    """Load and normalize results from JSONL, grouped by stable record key."""
    rows: List[Dict] = []
    with open(results_path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            item = json.loads(line.strip())
            if "output" in item and "generated_answers" not in item:
                item["generated_answers"] = [item["output"]]
            if "generated_answers" not in item:
                item["generated_answers"] = []
            if "answer" in item and "ground_truth" not in item:
                item["ground_truth"] = item["answer"]
            rows.append(item)

    grouped: Dict[str, Dict] = {}
    for item in rows:
        key = _record_key(item)
        if key in grouped:
            grouped[key]["generated_answers"].extend(item.get("generated_answers", []))
            if not grouped[key].get("ground_truth") and item.get("ground_truth"):
                grouped[key]["ground_truth"] = item.get("ground_truth")
        else:
            grouped[key] = item
    return list(grouped.values())


def _fallback_extract_answer(text: str) -> str:
    boxed_match = re.search(r"\\boxed\{([^}]+)\}", text or "")
    if boxed_match:
        return boxed_match.group(1).strip()
    numbers = re.findall(r"-?\d+(?:\.\d+)?", text or "")
    if numbers:
        return numbers[-1]
    return ""


def _normalize_ground_truth(
    item: Dict,
    dataset_name: str,
    parse_ground_truth_fn: Optional[ParseGroundTruthFn],
) -> str:
    if parse_ground_truth_fn is not None:
        answer = item.get("ground_truth", item.get("answer", item.get("gt")))
        if answer is not None:
            candidate = {"answer": answer}
            try:
                _, parsed = parse_ground_truth_fn(candidate, dataset_name)
                return str(parsed).strip()
            except Exception:
                pass
    return str(item.get("ground_truth", item.get("answer", ""))).strip()


def _prediction_matches(
    text: str,
    ground_truth: str,
    dataset_name: str,
    extract_answer_fn: Optional[ExtractAnswerFn],
    math_equal_fn: Optional[MathEqualFn],
) -> bool:
    if extract_answer_fn is not None and math_equal_fn is not None:
        try:
            pred = str(extract_answer_fn(text or "", dataset_name)).strip()
            return bool(math_equal_fn(pred, ground_truth, timeout=True))
        except Exception:
            pass
    pred = _fallback_extract_answer(text)
    return pred == ground_truth


def calculate_accuracy(
    results: List[Dict],
    dataset_name: str,
    extract_answer_fn: Optional[ExtractAnswerFn],
    parse_ground_truth_fn: Optional[ParseGroundTruthFn],
    math_equal_fn: Optional[MathEqualFn],
) -> Tuple[float, int, int]:
    """Calculate question-level accuracy (correct if any draw is correct)."""
    num_correct = 0
    num_total = 0

    for item in results:
        ground_truth = _normalize_ground_truth(item, dataset_name, parse_ground_truth_fn)
        if not ground_truth:
            continue

        generated_answers = item.get("generated_answers", [])
        if any(
            _prediction_matches(
                gen_text,
                ground_truth,
                dataset_name,
                extract_answer_fn,
                math_equal_fn,
            )
            for gen_text in generated_answers
        ):
            num_correct += 1
        num_total += 1

    accuracy = num_correct / num_total if num_total > 0 else 0.0
    return accuracy, num_correct, num_total


def compare_token_sequences(hf_text: str, vllm_text: str) -> Dict:
    """Compare two generated text sequences token-by-token.

    Returns:
        Dictionary with comparison metrics
    """
    # Simple word-level comparison (can be enhanced with actual tokenizer)
    hf_tokens = hf_text.split()
    vllm_tokens = vllm_text.split()

    min_len = min(len(hf_tokens), len(vllm_tokens))
    max_len = max(len(hf_tokens), len(vllm_tokens))

    # Count matching tokens in prefix
    matching_prefix = 0
    for i in range(min_len):
        if hf_tokens[i] == vllm_tokens[i]:
            matching_prefix += 1
        else:
            break

    # Total matches
    total_matches = sum(1 for h, v in zip(hf_tokens, vllm_tokens) if h == v)

    return {
        "hf_length": len(hf_tokens),
        "vllm_length": len(vllm_tokens),
        "matching_prefix": matching_prefix,
        "total_matches": total_matches,
        "match_ratio": total_matches / max_len if max_len > 0 else 0.0,
        "length_diff": abs(len(hf_tokens) - len(vllm_tokens)),
    }


def generate_comparison_report(
    hf_results: List[Dict],
    vllm_results: List[Dict],
    args: argparse.Namespace,
    helper_status: str,
    extract_answer_fn: Optional[ExtractAnswerFn],
    parse_ground_truth_fn: Optional[ParseGroundTruthFn],
    math_equal_fn: Optional[MathEqualFn],
) -> str:
    """Generate detailed comparison report.

    Returns:
        Report string
    """
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("HuggingFace vs vLLM Benchmark Comparison Report")
    report_lines.append("=" * 80)
    report_lines.append("")

    # Summary statistics
    report_lines.append("Summary Statistics:")
    report_lines.append("-" * 80)
    report_lines.append(f"HuggingFace Results: {len(hf_results)} questions")
    report_lines.append(f"vLLM Results: {len(vllm_results)} questions")
    report_lines.append(f"Math Eval Mode: {helper_status}")
    report_lines.append("")

    # Calculate accuracies
    hf_acc, hf_correct, hf_total = calculate_accuracy(
        hf_results,
        args.dataset_name,
        extract_answer_fn,
        parse_ground_truth_fn,
        math_equal_fn,
    )
    vllm_acc, vllm_correct, vllm_total = calculate_accuracy(
        vllm_results,
        args.dataset_name,
        extract_answer_fn,
        parse_ground_truth_fn,
        math_equal_fn,
    )

    report_lines.append("Accuracy Metrics:")
    report_lines.append(f"  HuggingFace: {hf_acc:.2%} ({hf_correct}/{hf_total})")
    report_lines.append(f"  vLLM:        {vllm_acc:.2%} ({vllm_correct}/{vllm_total})")
    report_lines.append(f"  Difference:  {abs(hf_acc - vllm_acc):.2%}")
    report_lines.append("")

    # Align results by stable record key
    hf_by_id = {_record_key(item): item for item in hf_results}
    vllm_by_id = {_record_key(item): item for item in vllm_results}
    common_ids = set(hf_by_id.keys()) & set(vllm_by_id.keys())

    if common_ids:
        report_lines.append(f"Common Questions: {len(common_ids)}")
        report_lines.append("")

        # Token-level comparison for common questions
        if args.detailed:
            report_lines.append("Detailed Per-Question Comparison:")
            report_lines.append("-" * 80)

            match_ratios = []
            for qid in sorted(common_ids):
                hf_item = hf_by_id[qid]
                vllm_item = vllm_by_id[qid]

                # Compare first sample from each
                hf_answers = hf_item.get("generated_answers") or [""]
                vllm_answers = vllm_item.get("generated_answers") or [""]
                hf_text = hf_answers[0]
                vllm_text = vllm_answers[0]

                comparison = compare_token_sequences(hf_text, vllm_text)
                match_ratios.append(comparison["match_ratio"])

                report_lines.append(f"\nQuestion ID: {qid}")
                report_lines.append(f"  HF Length:      {comparison['hf_length']} tokens")
                report_lines.append(f"  vLLM Length:    {comparison['vllm_length']} tokens")
                report_lines.append(f"  Match Ratio:    {comparison['match_ratio']:.2%}")
                report_lines.append(f"  Prefix Match:   {comparison['matching_prefix']} tokens")
                report_lines.append(f"  Length Diff:    {comparison['length_diff']} tokens")

            report_lines.append("")
            report_lines.append("Overall Token Match Statistics:")
            report_lines.append(f"  Mean Match Ratio:   {np.mean(match_ratios):.2%}")
            report_lines.append(f"  Median Match Ratio: {np.median(match_ratios):.2%}")
            report_lines.append(f"  Std Match Ratio:    {np.std(match_ratios):.2%}")

    report_lines.append("")
    report_lines.append("=" * 80)

    return "\n".join(report_lines)


def main():
    """Main entry point for comparison script."""
    args = parse_args()

    extract_answer_fn, parse_ground_truth_fn, math_equal_fn, helper_error = _load_eval_helpers(
        args.strict_math_eval
    )
    helper_status = "strict parser+grader"
    if helper_error is not None:
        helper_status = f"fallback simple extractor ({helper_error})"

    print("Loading results...")
    hf_results = load_results(args.hf_results)
    vllm_results = load_results(args.vllm_results)
    print(f"  HuggingFace: {len(hf_results)} questions")
    print(f"  vLLM:        {len(vllm_results)} questions")
    print(f"  Eval mode:   {helper_status}")

    print("\nGenerating comparison report...")
    report = generate_comparison_report(
        hf_results,
        vllm_results,
        args,
        helper_status,
        extract_answer_fn,
        parse_ground_truth_fn,
        math_equal_fn,
    )

    # Save report
    output_path = Path(args.output_report)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        f.write(report)

    # Also print to console
    print("\n" + report)
    print(f"\nReport saved to: {args.output_report}")


if __name__ == "__main__":
    main()
