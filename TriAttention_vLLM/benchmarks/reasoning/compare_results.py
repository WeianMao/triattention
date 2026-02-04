#!/usr/bin/env python3
"""Compare HuggingFace SpeckV and vLLM TriAttention benchmark results.

This script loads results from both implementations and performs detailed
comparison to identify any discrepancies in output quality, token generation,
or accuracy metrics.

Usage:
    python compare_results.py \
        --hf-results /path/to/hf_results.jsonl \
        --vllm-results /path/to/vllm_results.jsonl \
        --output-report comparison_report.txt
"""
import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare HuggingFace and vLLM reasoning benchmark results"
    )
    parser.add_argument(
        "--hf-results",
        type=str,
        required=True,
        help="Path to HuggingFace results JSONL"
    )
    parser.add_argument(
        "--vllm-results",
        type=str,
        required=True,
        help="Path to vLLM results JSONL"
    )
    parser.add_argument(
        "--output-report",
        type=str,
        default="comparison_report.txt",
        help="Path to save comparison report"
    )
    parser.add_argument(
        "--detailed",
        action="store_true",
        help="Generate detailed per-question comparison"
    )
    return parser.parse_args()


def load_results(results_path: str) -> List[Dict]:
    """Load results from JSONL file.

    Handles both formats:
    - HF format: "output" field (single string per entry, multiple entries per question)
    - vLLM format: "generated_answers" field (list of strings per entry)
    """
    results = []
    with open(results_path, "r") as f:
        for line in f:
            item = json.loads(line.strip())

            # Convert HF format to vLLM format for consistency
            if "output" in item and "generated_answers" not in item:
                # HF format: each line is one sample
                # Group by question to match vLLM format
                item["generated_answers"] = [item["output"]]

            # Handle answer field naming differences
            if "answer" in item and "ground_truth" not in item:
                item["ground_truth"] = item["answer"]

            results.append(item)

    # For HF format, we need to group multiple entries per question
    # This is handled by grouping on question text
    grouped = {}
    for item in results:
        key = item.get("question", "") or item.get("index", "")
        if key in grouped:
            # Merge generated_answers
            grouped[key]["generated_answers"].extend(item.get("generated_answers", []))
        else:
            grouped[key] = item

    return list(grouped.values())


def extract_answer(text: str) -> str:
    """Extract numerical answer from generated text.

    Looks for patterns like \\boxed{...} or final numbers.
    This is a simple implementation and may need refinement.
    """
    # Look for \\boxed{...} pattern
    import re
    boxed_match = re.search(r'\\boxed\{([^}]+)\}', text)
    if boxed_match:
        return boxed_match.group(1).strip()

    # Look for last number in text as fallback
    numbers = re.findall(r'\b\d+(?:\.\d+)?\b', text)
    if numbers:
        return numbers[-1]

    return ""


def calculate_accuracy(results: List[Dict]) -> Tuple[float, int, int]:
    """Calculate accuracy by comparing generated answers with ground truth.

    Returns:
        (accuracy, num_correct, num_total)
    """
    num_correct = 0
    num_total = 0

    for item in results:
        ground_truth = str(item.get("ground_truth", "")).strip()
        if not ground_truth:
            continue

        generated_answers = item.get("generated_answers", [])

        # Check if any generated answer matches ground truth
        for gen_text in generated_answers:
            extracted = extract_answer(gen_text)
            if extracted == ground_truth:
                num_correct += 1
                break

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
    args: argparse.Namespace
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
    report_lines.append("")

    # Calculate accuracies
    hf_acc, hf_correct, hf_total = calculate_accuracy(hf_results)
    vllm_acc, vllm_correct, vllm_total = calculate_accuracy(vllm_results)

    report_lines.append("Accuracy Metrics:")
    report_lines.append(f"  HuggingFace: {hf_acc:.2%} ({hf_correct}/{hf_total})")
    report_lines.append(f"  vLLM:        {vllm_acc:.2%} ({vllm_correct}/{vllm_total})")
    report_lines.append(f"  Difference:  {abs(hf_acc - vllm_acc):.2%}")
    report_lines.append("")

    # Align results by question ID
    hf_by_id = {item.get("id"): item for item in hf_results}
    vllm_by_id = {item.get("id"): item for item in vllm_results}
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
                hf_text = hf_item.get("generated_answers", [""])[0]
                vllm_text = vllm_item.get("generated_answers", [""])[0]

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

    print("Loading results...")
    hf_results = load_results(args.hf_results)
    vllm_results = load_results(args.vllm_results)
    print(f"  HuggingFace: {len(hf_results)} questions")
    print(f"  vLLM:        {len(vllm_results)} questions")

    print("\nGenerating comparison report...")
    report = generate_comparison_report(hf_results, vllm_results, args)

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
