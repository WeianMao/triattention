#!/usr/bin/env python3
"""Check output format compatibility between vLLM and HF implementations.

This script compares the JSONL output format to ensure compare_results.py
will work correctly.
"""
import json
import argparse
from pathlib import Path


def check_format(file_path: str, backend_name: str) -> dict:
    """Check the format of a results file.

    Args:
        file_path: Path to JSONL results file
        backend_name: Name of backend (for display)

    Returns:
        Dictionary with format info
    """
    print(f"\n{'='*80}")
    print(f"Checking {backend_name} format: {file_path}")
    print(f"{'='*80}")

    with open(file_path, 'r') as f:
        lines = [json.loads(line.strip()) for line in f if line.strip()]

    if not lines:
        print("ERROR: File is empty!")
        return {}

    # Check first entry
    first_entry = lines[0]

    print(f"\nTotal entries: {len(lines)}")
    print(f"\nFirst entry keys:")
    for key in sorted(first_entry.keys()):
        value = first_entry[key]
        if isinstance(value, str):
            preview = value[:50] + "..." if len(value) > 50 else value
            print(f"  {key}: (str, len={len(value)}) {preview}")
        elif isinstance(value, list):
            print(f"  {key}: (list, len={len(value)})")
            if value and isinstance(value[0], str):
                preview = value[0][:50] + "..." if len(value[0]) > 50 else value[0]
                print(f"    First item: {preview}")
        else:
            print(f"  {key}: {value}")

    # Check required fields for compare_results.py
    required_fields = ["question", "generated_answers"]
    optional_fields = ["id", "ground_truth", "answer"]

    missing_required = [f for f in required_fields if f not in first_entry]
    available_optional = [f for f in optional_fields if f in first_entry]

    print(f"\nCompatibility check:")
    if missing_required:
        print(f"  ✗ Missing required fields: {missing_required}")
    else:
        print(f"  ✓ All required fields present")

    print(f"  Available optional fields: {available_optional}")

    # Check answer field naming
    answer_field = None
    for field in ["ground_truth", "answer"]:
        if field in first_entry:
            answer_field = field
            break

    if answer_field:
        print(f"  ✓ Answer field: '{answer_field}'")
    else:
        print(f"  ✗ No answer field found (ground_truth or answer)")

    # Check generated_answers format
    if "generated_answers" in first_entry:
        gen_answers = first_entry["generated_answers"]
        if isinstance(gen_answers, list):
            print(f"  ✓ generated_answers is list with {len(gen_answers)} items")
        else:
            print(f"  ✗ generated_answers should be list, got {type(gen_answers)}")

    # Check output field (HF specific)
    if "output" in first_entry:
        print(f"  Note: HF format uses 'output' field (single string)")
        print(f"        vLLM format uses 'generated_answers' (list)")

    return {
        "num_entries": len(lines),
        "keys": list(first_entry.keys()),
        "answer_field": answer_field,
        "has_generated_answers": "generated_answers" in first_entry,
        "has_output": "output" in first_entry,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Check output format compatibility"
    )
    parser.add_argument(
        "--vllm-results",
        type=str,
        required=True,
        help="Path to vLLM results JSONL"
    )
    parser.add_argument(
        "--hf-results",
        type=str,
        default=None,
        help="Path to HF results JSONL (optional)"
    )
    args = parser.parse_args()

    # Check vLLM format
    vllm_info = check_format(args.vllm_results, "vLLM")

    # Check HF format if provided
    hf_info = None
    if args.hf_results:
        hf_info = check_format(args.hf_results, "HuggingFace")

    # Compare formats
    if hf_info:
        print(f"\n{'='*80}")
        print("Format Comparison")
        print(f"{'='*80}")

        # Check field compatibility
        vllm_keys = set(vllm_info["keys"])
        hf_keys = set(hf_info["keys"])

        common_keys = vllm_keys & hf_keys
        vllm_only = vllm_keys - hf_keys
        hf_only = hf_keys - vllm_keys

        print(f"\nCommon fields: {sorted(common_keys)}")
        print(f"vLLM only: {sorted(vllm_only)}")
        print(f"HF only: {sorted(hf_only)}")

        # Check for conversion needed
        if hf_info["has_output"] and not vllm_info["has_generated_answers"]:
            print("\n⚠ WARNING: HF uses 'output', vLLM uses 'generated_answers'")
            print("  compare_results.py may need format conversion")
        elif vllm_info["has_generated_answers"] and hf_info["has_output"]:
            print("\n⚠ NOTE: Format difference detected:")
            print("  - HF format: 'output' (single string per draw)")
            print("  - vLLM format: 'generated_answers' (list of strings)")
            print("  compare_results.py should handle both formats")
        else:
            print("\n✓ Formats appear compatible")

    print(f"\n{'='*80}\n")


if __name__ == "__main__":
    main()
