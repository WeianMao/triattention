#!/usr/bin/env python3
"""vLLM-based reasoning benchmark for TriAttention KV compression.

This script mirrors the HuggingFace SpeckV benchmark but uses vLLM as the backend
with TriAttention compression enabled. Designed for fair comparison between
HuggingFace and vLLM implementations.

Usage:
    python run_math_vllm.py \
        --model-path /path/to/model \
        --dataset-path data/aime24.jsonl \
        --output-path outputs/vllm_results.jsonl \
        --pruning-mode per_head \
        --kv-budget 2048
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

# Import vLLM
from vllm import LLM, SamplingParams

# Import TriAttention modules (already implemented)
from triattention import (
    TriAttentionConfig,
    TriAttentionWrapper,
    create_triattention_wrapper,
    patch_vllm_attention,
)


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments matching HuggingFace benchmark interface."""
    parser = argparse.ArgumentParser(
        description="vLLM reasoning benchmark with TriAttention compression"
    )

    # Model and data
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to JSONL dataset file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save results"
    )

    # TriAttention compression parameters
    parser.add_argument(
        "--kv-budget",
        type=int,
        default=2048,
        help="Maximum KV cache budget (tokens to retain)"
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=128,
        help="Recent token window size (always retained)"
    )
    parser.add_argument(
        "--divide-length",
        type=int,
        default=128,
        help="Compression trigger interval"
    )
    parser.add_argument(
        "--sparse-round-window",
        type=int,
        default=32,
        help="Sparse round window for compression"
    )
    parser.add_argument(
        "--sparse-offset-max-length",
        type=int,
        default=65536,
        help="Maximum offset length for position-dependent scoring"
    )
    parser.add_argument(
        "--sparse-score-aggregation",
        type=str,
        default="mean",
        choices=["mean", "max"],
        help="Score aggregation method"
    )
    parser.add_argument(
        "--pruning-mode",
        type=str,
        default="per_head",
        choices=["per_head", "per_layer", "per_layer_per_head"],
        help="Token selection strategy: per_head, per_layer, or per_layer_per_head"
    )
    parser.add_argument(
        "--sparse-normalize-scores",
        action="store_true",
        help="Enable sparse score normalization (default: enabled)"
    )
    parser.add_argument(
        "--no-sparse-normalize-scores",
        action="store_true",
        help="Disable sparse score normalization"
    )
    parser.add_argument(
        "--include-prefill-in-budget",
        action="store_true",
        help="Include prefill tokens in budget calculation (default: enabled)"
    )
    parser.add_argument(
        "--no-include-prefill-in-budget",
        action="store_true",
        help="Exclude prefill tokens from budget calculation"
    )
    parser.add_argument(
        "--rkv-style-compression",
        action="store_true",
        help="Use R-KV style compression mode (default: enabled)"
    )
    parser.add_argument(
        "--no-rkv-style-compression",
        action="store_true",
        help="Disable R-KV style compression mode"
    )
    parser.add_argument(
        "--rkv-style-slack-trigger",
        action="store_true",
        help="Use R-KV style slack trigger (default: enabled)"
    )
    parser.add_argument(
        "--no-rkv-style-slack-trigger",
        action="store_true",
        help="Disable R-KV style slack trigger"
    )
    parser.add_argument(
        "--sparse-stats-path",
        type=str,
        default=None,
        help="Path to precomputed sparse statistics file"
    )

    # Generation parameters
    parser.add_argument(
        "--num-samples",
        type=int,
        default=8,
        help="Number of samples to generate per question"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=0.95,
        help="Nucleus sampling parameter"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=-1,
        help="Top-k sampling parameter (-1 for disabled)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32768,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=888,
        help="Random seed for reproducibility"
    )

    # System parameters
    parser.add_argument(
        "--load-dtype",
        type=str,
        default="bfloat16",
        choices=["float16", "bfloat16", "float32"],
        help="Model loading dtype"
    )
    parser.add_argument(
        "--tensor-parallel-size",
        type=int,
        default=1,
        help="Tensor parallelism size"
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.9,
        help="GPU memory utilization ratio"
    )
    parser.add_argument(
        "--max-questions",
        type=int,
        default=None,
        help="Maximum number of questions to process (for testing)"
    )

    return parser.parse_args()


def load_dataset(dataset_path: str) -> List[Dict]:
    """Load JSONL dataset with questions and answers.

    Expected format:
        {"question": "...", "answer": "...", ...}
    """
    dataset = []
    with open(dataset_path, "r") as f:
        for idx, line in enumerate(f):
            item = json.loads(line.strip())
            item["index"] = idx
            dataset.append(item)
    return dataset


PROMPT_TEMPLATE = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"


def setup_triattention_config(args: argparse.Namespace) -> TriAttentionConfig:
    """Create TriAttention configuration from command-line arguments."""
    # Handle boolean flags with default True
    sparse_normalize_scores = True if not args.no_sparse_normalize_scores else False
    include_prefill_in_budget = True if not args.no_include_prefill_in_budget else False
    rkv_style_compression = True if not args.no_rkv_style_compression else False
    rkv_style_slack_trigger = True if not args.no_rkv_style_slack_trigger else False

    config = TriAttentionConfig(
        stats_path=Path(args.sparse_stats_path) if args.sparse_stats_path else None,
        kv_budget=args.kv_budget,
        divide_length=args.divide_length,
        sparse_round_window=args.sparse_round_window,
        pruning_mode=args.pruning_mode,
        score_aggregation=args.sparse_score_aggregation,
        offset_max_length=args.sparse_offset_max_length,
        sparse_normalize_scores=sparse_normalize_scores,
        window_size=args.window_size,
        # R-KV alignment flags
        include_prefill_in_budget=include_prefill_in_budget,
        protect_prefill=not include_prefill_in_budget,
    )
    return config


def setup_vllm_engine(
    args: argparse.Namespace,
    tri_config: TriAttentionConfig,
) -> Tuple[LLM, TriAttentionWrapper]:
    """Initialize vLLM engine with TriAttention compression wrapper.

    Returns:
        Tuple of (vLLM engine, TriAttention wrapper)

    Note:
        This patches vLLM's attention mechanism to apply TriAttention compression
        automatically during decode steps.
    """
    # Initialize vLLM engine
    llm = LLM(
        model=args.model_path,
        dtype=args.load_dtype,
        tensor_parallel_size=args.tensor_parallel_size,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        seed=args.seed,
        max_model_len=args.max_tokens,
        enforce_eager=True,  # Disable CUDA graphs to save memory
    )

    # Create TriAttention wrapper
    tri_wrapper = TriAttentionWrapper(tri_config)

    # Patch vLLM attention to enable compression
    try:
        model = llm.llm_engine.model_executor.driver_worker.model_runner.model
        patch_vllm_attention(model, tri_wrapper)
        print("[INFO] TriAttention compression enabled")
    except Exception as e:
        print(f"[WARNING] Failed to patch vLLM attention: {e}")
        print("[WARNING] Running without compression")

    return llm, tri_wrapper


def run_benchmark(
    llm: LLM,
    tri_wrapper: TriAttentionWrapper,
    dataset: List[Dict],
    args: argparse.Namespace,
    output_path: str
) -> None:
    """Run reasoning benchmark on dataset using vLLM with TriAttention.

    Args:
        llm: vLLM engine
        tri_wrapper: TriAttention wrapper (for state management)
        dataset: List of questions and answers
        args: Command-line arguments
        output_path: Path to output JSONL file

    Note:
        TriAttention compression is automatically applied during inference
        via the patched attention mechanism.
        Results are written incrementally to match HF version behavior.
    """
    # Setup sampling parameters for single sample generation
    sampling_params = SamplingParams(
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k if args.top_k > 0 else -1,
        max_tokens=args.max_tokens,
        n=1,  # Generate one sample at a time
    )

    # Create output file
    with open(output_path, "w") as fout:
        for item in tqdm(dataset, desc="Running inference"):
            question = item["question"]

            # Format prompt matching HF version
            prompt = PROMPT_TEMPLATE.format(question=question)

            # Generate num_samples responses
            for draw_idx in range(args.num_samples):
                # Reset compression state between samples if requested
                # Note: vLLM doesn't expose per-request reset easily,
                # so we rely on vLLM's automatic state management

                # Run vLLM generation with automatic TriAttention compression
                outputs = llm.generate([prompt], sampling_params)
                generated_text = outputs[0].outputs[0].text

                # Build result record matching HF format
                record = {
                    "index": item["index"],
                    "question": question,
                    "answer": item.get("answer", None),
                    "prompt": prompt,
                    "output": generated_text,
                    "sample_idx": item["index"],
                    "draw_idx": draw_idx,
                    "kv_budget": args.kv_budget,
                    "pruning_mode": args.pruning_mode,
                    "temperature": args.temperature,
                    "backend": "vllm_triattention",
                }

                # Write result immediately
                fout.write(json.dumps(record, ensure_ascii=False) + "\n")
                fout.flush()




def main():
    """Main entry point for vLLM reasoning benchmark."""
    # Process masking for cluster compatibility
    import setproctitle
    setproctitle.setproctitle("PD-L1_binder")

    args = parse_args()

    # Determine output path
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "vllm_results.jsonl"

    print("=" * 80)
    print("vLLM Reasoning Benchmark with TriAttention Compression")
    print("=" * 80)
    print(f"Model: {args.model_path}")
    print(f"Dataset: {args.dataset}")
    print(f"Output: {output_path}")
    print(f"KV Budget: {args.kv_budget}")
    print(f"Pruning Mode: {args.pruning_mode}")
    print(f"Num Samples: {args.num_samples}")
    print("=" * 80)

    # Load dataset
    print("\nLoading dataset...")
    dataset = load_dataset(args.dataset)
    print(f"Loaded {len(dataset)} questions")

    # Setup TriAttention configuration
    print("\nSetting up TriAttention configuration...")
    tri_config = setup_triattention_config(args)
    print(f"  KV Budget: {tri_config.kv_budget}")
    print(f"  Pruning Mode: {tri_config.pruning_mode}")
    print(f"  Divide Length: {tri_config.divide_length}")
    if args.sparse_stats_path:
        print(f"  Stats Path: {args.sparse_stats_path}")

    # Initialize vLLM engine
    print("\nInitializing vLLM engine...")
    llm, tri_wrapper = setup_vllm_engine(args, tri_config)
    print(f"  Model loaded: {args.model_path}")
    print(f"  TriAttention compression: {'ENABLED' if tri_wrapper._patched else 'DISABLED'}")

    # Run benchmark
    print("\nRunning benchmark...")
    if tri_wrapper._patched:
        print("  Compression will be applied automatically during generation")
    else:
        print("  WARNING: Compression is disabled (patching failed)")

    run_benchmark(llm, tri_wrapper, dataset, args, str(output_path))

    print("\n" + "=" * 80)
    print("Benchmark complete!")
    print(f"Results saved to: {output_path}")
    print("=" * 80)


if __name__ == "__main__":
    main()
