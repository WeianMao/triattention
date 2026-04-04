#!/usr/bin/env python3
"""
Quick Test Script for AQA-Bench DFS State Query

A minimal script to test the full pipeline:
1. Load dataset
2. Build prompt
3. Call LLM
4. Parse response
5. Evaluate metrics

Usage:
    python dfs_state_query/scripts/quick_test.py \
        --model-path Qwen/Qwen2.5-7B-Instruct \
        --dataset dfs_state_query/datasets/dfs_state_query_small.json \
        --num-samples 3

Environment:
    Requires: transformers, torch, tqdm
"""

import json
import argparse
from pathlib import Path
from typing import Dict, Optional
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from prompt_utils import build_prompt


class SimpleLLM:
    """Minimal LLM wrapper for quick testing."""

    def __init__(self, model_path: str):
        print(f"Loading model: {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )
        print("Model loaded!")

    def generate(self, prompt: str, max_new_tokens: int = 32768) -> str:
        """Generate response."""
        messages = [{"role": "user", "content": prompt}]

        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False
            )

        response = self.tokenizer.decode(
            output[0, input_ids.shape[1]:],
            skip_special_tokens=True
        )

        return response


def parse_response(response: str) -> Optional[Dict]:
    """Parse JSON from response."""
    import re

    # Try direct parse
    try:
        return json.loads(response.strip())
    except Exception:
        pass

    # Try markdown code block
    patterns = [
        r'```json\s*(\{.*?\})\s*```',
        r'```\s*(\{.*?\})\s*```',
        r'(\{[^{}]*"current_node"[^{}]*\})',
    ]

    for pattern in patterns:
        match = re.search(pattern, response, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except Exception:
                continue

    # Find any JSON-like structure
    try:
        start = response.find('{')
        end = response.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(response[start:end])
    except Exception:
        pass

    return None


def evaluate(prediction: Dict, ground_truth: Dict) -> Dict:
    """Calculate metrics."""
    metrics = {}

    # Current node
    metrics["current_node_correct"] = (
        prediction.get("current_node") == ground_truth["current_node"]
    )

    # Stack
    pred_stack = prediction.get("stack", [])
    true_stack = ground_truth["stack"]
    metrics["stack_exact_match"] = (pred_stack == true_stack)

    # Visited nodes
    pred_visited = set(prediction.get("visited_nodes", []))
    true_visited = set(ground_truth["visited_nodes"])
    metrics["visited_exact_match"] = (pred_visited == true_visited)

    # F1 for visited
    if pred_visited and true_visited:
        precision = len(pred_visited & true_visited) / len(pred_visited)
        recall = len(pred_visited & true_visited) / len(true_visited)
        f1 = 2 * precision * recall / (precision + recall) if precision + recall > 0 else 0
        metrics["visited_f1"] = f1
    else:
        metrics["visited_f1"] = 0.0

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Quick test for DFS State Query")
    parser.add_argument("--model-path", type=str, required=True,
                       help="HuggingFace model path")
    parser.add_argument("--dataset", type=str,
                       default="dfs_state_query/datasets/dfs_state_query_small.json",
                       help="Dataset path")
    parser.add_argument("--num-samples", type=int, default=3,
                       help="Number of samples to test")
    parser.add_argument("--output", type=str, default=None,
                       help="Optional output JSON path")

    args = parser.parse_args()

    print("="*80)
    print("AQA-Bench Quick Test - DFS State Query")
    print("="*80)

    # Load dataset
    print(f"\nLoading dataset: {args.dataset}")
    with open(args.dataset, 'r') as f:
        dataset = json.load(f)

    dataset = dataset[:args.num_samples]
    print(f"Testing {len(dataset)} samples")

    # Load model
    model = SimpleLLM(args.model_path)

    # Run tests
    results = []
    correct_node = 0
    correct_stack = 0
    correct_visited = 0

    for i, test_case in enumerate(dataset):
        print(f"\n{'='*80}")
        print(f"Sample {i+1}/{len(dataset)} (ID: {test_case['id']})")
        print(f"{'='*80}")

        # Build prompt
        prompt = build_prompt(test_case)
        print(f"Graph: {test_case['metadata']['graph_nodes']} nodes, {test_case['metadata']['graph_edges']} edges")
        print(f"Steps to simulate: {test_case['steps']}")

        # Generate
        print("\nGenerating response...")
        response = model.generate(prompt)
        print(f"Raw response:\n{response}")

        # Parse
        prediction = parse_response(response)

        if prediction is None:
            print("\n❌ Failed to parse response")
            results.append({
                "id": test_case["id"],
                "parse_error": True,
                "response": response
            })
            continue

        # Evaluate
        ground_truth = test_case["answer"]
        metrics = evaluate(prediction, ground_truth)

        print(f"\n📊 Results:")
        print(f"  Prediction:    {prediction}")
        print(f"  Ground Truth:  {ground_truth}")
        print(f"\n  Metrics:")
        print(f"    Current Node:  {'✅' if metrics['current_node_correct'] else '❌'}")
        print(f"    Stack Match:   {'✅' if metrics['stack_exact_match'] else '❌'}")
        print(f"    Visited Match: {'✅' if metrics['visited_exact_match'] else '❌'}")
        print(f"    Visited F1:    {metrics['visited_f1']:.2%}")

        if metrics['current_node_correct']:
            correct_node += 1
        if metrics['stack_exact_match']:
            correct_stack += 1
        if metrics['visited_exact_match']:
            correct_visited += 1

        results.append({
            "id": test_case["id"],
            "prediction": prediction,
            "ground_truth": ground_truth,
            "metrics": metrics,
            "response": response
        })

    # Summary
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    n = len(dataset)
    print(f"Current Node Accuracy:  {correct_node}/{n} ({correct_node/n:.1%})")
    print(f"Stack Exact Match:      {correct_stack}/{n} ({correct_stack/n:.1%})")
    print(f"Visited Exact Match:    {correct_visited}/{n} ({correct_visited/n:.1%})")

    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            json.dump({
                "model": args.model_path,
                "num_samples": len(dataset),
                "accuracy": {
                    "current_node": correct_node / n,
                    "stack": correct_stack / n,
                    "visited": correct_visited / n
                },
                "results": results
            }, f, indent=2, ensure_ascii=False)

        print(f"\nResults saved to: {output_path}")

    print("\n✅ Test complete!")


if __name__ == "__main__":
    main()
