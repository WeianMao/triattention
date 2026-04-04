#!/usr/bin/env python3
"""
DFS State Query Evaluation Script (English CoT Version)

Evaluates LLM's ability to simulate DFS execution with step-by-step reasoning.
Uses English prompts and requires chain-of-thought explanation.

Usage:
    python dfs_state_query/scripts/eval_dfs_cot_english.py \
        --dataset dfs_state_query/datasets/legacy/dfs_state_query_100.json \
        --model-path /path/to/model \
        --output dfs_state_query/analysis/dfs_cot_eval.json \
        --max-samples 10
"""

import json
import argparse
import re
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, GenerationConfig


class LLMModel:
    """Simple LLM wrapper for DFS state query evaluation."""

    def __init__(self, model_path: str, max_new_tokens: int = 1024):
        self.model_path = model_path
        self.max_new_tokens = max_new_tokens

        print(f"Loading model from {model_path}...")
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True
        )

        # Setup generation config
        try:
            self.model.generation_config = GenerationConfig.from_pretrained(model_path)
        except Exception:
            self.model.generation_config = GenerationConfig()

        eos_token_id = self.model.generation_config.eos_token_id
        if isinstance(eos_token_id, list) and eos_token_id:
            eos_token_id = eos_token_id[0]

        pad_token_id = self.model.generation_config.pad_token_id
        if isinstance(pad_token_id, list):
            pad_token_id = pad_token_id[0] if pad_token_id else None

        if pad_token_id is None:
            pad_token_id = eos_token_id

        self.model.generation_config.pad_token_id = pad_token_id
        if self.tokenizer.pad_token_id is None and pad_token_id is not None:
            self.tokenizer.pad_token_id = pad_token_id
        print("Model loaded successfully!")

    def generate(self, prompt: str, max_new_tokens: Optional[int] = None) -> str:
        """Generate response for a single prompt."""
        if max_new_tokens is None:
            max_new_tokens = self.max_new_tokens

        # Format as chat message
        messages = [{"role": "user", "content": prompt}]

        # Tokenize
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            return_tensors="pt"
        ).to(self.model.device)

        # Generate
        with torch.no_grad():
            output = self.model.generate(
                input_ids,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=None,
                top_p=None
            )

        # Decode only the generated part
        output = output[:, input_ids.shape[1]:]
        response = self.tokenizer.batch_decode(output, skip_special_tokens=True)[0]

        return response


def build_prompt_cot(test_case: Dict) -> str:
    """
    Build evaluation prompt with chain-of-thought requirement.

    Uses English and explicitly asks for step-by-step reasoning.
    """
    graph = test_case["graph"]
    start_node = test_case["start_node"]
    steps = test_case["steps"]

    nodes = graph["nodes"]
    edges = graph["edges"]

    prompt = f"""Given the following undirected graph:
Nodes: {nodes}
Edges: {edges}

Starting from node {start_node}, perform a Depth-First Search (DFS).

DFS Rules:
1. Visit unvisited neighbors first (choose the smallest numbered neighbor)
2. If all neighbors are visited, backtrack to the parent node
3. The stack stores the complete path from the start node to the current node

Question: After executing {steps} steps of DFS:
1. What is the current node?
2. What is the current DFS stack state (path from start to current node)?
3. What are all the visited nodes?

Please solve this step-by-step:
1. First, analyze the graph structure and identify neighbors for each node
2. Then, simulate each DFS step one by one, explaining your choices
3. Track the current node, stack, and visited set at each step
4. Finally, provide the answer after step {steps}

Format your response as follows:
- Start with "Analysis:" and explain your reasoning step-by-step
- End with "Answer:" followed by a JSON object (no extra text after JSON)

Answer JSON format:
{{
  "current_node": <node_number>,
  "stack": [<path_nodes_list>],
  "visited_nodes": [<visited_nodes_list_sorted>]
}}"""

    return prompt


def parse_cot_response(response: str) -> Tuple[Optional[str], Optional[Dict]]:
    """
    Parse chain-of-thought response.

    Returns:
        (reasoning, prediction) where reasoning is the CoT explanation
        and prediction is the parsed JSON answer.
    """
    # Try to split into reasoning and answer parts
    reasoning = None
    answer_json = None

    # Look for "Answer:" section
    answer_match = re.search(r'Answer:\s*(.*)', response, re.DOTALL | re.IGNORECASE)
    if answer_match:
        answer_text = answer_match.group(1).strip()
        reasoning = response[:answer_match.start()].strip()

        # Try to parse JSON from answer section
        answer_json = parse_json_from_text(answer_text)

    # If no explicit "Answer:" section, try to find JSON anywhere
    if answer_json is None:
        answer_json = parse_json_from_text(response)
        if answer_json and reasoning is None:
            # Use everything before JSON as reasoning
            json_str = json.dumps(answer_json)
            json_pos = response.find(json_str)
            if json_pos > 0:
                reasoning = response[:json_pos].strip()

    return reasoning, answer_json


def parse_json_from_text(text: str) -> Optional[Dict]:
    """
    Parse JSON from text, trying multiple patterns.
    """
    # Try direct JSON parse first
    try:
        return json.loads(text.strip())
    except json.JSONDecodeError:
        pass

    # Try to find JSON block in text
    json_patterns = [
        r'```json\s*(\{.*?\})\s*```',  # Markdown code block
        r'```\s*(\{.*?\})\s*```',       # Code block without json tag
        r'(\{[^{}]*"current_node"[^{}]*\})',  # Simple JSON with current_node
    ]

    for pattern in json_patterns:
        match = re.search(pattern, text, re.DOTALL)
        if match:
            try:
                return json.loads(match.group(1))
            except json.JSONDecodeError:
                continue

    # Last resort: try to find any JSON-like structure
    try:
        start = text.find('{')
        end = text.rfind('}') + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception:
        pass

    return None


def evaluate_prediction(prediction: Dict, ground_truth: Dict) -> Dict:
    """
    Evaluate prediction against ground truth.

    Returns dict with various accuracy metrics.
    """
    metrics = {}

    # Current node accuracy
    pred_node = prediction.get("current_node")
    true_node = ground_truth["current_node"]
    metrics["current_node_correct"] = (pred_node == true_node)

    # Stack evaluation
    pred_stack = prediction.get("stack", [])
    true_stack = ground_truth["stack"]

    # Exact stack match
    metrics["stack_exact_match"] = (pred_stack == true_stack)

    # Stack depth accuracy
    metrics["stack_depth_correct"] = (len(pred_stack) == len(true_stack))

    # Stack top correct (if both non-empty)
    if pred_stack and true_stack:
        metrics["stack_top_correct"] = (pred_stack[-1] == true_stack[-1])
    else:
        metrics["stack_top_correct"] = (len(pred_stack) == len(true_stack) == 0)

    # Stack intersection ratio
    if true_stack:
        pred_set = set(pred_stack)
        true_set = set(true_stack)
        metrics["stack_intersection_ratio"] = len(pred_set & true_set) / len(true_set)
    else:
        metrics["stack_intersection_ratio"] = 1.0 if not pred_stack else 0.0

    # Visited nodes evaluation
    pred_visited = set(prediction.get("visited_nodes", []))
    true_visited = set(ground_truth["visited_nodes"])

    # Exact match
    metrics["visited_exact_match"] = (pred_visited == true_visited)

    # Precision, Recall, F1
    if pred_visited:
        precision = len(pred_visited & true_visited) / len(pred_visited)
    else:
        precision = 1.0 if not true_visited else 0.0

    if true_visited:
        recall = len(pred_visited & true_visited) / len(true_visited)
    else:
        recall = 1.0 if not pred_visited else 0.0

    if precision + recall > 0:
        f1 = 2 * precision * recall / (precision + recall)
    else:
        f1 = 0.0

    metrics["visited_precision"] = precision
    metrics["visited_recall"] = recall
    metrics["visited_f1"] = f1

    return metrics


def run_evaluation(
    dataset_path: str,
    model_path: str,
    output_path: str,
    max_samples: Optional[int] = None,
    max_new_tokens: int = 1024,
    verbose: bool = True
):
    """Run full evaluation on dataset with CoT prompts."""

    # Load dataset
    print(f"Loading dataset from {dataset_path}...")
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    if max_samples:
        dataset = dataset[:max_samples]

    print(f"Loaded {len(dataset)} samples")

    # Load model
    model = LLMModel(model_path, max_new_tokens=max_new_tokens)

    # Run evaluation
    results = []
    all_metrics = []

    for test_case in tqdm(dataset, desc="Evaluating"):
        case_id = test_case["id"]

        # Build prompt with CoT
        prompt = build_prompt_cot(test_case)

        # Generate response
        try:
            response = model.generate(prompt)
        except Exception as e:
            print(f"\nError generating response for case {case_id}: {e}")
            results.append({
                "id": case_id,
                "error": str(e),
                "prompt": prompt,
            })
            continue

        # Parse CoT response
        reasoning, prediction = parse_cot_response(response)

        if prediction is None:
            if verbose:
                print(f"\nFailed to parse response for case {case_id}")
                print(f"Response: {response[:200]}...")

            results.append({
                "id": case_id,
                "prompt": prompt,
                "response": response,
                "reasoning": reasoning,
                "prediction": None,
                "ground_truth": test_case["answer"],
                "parse_error": True
            })
            continue

        # Evaluate
        metrics = evaluate_prediction(prediction, test_case["answer"])
        all_metrics.append(metrics)

        # Store result
        result = {
            "id": case_id,
            "prompt": prompt,
            "response": response,
            "reasoning": reasoning,
            "prediction": prediction,
            "ground_truth": test_case["answer"],
            "metrics": metrics,
            "metadata": test_case.get("metadata", {})
        }
        results.append(result)

        if verbose and case_id < 3:
            print(f"\n--- Sample {case_id} ---")
            print(f"Reasoning length: {len(reasoning) if reasoning else 0} chars")
            print(f"Prediction: {prediction}")
            print(f"Ground truth: {test_case['answer']}")
            print(f"Metrics: {metrics}")

    # Compute aggregate metrics
    if all_metrics:
        aggregate = {}
        for key in all_metrics[0].keys():
            values = [m[key] for m in all_metrics]
            aggregate[key] = sum(values) / len(values)

        print("\n" + "="*80)
        print("AGGREGATE METRICS")
        print("="*80)
        for key, value in aggregate.items():
            print(f"{key:30s}: {value:.4f}")

        # Add to results
        results_dict = {
            "config": {
                "prompt_type": "chain_of_thought",
                "language": "english",
                "max_new_tokens": max_new_tokens
            },
            "aggregate_metrics": aggregate,
            "num_samples": len(dataset),
            "num_evaluated": len(all_metrics),
            "num_parse_errors": len(dataset) - len(all_metrics),
            "results": results
        }
    else:
        print("\nNo valid results to aggregate!")
        results_dict = {
            "config": {
                "prompt_type": "chain_of_thought",
                "language": "english",
                "max_new_tokens": max_new_tokens
            },
            "aggregate_metrics": {},
            "num_samples": len(dataset),
            "num_evaluated": 0,
            "num_parse_errors": len(dataset),
            "results": results
        }

    # Save results
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results_dict, f, indent=2)

    print(f"\nResults saved to {output_path}")

    return results_dict


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate DFS state query with chain-of-thought reasoning (English)"
    )
    parser.add_argument("--dataset", type=str, required=True,
                        help="Path to DFS state query dataset JSON")
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to HuggingFace model")
    parser.add_argument("--output", type=str, required=True,
                        help="Output path for results JSON")
    parser.add_argument("--max-samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for testing)")
    parser.add_argument("--max-new-tokens", type=int, default=1024,
                        help="Maximum new tokens to generate (increased for CoT)")
    parser.add_argument("--verbose", action="store_true",
                        help="Print verbose output")

    args = parser.parse_args()

    run_evaluation(
        dataset_path=args.dataset,
        model_path=args.model_path,
        output_path=args.output,
        max_samples=args.max_samples,
        max_new_tokens=args.max_new_tokens,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main()
