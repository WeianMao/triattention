#!/usr/bin/env python3
"""
Quick test script for eval_dfs_state_query.py

Tests the evaluation pipeline without requiring a real LLM.
Uses mock responses to verify parsing and metric calculation.
"""

import json
import sys
from pathlib import Path

# Add parent dir to path
sys.path.insert(0, str(Path(__file__).parent))

from eval_dfs_state_query import (
    build_prompt,
    parse_json_response,
    evaluate_prediction
)


def test_prompt_building():
    """Test prompt generation."""
    print("Testing prompt building...")

    test_case = {
        "id": 0,
        "graph": {
            "nodes": [0, 1, 2, 3],
            "edges": [[0, 1], [1, 2], [2, 3]]
        },
        "start_node": 0,
        "steps": 2,
        "answer": {
            "current_node": 2,
            "stack": [0, 1, 2],
            "visited_nodes": [0, 1, 2]
        }
    }

    prompt = build_prompt(test_case)
    print(f"Generated prompt:\n{prompt[:200]}...")

    assert "DFS" in prompt
    assert "节点" in prompt
    assert str(test_case["steps"]) in prompt
    print("✓ Prompt building works\n")


def test_json_parsing():
    """Test JSON response parsing."""
    print("Testing JSON parsing...")

    # Test cases with various formats
    test_responses = [
        # Clean JSON
        '{"current_node": 5, "stack": [0, 1, 5], "visited_nodes": [0, 1, 2, 5]}',

        # JSON in markdown code block
        '''```json
{
  "current_node": 5,
  "stack": [0, 1, 5],
  "visited_nodes": [0, 1, 2, 5]
}
```''',

        # JSON with extra text
        '''Based on the DFS rules, after 10 steps:

{
  "current_node": 5,
  "stack": [0, 1, 5],
  "visited_nodes": [0, 1, 2, 5]
}

The current node is 5.''',

        # Markdown without json tag
        '''```
{
  "current_node": 5,
  "stack": [0, 1, 5],
  "visited_nodes": [0, 1, 2, 5]
}
```''',
    ]

    expected = {
        "current_node": 5,
        "stack": [0, 1, 5],
        "visited_nodes": [0, 1, 2, 5]
    }

    for i, response in enumerate(test_responses):
        result = parse_json_response(response)
        assert result is not None, f"Failed to parse response {i}"
        assert result["current_node"] == expected["current_node"], f"Response {i}: node mismatch"
        assert result["stack"] == expected["stack"], f"Response {i}: stack mismatch"
        print(f"✓ Parsed format {i+1}")

    print("✓ All parsing tests passed\n")


def test_metric_calculation():
    """Test metric calculation."""
    print("Testing metric calculation...")

    ground_truth = {
        "current_node": 5,
        "stack": [0, 1, 2, 5],
        "visited_nodes": [0, 1, 2, 3, 4, 5]
    }

    # Perfect prediction
    prediction = {
        "current_node": 5,
        "stack": [0, 1, 2, 5],
        "visited_nodes": [0, 1, 2, 3, 4, 5]
    }

    metrics = evaluate_prediction(prediction, ground_truth)
    assert metrics["current_node_correct"] == True
    assert metrics["stack_exact_match"] == True
    assert metrics["visited_exact_match"] == True
    assert metrics["visited_f1"] == 1.0
    print("✓ Perfect prediction metrics correct")

    # Partial match
    prediction = {
        "current_node": 5,  # Correct
        "stack": [0, 1, 3, 5],  # Wrong path
        "visited_nodes": [0, 1, 2, 5]  # Missing some nodes
    }

    metrics = evaluate_prediction(prediction, ground_truth)
    assert metrics["current_node_correct"] == True
    assert metrics["stack_exact_match"] == False
    assert metrics["stack_top_correct"] == True  # Top is correct
    assert metrics["visited_exact_match"] == False
    assert 0 < metrics["visited_f1"] < 1
    print("✓ Partial match metrics correct")

    # Wrong prediction
    prediction = {
        "current_node": 3,
        "stack": [0, 1, 3],
        "visited_nodes": [0, 1, 3]
    }

    metrics = evaluate_prediction(prediction, ground_truth)
    assert metrics["current_node_correct"] == False
    assert metrics["stack_exact_match"] == False
    assert metrics["visited_exact_match"] == False
    print("✓ Wrong prediction metrics correct")

    print("✓ All metric tests passed\n")


def test_with_real_dataset():
    """Test with actual dataset file."""
    print("Testing with real dataset...")

    dataset_path = Path(__file__).resolve().parents[1] / "datasets" / "dfs_state_query_small.json"

    if not dataset_path.exists():
        print(f"⚠ Dataset not found at {dataset_path}, skipping")
        return

    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded {len(dataset)} samples")

    # Test first sample
    test_case = dataset[0]
    prompt = build_prompt(test_case)

    print(f"Sample {test_case['id']}:")
    print(f"  Nodes: {test_case['metadata']['graph_nodes']}")
    print(f"  Steps: {test_case['steps']}")
    print(f"  Expected current_node: {test_case['answer']['current_node']}")
    print(f"  Prompt length: {len(prompt)} chars")

    # Simulate perfect response
    mock_response = json.dumps(test_case["answer"])
    parsed = parse_json_response(mock_response)
    assert parsed is not None

    metrics = evaluate_prediction(parsed, test_case["answer"])
    assert all(v == True or v == 1.0 for v in metrics.values())

    print("✓ Real dataset test passed\n")


def main():
    print("="*80)
    print("DFS State Query Evaluation Script - Unit Tests")
    print("="*80 + "\n")

    try:
        test_prompt_building()
        test_json_parsing()
        test_metric_calculation()
        test_with_real_dataset()

        print("="*80)
        print("All tests passed! ✓")
        print("="*80)
        print("\nThe evaluation script is ready to use.")
        print("\nNext steps:")
        print("1. Prepare your model")
        print("2. Run: python dfs_state_query/scripts/eval_dfs_state_query.py "
              "--dataset dfs_state_query/datasets/dfs_state_query_small.json "
              "--model-path <your-model> --output dfs_state_query/analysis/test.json")

    except AssertionError as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
