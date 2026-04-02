#!/usr/bin/env python3
"""
DFS Dataset Verification Script

Verifies the correctness of generated DFS state query datasets.

Usage:
    python dfs_state_query/scripts/verify_dfs_dataset.py \
        dfs_state_query/datasets/dfs_state_query_small.json --sample-id 0
    python dfs_state_query/scripts/verify_dfs_dataset.py \
        dfs_state_query/datasets/dfs_state_query_small.json --verify-all
"""

import json
import argparse
from pathlib import Path
from typing import List, Dict

import networkx as nx


def simulate_dfs_steps(graph: nx.Graph, start_node: int, num_steps: int) -> Dict:
    """
    Simulate DFS for num_steps and return the state.

    Returns:
        Dictionary with current_node, stack, visited_nodes
    """
    stack = [start_node]
    visited = {start_node}

    for step in range(num_steps):
        if not stack:
            break

        current = stack[-1]
        neighbors = sorted(list(graph.neighbors(current)))
        unvisited = [n for n in neighbors if n not in visited]

        if unvisited:
            next_node = unvisited[0]
            stack.append(next_node)
            visited.add(next_node)
        else:
            stack.pop()

    return {
        "current_node": stack[-1] if stack else None,
        "stack": stack.copy(),
        "visited_nodes": sorted(list(visited))
    }


def check_validity(test_case: Dict) -> Dict:
    """
    Check if test case is valid (not confusing for LLMs).

    Returns:
        Dictionary with validity status and reasons
    """
    issues = []

    # Rule 1: steps should not exceed total_dfs_steps
    steps = test_case["steps"]
    total_steps = test_case["metadata"]["total_dfs_steps"]

    if steps > total_steps:
        issues.append(f"steps ({steps}) > total_dfs_steps ({total_steps})")

    # Rule 2: stack should not be empty at target step
    if test_case["answer"]["current_node"] is None:
        issues.append("current_node is None (stack is empty)")

    return {
        "valid": len(issues) == 0,
        "issues": issues
    }


def verify_test_case(test_case: Dict, verbose: bool = False, check_valid: bool = True) -> Dict:
    """
    Verify a single test case.

    Returns:
        Dictionary with verification results
    """
    result = {
        "id": test_case["id"],
        "valid": True,
        "correct": True,
        "issues": []
    }

    # Check validity first
    if check_valid:
        validity = check_validity(test_case)
        result["valid"] = validity["valid"]
        result["issues"] = validity["issues"]

        if not validity["valid"]:
            if verbose:
                print(f"\n{'='*80}")
                print(f"Test Case ID: {test_case['id']} - INVALID")
                print(f"{'='*80}")
                print(f"Issues:")
                for issue in validity["issues"]:
                    print(f"  ✗ {issue}")
                print(f"{'='*80}\n")
            return result

    # Build graph
    graph = nx.Graph()
    graph.add_nodes_from(test_case["graph"]["nodes"])
    graph.add_edges_from(test_case["graph"]["edges"])

    # Simulate DFS
    start_node = test_case["start_node"]
    steps = test_case["steps"]

    simulated = simulate_dfs_steps(graph, start_node, steps)
    expected = test_case["answer"]

    # Compare results
    node_match = simulated["current_node"] == expected["current_node"]
    stack_match = simulated["stack"] == expected["stack"]
    visited_match = simulated["visited_nodes"] == expected["visited_nodes"]

    is_correct = node_match and stack_match and visited_match
    result["correct"] = is_correct

    if verbose or not is_correct:
        print(f"\n{'='*80}")
        print(f"Test Case ID: {test_case['id']}")
        print(f"{'='*80}")
        print(f"Graph: {test_case['metadata']['graph_nodes']} nodes, "
              f"{test_case['metadata']['graph_edges']} edges")
        print(f"Steps: {steps} / {test_case['metadata']['total_dfs_steps']} (total)")
        print(f"\nExpected:")
        print(f"  Current node: {expected['current_node']}")
        print(f"  Stack: {expected['stack']}")
        print(f"  Visited: {expected['visited_nodes']}")
        print(f"\nSimulated:")
        print(f"  Current node: {simulated['current_node']}")
        print(f"  Stack: {simulated['stack']}")
        print(f"  Visited: {simulated['visited_nodes']}")
        print(f"\nVerification:")
        print(f"  Current node match: {node_match} {'✓' if node_match else '✗'}")
        print(f"  Stack match: {stack_match} {'✓' if stack_match else '✗'}")
        print(f"  Visited match: {visited_match} {'✓' if visited_match else '✗'}")
        print(f"  Overall: {'PASS ✓' if is_correct else 'FAIL ✗'}")
        print(f"{'='*80}\n")

    return result


def verify_dataset(dataset_path: str, verify_all: bool = False,
                   sample_ids: List[int] = None, verbose: bool = False):
    """
    Verify one or more test cases from the dataset.
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Loaded dataset: {dataset_path}")
    print(f"Total samples: {len(dataset)}\n")

    # Determine which samples to verify
    if verify_all:
        samples_to_verify = dataset
    elif sample_ids:
        samples_to_verify = [tc for tc in dataset if tc['id'] in sample_ids]
    else:
        # Default: verify first sample
        samples_to_verify = [dataset[0]]

    # Verify each sample
    results = []
    for test_case in samples_to_verify:
        result = verify_test_case(test_case, verbose=verbose)
        results.append(result)

    # Summary
    print(f"\n{'='*80}")
    print("VERIFICATION SUMMARY")
    print(f"{'='*80}")
    total = len(results)
    valid = sum(1 for r in results if r['valid'])
    invalid = total - valid
    passed = sum(1 for r in results if r['valid'] and r['correct'])
    failed = sum(1 for r in results if r['valid'] and not r['correct'])

    print(f"Total verified: {total}")
    print(f"\nValidity Check:")
    print(f"  Valid: {valid} ({100*valid/total:.1f}%)")
    print(f"  Invalid: {invalid} ({100*invalid/total:.1f}%)")

    if invalid > 0:
        invalid_ids = [r['id'] for r in results if not r['valid']]
        print(f"  Invalid IDs: {invalid_ids}")

        # Show issue breakdown
        issue_counts = {}
        for r in results:
            if not r['valid']:
                for issue in r['issues']:
                    issue_counts[issue] = issue_counts.get(issue, 0) + 1

        print(f"\n  Issue breakdown:")
        for issue, count in issue_counts.items():
            print(f"    - {issue}: {count} samples")

    print(f"\nCorrectness Check (valid samples only):")
    if valid > 0:
        print(f"  Passed: {passed} ({100*passed/valid:.1f}%)")
        print(f"  Failed: {failed} ({100*failed/valid:.1f}%)")

        if failed > 0:
            failed_ids = [r['id'] for r in results if r['valid'] and not r['correct']]
            print(f"  Failed IDs: {failed_ids}")
    else:
        print(f"  No valid samples to check correctness")

    print(f"{'='*80}\n")

    return invalid == 0 and failed == 0


def filter_dataset(dataset_path: str, output_path: str, verbose: bool = False):
    """
    Filter out invalid samples and save cleaned dataset.
    """
    # Load dataset
    with open(dataset_path, 'r') as f:
        dataset = json.load(f)

    print(f"Filtering dataset: {dataset_path}")
    print(f"Original samples: {len(dataset)}\n")

    # Check all samples
    valid_samples = []
    invalid_samples = []

    for test_case in dataset:
        validity = check_validity(test_case)
        if validity["valid"]:
            valid_samples.append(test_case)
        else:
            invalid_samples.append({
                "id": test_case["id"],
                "issues": validity["issues"]
            })
            if verbose:
                print(f"Filtering out ID {test_case['id']}:")
                for issue in validity["issues"]:
                    print(f"  ✗ {issue}")

    # Summary
    print(f"\n{'='*80}")
    print("FILTER SUMMARY")
    print(f"{'='*80}")
    print(f"Original samples: {len(dataset)}")
    print(f"Valid samples: {len(valid_samples)}")
    print(f"Filtered out: {len(invalid_samples)}")

    if invalid_samples:
        print(f"\nFiltered IDs: {[s['id'] for s in invalid_samples]}")

    # Save filtered dataset
    if output_path:
        # Re-index valid samples
        for i, sample in enumerate(valid_samples):
            sample["id"] = i

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, 'w') as f:
            json.dump(valid_samples, f, indent=2)

        print(f"\nFiltered dataset saved to: {output_path}")
        print(f"New sample count: {len(valid_samples)}")

    print(f"{'='*80}\n")

    return len(invalid_samples) == 0


def visualize_dfs_trace(test_case: Dict, output_file: str = None):
    """
    Visualize the DFS trace step by step (optional feature).
    """
    graph = nx.Graph()
    graph.add_nodes_from(test_case["graph"]["nodes"])
    graph.add_edges_from(test_case["graph"]["edges"])

    start_node = test_case["start_node"]
    target_steps = test_case["steps"]

    print(f"\nDFS Trace Visualization (ID: {test_case['id']})")
    print(f"{'='*80}")
    print(f"Start: Node {start_node}")
    print(f"Target steps: {target_steps}\n")

    stack = [start_node]
    visited = {start_node}

    print(f"Step 0: Visit node {start_node}")
    print(f"  Stack: {stack}")
    print(f"  Visited: {sorted(visited)}\n")

    for step in range(1, min(target_steps + 1, 20)):  # Limit output
        if not stack:
            break

        current = stack[-1]
        neighbors = sorted(list(graph.neighbors(current)))
        unvisited = [n for n in neighbors if n not in visited]

        if unvisited:
            next_node = unvisited[0]
            stack.append(next_node)
            visited.add(next_node)
            action = f"Visit node {next_node} from {current}"
        else:
            popped = stack.pop()
            action = f"Backtrack from {popped}"
            if stack:
                action += f" to {stack[-1]}"

        print(f"Step {step}: {action}")
        print(f"  Stack: {stack}")
        print(f"  Visited: {sorted(visited)}\n")

    if target_steps > 20:
        print(f"... (showing first 20 steps, target is {target_steps})\n")

    print(f"Final state at step {target_steps}:")
    final = simulate_dfs_steps(graph, start_node, target_steps)
    print(f"  Current node: {final['current_node']}")
    print(f"  Stack: {final['stack']}")
    print(f"  Visited: {final['visited_nodes']}")
    print(f"{'='*80}\n")


def main():
    parser = argparse.ArgumentParser(
        description="Verify DFS state query dataset",
        epilog="""
Examples:
  # Verify all samples
  %(prog)s dfs_state_query/datasets/dfs_state_query_small.json --verify-all

  # Check specific samples
  %(prog)s dfs_state_query/datasets/dfs_state_query_small.json --sample-id 0 1 2 --verbose

  # Filter out invalid samples
  %(prog)s dfs_state_query/datasets/dfs_state_query_small.json --filter --output dfs_state_query/datasets/dfs_cleaned.json

  # Visualize DFS trace
  %(prog)s dfs_state_query/datasets/dfs_state_query_small.json --visualize 0
        """,
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("dataset", type=str, help="Path to dataset JSON file")
    parser.add_argument("--verify-all", action="store_true",
                        help="Verify all samples in the dataset")
    parser.add_argument("--sample-id", type=int, nargs="+",
                        help="Specific sample IDs to verify")
    parser.add_argument("--filter", action="store_true",
                        help="Filter out invalid samples (use with --output)")
    parser.add_argument("--output", "-o", type=str,
                        help="Output path for filtered dataset")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print detailed verification info for each sample")
    parser.add_argument("--visualize", type=int, metavar="ID",
                        help="Visualize DFS trace for a specific sample ID")

    args = parser.parse_args()

    if args.visualize is not None:
        # Load and visualize
        with open(args.dataset, 'r') as f:
            dataset = json.load(f)
        test_case = next(tc for tc in dataset if tc['id'] == args.visualize)
        visualize_dfs_trace(test_case)
    elif args.filter:
        # Filter mode
        if not args.output:
            print("Error: --output is required when using --filter")
            return 1

        success = filter_dataset(
            args.dataset,
            args.output,
            verbose=args.verbose
        )

        if success:
            print("✓ All samples are valid (no filtering needed)")
            return 0
        else:
            print("✓ Invalid samples filtered out")
            return 0
    else:
        # Verify mode
        success = verify_dataset(
            args.dataset,
            verify_all=args.verify_all,
            sample_ids=args.sample_id,
            verbose=args.verbose
        )

        if success:
            print("✓ All verified samples are valid and correct!")
            return 0
        else:
            print("✗ Some samples failed verification!")
            return 1


if __name__ == "__main__":
    exit(main())
