#!/usr/bin/env python3
"""
Create a subset of DFS state query dataset with small step counts.

Generates new samples with steps in range [min_steps, max_steps].
Useful for quick testing and debugging.

Usage:
    pixi run python dfs_state_query/scripts/create_small_steps_subset.py \
        --num-samples 20 \
        --min-steps 5 \
        --max-steps 19 \
        --output dfs_state_query/datasets/dfs_state_query_small.json
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict
import networkx as nx


def generate_graph(graph_type: str, num_nodes: int) -> nx.Graph:
    """Generate a graph of specified type."""
    if graph_type == "tree":
        # Use random_labeled_tree for networkx >= 3.4
        try:
            return nx.random_labeled_tree(num_nodes, seed=random.randint(0, 100000))
        except AttributeError:
            # Fallback for older versions
            return nx.random_tree(num_nodes, seed=random.randint(0, 100000))
    elif graph_type == "sparse":
        # Sparse graph: tree + a few random edges
        try:
            G = nx.random_labeled_tree(num_nodes, seed=random.randint(0, 100000))
        except AttributeError:
            G = nx.random_tree(num_nodes, seed=random.randint(0, 100000))
        # Add 1-3 random edges
        for _ in range(random.randint(1, 3)):
            u, v = random.sample(range(num_nodes), 2)
            G.add_edge(u, v)
        return G
    else:
        raise ValueError(f"Unknown graph type: {graph_type}")


def simulate_dfs(graph: nx.Graph, start_node: int, target_steps: int = None) -> List[Dict]:
    """
    Simulate complete DFS and record state at each step.

    Returns:
        List of states at each step
    """
    stack = [start_node]
    visited = {start_node}
    states = []

    # Record initial state (step 0)
    states.append({
        "step": 0,
        "current_node": start_node,
        "stack": stack.copy(),
        "visited_nodes": sorted(list(visited)),
        "action": "start"
    })

    step = 0
    while stack:
        step += 1
        current = stack[-1]
        neighbors = sorted(list(graph.neighbors(current)))
        unvisited = [n for n in neighbors if n not in visited]

        if unvisited:
            next_node = unvisited[0]
            stack.append(next_node)
            visited.add(next_node)
            action = "visit"
        else:
            stack.pop()
            action = "backtrack"

        # Record state
        states.append({
            "step": step,
            "current_node": stack[-1] if stack else None,
            "stack": stack.copy(),
            "visited_nodes": sorted(list(visited)),
            "action": action
        })

        if target_steps and step >= target_steps:
            break

    return states


def generate_test_case(case_id: int, min_nodes: int, max_nodes: int,
                       min_steps: int, max_steps: int,
                       graph_types: List[str]) -> Dict:
    """Generate a single test case."""

    # Keep trying until we get a valid case
    max_attempts = 50
    for attempt in range(max_attempts):
        # Random parameters
        num_nodes = random.randint(min_nodes, max_nodes)
        graph_type = random.choice(graph_types)

        # Generate graph
        graph = generate_graph(graph_type, num_nodes)

        # Simulate full DFS
        start_node = 0
        states = simulate_dfs(graph, start_node)
        total_steps = len(states) - 1  # Subtract initial state

        # Check if we can get a valid step count in our range
        available_steps = [s["step"] for s in states
                          if min_steps <= s["step"] <= max_steps
                          and s["current_node"] is not None]

        if not available_steps:
            continue

        # Pick a random step from available range
        target_step = random.choice(available_steps)
        state = states[target_step]

        # Build test case
        test_case = {
            "id": case_id,
            "graph": {
                "nodes": list(range(num_nodes)),
                "edges": [[u, v] for u, v in graph.edges()]
            },
            "start_node": start_node,
            "steps": target_step,
            "answer": {
                "current_node": state["current_node"],
                "stack": state["stack"],
                "visited_nodes": state["visited_nodes"]
            },
            "metadata": {
                "total_dfs_steps": total_steps,
                "graph_nodes": num_nodes,
                "graph_edges": graph.number_of_edges(),
                "graph_type": graph_type,
                "action": state["action"]
            }
        }

        return test_case

    raise RuntimeError(f"Failed to generate valid test case after {max_attempts} attempts")


def main():
    parser = argparse.ArgumentParser(
        description="Generate small-step DFS state query dataset"
    )
    parser.add_argument("--num-samples", type=int, default=20,
                       help="Number of samples to generate (default: 20)")
    parser.add_argument("--min-steps", type=int, default=5,
                       help="Minimum step count (default: 5)")
    parser.add_argument("--max-steps", type=int, default=19,
                       help="Maximum step count (default: 19)")
    parser.add_argument("--min-nodes", type=int, default=10,
                       help="Minimum graph nodes (default: 10)")
    parser.add_argument("--max-nodes", type=int, default=20,
                       help="Maximum graph nodes (default: 20)")
    parser.add_argument("--graph-types", type=str, nargs="+",
                       default=["tree", "sparse"],
                       help="Graph types to generate")
    parser.add_argument("--output", "-o", type=str, required=True,
                       help="Output JSON file path")
    parser.add_argument("--seed", type=int, default=42,
                       help="Random seed (default: 42)")
    parser.add_argument("--show-stats", action="store_true",
                       help="Show statistics after generation")

    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)

    print(f"Generating {args.num_samples} test cases...")
    print(f"Step range: [{args.min_steps}, {args.max_steps}]")
    print(f"Node range: [{args.min_nodes}, {args.max_nodes}]")
    print(f"Graph types: {args.graph_types}")
    print()

    # Generate test cases
    dataset = []
    for i in range(args.num_samples):
        try:
            test_case = generate_test_case(
                case_id=i,
                min_nodes=args.min_nodes,
                max_nodes=args.max_nodes,
                min_steps=args.min_steps,
                max_steps=args.max_steps,
                graph_types=args.graph_types
            )
            dataset.append(test_case)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{args.num_samples} samples...")
        except Exception as e:
            print(f"Error generating sample {i}: {e}")
            continue

    print(f"\nGenerated {len(dataset)} samples successfully!")

    # Save dataset
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"Dataset saved to: {output_path}")

    # Show statistics
    if args.show_stats and dataset:
        steps = [tc["steps"] for tc in dataset]
        nodes = [tc["metadata"]["graph_nodes"] for tc in dataset]
        actions = [tc["metadata"]["action"] for tc in dataset]

        print(f"\n{'='*60}")
        print("DATASET STATISTICS")
        print(f"{'='*60}")
        print(f"Total samples: {len(dataset)}")
        print(f"\nSteps:")
        print(f"  Range: [{min(steps)}, {max(steps)}]")
        print(f"  Mean: {sum(steps)/len(steps):.1f}")
        print(f"  Distribution:")
        for i in range(args.min_steps, args.max_steps + 1, 5):
            count = sum(1 for s in steps if i <= s < i + 5)
            if count > 0:
                print(f"    [{i:2d}, {i+5:2d}): {count:2d} samples")

        print(f"\nGraph nodes:")
        print(f"  Range: [{min(nodes)}, {max(nodes)}]")
        print(f"  Mean: {sum(nodes)/len(nodes):.1f}")

        print(f"\nActions:")
        action_counts = {}
        for a in actions:
            action_counts[a] = action_counts.get(a, 0) + 1
        for action, count in sorted(action_counts.items()):
            print(f"  {action}: {count} ({100*count/len(dataset):.1f}%)")

        print(f"{'='*60}\n")
    elif args.show_stats and not dataset:
        print("\n⚠ No samples generated, skipping statistics")

    # Verify first sample
    if dataset:
        print("Verifying first sample...")
        from verify_dfs_dataset import verify_test_case
        result = verify_test_case(dataset[0], verbose=True)

        if result["valid"] and result["correct"]:
            print("✓ First sample verified successfully!")
        else:
            print("✗ First sample verification failed!")
            print(f"  Valid: {result['valid']}")
            print(f"  Correct: {result['correct']}")
            if result['issues']:
                print(f"  Issues: {result['issues']}")
    else:
        print("\n⚠ No samples to verify!")


if __name__ == "__main__":
    main()
