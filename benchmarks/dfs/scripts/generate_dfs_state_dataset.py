#!/usr/bin/env python3
"""
DFS State Query Dataset Generator

Generates a dataset for testing LLM's ability to simulate DFS execution
and report the state after k steps.

Usage:
    python dfs_state_query/scripts/generate_dfs_state_dataset.py \
        --num-samples 100 \
        --output dfs_state_query/datasets/dfs_state_query_small.json
"""

import json
import random
import argparse
from pathlib import Path
from typing import List, Dict, Set, Tuple

import networkx as nx

DEFAULT_OUTPUT_PATH = (
    Path(__file__).resolve().parents[1] / "datasets" / "dfs_state_query_small.json"
)


def simulate_dfs_with_stack_trace(
    graph: nx.Graph,
    start_node: int,
    max_steps: int = None
) -> List[Dict]:
    """
    Simulate DFS and record state at each step.

    Args:
        graph: NetworkX graph
        start_node: Starting node for DFS
        max_steps: Maximum steps to simulate (None = complete traversal)

    Returns:
        List of states, each containing:
            - step: int (step number, 0-indexed)
            - current_node: int (current node, None if stack is empty)
            - stack: List[int] (current DFS stack)
            - visited: Set[int] (visited nodes)
            - action: str ("start" | "visit" | "backtrack")
    """
    stack = [start_node]
    visited = {start_node}
    trace = []

    # Record initial state
    trace.append({
        "step": 0,
        "current_node": start_node,
        "stack": stack.copy(),
        "visited": visited.copy(),
        "action": "start"
    })

    step = 0
    while stack and (max_steps is None or step < max_steps):
        current = stack[-1]

        # Find unvisited neighbors
        neighbors = list(graph.neighbors(current))
        unvisited_neighbors = [n for n in neighbors if n not in visited]

        if unvisited_neighbors:
            # Visit first unvisited neighbor (consistent ordering)
            # Sort to ensure deterministic behavior
            unvisited_neighbors.sort()
            next_node = unvisited_neighbors[0]
            stack.append(next_node)
            visited.add(next_node)
            action = "visit"
            current = next_node
        else:
            # Backtrack
            stack.pop()
            current = stack[-1] if stack else None
            action = "backtrack"

        step += 1
        trace.append({
            "step": step,
            "current_node": current,
            "stack": stack.copy(),
            "visited": visited.copy(),
            "action": action
        })

    return trace


def generate_random_graph(num_nodes: int, graph_type: str = "tree") -> nx.Graph:
    """
    Generate a random connected graph.

    Args:
        num_nodes: Number of nodes
        graph_type: "tree" or "sparse" or "dense"

    Returns:
        NetworkX undirected graph
    """
    if graph_type == "tree":
        # Random tree (n nodes, n-1 edges)
        try:
            # For networkx >= 3.4
            graph = nx.random_labeled_tree(num_nodes)
        except AttributeError:
            # For networkx < 3.4
            graph = nx.random_tree(num_nodes).to_undirected()
    elif graph_type == "sparse":
        # Sparse graph with avg degree ~2-3
        k = 4  # nearest neighbors to connect
        p = 0.1  # rewiring probability
        graph = nx.connected_watts_strogatz_graph(num_nodes, k, p)
    elif graph_type == "dense":
        # Denser graph
        p = 0.3  # edge probability
        while True:
            graph = nx.erdos_renyi_graph(num_nodes, p)
            if nx.is_connected(graph):
                break
    else:
        raise ValueError(f"Unknown graph_type: {graph_type}")

    # Ensure nodes are labeled 0, 1, 2, ...
    mapping = {old: new for new, old in enumerate(sorted(graph.nodes()))}
    graph = nx.relabel_nodes(graph, mapping)

    return graph


def create_test_case(
    graph: nx.Graph,
    start_node: int,
    target_steps: int,
    case_id: int = 0
) -> Dict:
    """
    Create a single test case.

    Args:
        graph: NetworkX graph
        start_node: Starting node
        target_steps: Number of DFS steps to execute
        case_id: Unique ID for this test case

    Returns:
        Test case dictionary
    """
    # Simulate full DFS to get total steps
    full_trace = simulate_dfs_with_stack_trace(graph, start_node)
    total_steps = len(full_trace) - 1  # Exclude initial state

    # Ensure target_steps is valid
    if target_steps > total_steps:
        target_steps = total_steps

    # Get state at target step
    target_state = full_trace[target_steps]

    # Extract graph structure
    nodes = list(graph.nodes())
    edges = list(graph.edges())

    # Build test case
    test_case = {
        "id": case_id,
        "graph": {
            "nodes": nodes,
            "edges": edges
        },
        "start_node": start_node,
        "steps": target_steps,
        "answer": {
            "current_node": target_state["current_node"],
            "stack": target_state["stack"],
            "visited_nodes": sorted(list(target_state["visited"]))
        },
        "metadata": {
            "total_dfs_steps": total_steps,
            "graph_nodes": len(nodes),
            "graph_edges": len(edges),
            "action": target_state["action"]
        }
    }

    # Optionally include full trace for debugging
    # test_case["full_trace"] = full_trace

    return test_case


def generate_dataset(
    num_samples: int,
    min_nodes: int = 20,
    max_nodes: int = 30,
    min_steps: int = 20,
    max_steps: int = 50,
    graph_types: List[str] = ["tree", "sparse"],
    seed: int = 42,
    uniform_steps: bool = True,
    max_attempts_per_step: int = 200
) -> List[Dict]:
    """
    Generate a dataset of DFS state query test cases.

    Args:
        num_samples: Number of test cases to generate
        min_nodes: Minimum number of nodes in graph
        max_nodes: Maximum number of nodes in graph
        min_steps: Minimum DFS steps
        max_steps: Maximum DFS steps
        graph_types: List of graph types to sample from
        seed: Random seed

    Returns:
        List of test cases
    """
    random.seed(seed)
    dataset = []

    if not uniform_steps:
        for i in range(num_samples):
            # Random graph configuration
            num_nodes = random.randint(min_nodes, max_nodes)
            graph_type = random.choice(graph_types)

            # Generate graph
            try:
                graph = generate_random_graph(num_nodes, graph_type)
            except Exception as e:
                print(f"Warning: Failed to generate graph for sample {i}: {e}")
                continue

            # Starting node (usually 0)
            start_node = 0

            # Simulate to determine valid step range
            full_trace = simulate_dfs_with_stack_trace(graph, start_node)
            total_steps = len(full_trace) - 1

            # Random target steps within valid range
            actual_min_steps = min(min_steps, total_steps)
            actual_max_steps = min(max_steps, total_steps)

            if actual_min_steps > actual_max_steps:
                print(f"Warning: Graph {i} too small (only {total_steps} steps), skipping")
                continue

            target_steps = random.randint(actual_min_steps, actual_max_steps)

            # Create test case
            test_case = create_test_case(graph, start_node, target_steps, case_id=i)
            dataset.append(test_case)

            if (i + 1) % 10 == 0:
                print(f"Generated {i + 1}/{num_samples} samples")

        return dataset

    steps = list(range(min_steps, max_steps + 1))
    if not steps:
        raise ValueError("min_steps must be <= max_steps for uniform sampling")

    base_count = num_samples // len(steps)
    remainder = num_samples % len(steps)
    random.shuffle(steps)
    step_targets = {step: base_count for step in steps}
    for step in steps[:remainder]:
        step_targets[step] += 1

    case_id = 0
    for step in sorted(step_targets):
        target_count = step_targets[step]
        if target_count == 0:
            continue

        generated = 0
        attempts = 0
        max_attempts = max_attempts_per_step * target_count

        while generated < target_count and attempts < max_attempts:
            attempts += 1
            num_nodes = random.randint(min_nodes, max_nodes)
            graph_type = random.choice(graph_types)

            try:
                graph = generate_random_graph(num_nodes, graph_type)
            except Exception as e:
                print(f"Warning: Failed to generate graph for step {step}: {e}")
                continue

            start_node = 0
            full_trace = simulate_dfs_with_stack_trace(graph, start_node)
            total_steps = len(full_trace) - 1

            if total_steps < step:
                continue

            test_case = create_test_case(graph, start_node, step, case_id=case_id)
            dataset.append(test_case)
            case_id += 1
            generated += 1

        if generated < target_count:
            print(
                f"Warning: Step {step} shortfall "
                f"({generated}/{target_count}) after {attempts} attempts"
            )

        if len(dataset) % 10 == 0:
            print(f"Generated {len(dataset)}/{num_samples} samples")

    return dataset


def save_dataset(dataset: List[Dict], output_path: str):
    """Save dataset to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(dataset, f, indent=2)

    print(f"\nDataset saved to: {output_path}")
    print(f"Total samples: {len(dataset)}")


def print_sample(test_case: Dict):
    """Print a sample test case for inspection."""
    print("\n" + "="*80)
    print(f"Sample Test Case (ID: {test_case['id']})")
    print("="*80)
    print(f"Graph: {test_case['metadata']['graph_nodes']} nodes, "
          f"{test_case['metadata']['graph_edges']} edges")
    print(f"Start node: {test_case['start_node']}")
    print(f"Steps: {test_case['steps']} (Total DFS steps: {test_case['metadata']['total_dfs_steps']})")
    print(f"\nEdges: {test_case['graph']['edges'][:10]}..." if len(test_case['graph']['edges']) > 10
          else f"\nEdges: {test_case['graph']['edges']}")
    print(f"\nAnswer:")
    print(f"  Current node: {test_case['answer']['current_node']}")
    print(f"  Stack: {test_case['answer']['stack']}")
    print(f"  Visited: {test_case['answer']['visited_nodes']}")
    print(f"  Action at this step: {test_case['metadata']['action']}")
    print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate DFS state query dataset")
    parser.add_argument("--num-samples", type=int, default=100,
                        help="Number of test cases to generate")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT_PATH),
                        help="Output JSON file path")
    parser.add_argument("--min-nodes", type=int, default=20,
                        help="Minimum number of nodes")
    parser.add_argument("--max-nodes", type=int, default=30,
                        help="Maximum number of nodes")
    parser.add_argument("--min-steps", type=int, default=20,
                        help="Minimum DFS steps")
    parser.add_argument("--max-steps", type=int, default=50,
                        help="Maximum DFS steps")
    parser.add_argument("--graph-types", type=str, nargs="+",
                        default=["tree", "sparse"],
                        choices=["tree", "sparse", "dense"],
                        help="Types of graphs to generate")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--uniform-steps", action="store_true", default=True,
                        help="Enforce uniform distribution across steps")
    parser.add_argument("--no-uniform-steps", dest="uniform_steps",
                        action="store_false",
                        help="Disable uniform step distribution")
    parser.add_argument("--max-attempts-per-step", type=int, default=200,
                        help="Generation attempts per step bucket")
    parser.add_argument("--show-sample", action="store_true",
                        help="Print first sample for inspection")

    args = parser.parse_args()

    print(f"Generating {args.num_samples} test cases...")
    print(f"  Nodes: {args.min_nodes}-{args.max_nodes}")
    print(f"  Steps: {args.min_steps}-{args.max_steps}")
    print(f"  Graph types: {args.graph_types}")
    print(f"  Random seed: {args.seed}\n")

    dataset = generate_dataset(
        num_samples=args.num_samples,
        min_nodes=args.min_nodes,
        max_nodes=args.max_nodes,
        min_steps=args.min_steps,
        max_steps=args.max_steps,
        graph_types=args.graph_types,
        seed=args.seed,
        uniform_steps=args.uniform_steps,
        max_attempts_per_step=args.max_attempts_per_step
    )

    if dataset:
        save_dataset(dataset, args.output)

        if args.show_sample:
            print_sample(dataset[0])

        # Print statistics
        print("\nDataset Statistics:")
        print(f"  Samples: {len(dataset)}")
        avg_nodes = sum(tc['metadata']['graph_nodes'] for tc in dataset) / len(dataset)
        avg_steps = sum(tc['steps'] for tc in dataset) / len(dataset)
        avg_total_steps = sum(tc['metadata']['total_dfs_steps'] for tc in dataset) / len(dataset)
        print(f"  Average nodes: {avg_nodes:.1f}")
        print(f"  Average target steps: {avg_steps:.1f}")
        print(f"  Average total DFS steps: {avg_total_steps:.1f}")

        step_counts = {}
        for tc in dataset:
            step_counts[tc["steps"]] = step_counts.get(tc["steps"], 0) + 1
        min_step = min(step_counts)
        max_step = max(step_counts)
        print(f"  Step distribution: steps {min_step}-{max_step}")
        for step in range(min_step, max_step + 1):
            print(f"    step {step}: {step_counts.get(step, 0)}")

        # Action distribution
        action_counts = {}
        for tc in dataset:
            action = tc['metadata']['action']
            action_counts[action] = action_counts.get(action, 0) + 1
        print(f"  Action distribution: {action_counts}")
    else:
        print("ERROR: No valid samples generated!")


if __name__ == "__main__":
    main()
