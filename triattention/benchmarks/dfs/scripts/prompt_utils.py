"""Shared prompt-building utilities for DFS state query evaluation."""

from typing import Dict


def build_prompt(test_case: Dict) -> str:
    """Build evaluation prompt for DFS state query (English, step-by-step reasoning)."""
    graph = test_case["graph"]
    start_node = test_case["start_node"]
    steps = test_case["steps"]

    nodes = graph["nodes"]
    edges = graph["edges"]

    prompt = f"""Given the following undirected graph:
    Nodes: {nodes}
    Edges: {edges}

    Starting from node {start_node}, perform a Depth-First Search (DFS).

    IMPORTANT STEP DEFINITION:
    - Step 0: Before any movement, we are at the start node {start_node}.
    - Step 1: The first step means we have already moved from the start node to its first chosen neighbor.
    So after N steps, you should simulate N actual DFS actions (moves or backtracks), not counting the initial position.

    DFS Rules:
    1. Always visit unvisited neighbors first (choose the smallest numbered neighbor)
    2. If all neighbors are visited, backtrack to the parent node
    3. The stack always stores the entire path from the start node to the current node

    Question: After executing {steps} DFS steps (starting from step 1 as the first move):
    1. What is the current node?
    2. What is the current DFS stack state (path from start to current node)?
    3. What are all the visited nodes?

    Please solve this step-by-step:
    1. First, analyze the graph structure and list neighbors for each node (sorted ascending)
    2. Then simulate the DFS action for EACH step from step 1 to step {steps}
    3. At each step clearly update: current node, stack, visited set
    4. Finally, provide the result AFTER completing step {steps}

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
