# DFS State Query Dataset

## Overview

A dataset for testing LLMs' ability to simulate DFS (Depth-First Search) execution. Unlike existing interactive DFS benchmarks, this dataset requires the model to directly predict the state after executing k steps of DFS given a graph structure and step count k.

## Dataset Characteristics

- **Task type**: Intermediate state query (non-interactive)
- **Step range**: 20-50 steps (long reasoning chains)
- **Graph scale**: 20-30 nodes
- **Graph types**: Random trees and sparse graphs
- **Sample count**: 100 test samples

## Data Format

Each test sample contains the following fields:

```json
{
  "id": 0,
  "graph": {
    "nodes": [0, 1, 2, ...],
    "edges": [[0, 1], [0, 2], ...]
  },
  "start_node": 0,
  "steps": 37,
  "answer": {
    "current_node": 4,
    "stack": [0, 17, 23, 4],
    "visited_nodes": [0, 1, 2, 3, 4, ...]
  },
  "metadata": {
    "total_dfs_steps": 58,
    "graph_nodes": 30,
    "graph_edges": 29,
    "action": "visit" or "backtrack"
  }
}
```

### Field Descriptions

- **graph**: Undirected graph structure
  - `nodes`: Node list (labels: 0, 1, 2, ...)
  - `edges`: Edge list (undirected edges)

- **start_node**: DFS starting node (typically 0)

- **steps**: Number of DFS steps to execute

- **answer**: Ground truth
  - `current_node`: Current node after executing k steps
  - `stack`: Current DFS stack state (path from root to current node)
  - `visited_nodes`: All visited nodes (sorted)

- **metadata**: Metadata
  - `total_dfs_steps`: Total steps for complete DFS traversal
  - `graph_nodes`: Number of graph nodes
  - `graph_edges`: Number of graph edges
  - `action`: Action type at step k ("visit" or "backtrack")

## DFS Rules

Standard depth-first search rules:

1. **Visit strategy**: If the current node has unvisited neighbors, choose the smallest-numbered neighbor (ensures determinism)
2. **Backtrack strategy**: If all neighbors of the current node are visited, backtrack to the parent node
3. **Stack maintenance**: The stack stores the complete path from the start node to the current node

## Usage

### 1. Generate Dataset

```bash
python scripts/generate_dfs_state_dataset.py \
    --num-samples 100 \
    --output datasets/dfs_state_query_100.json \
    --min-nodes 20 \
    --max-nodes 30 \
    --min-steps 20 \
    --max-steps 50 \
    --seed 42
```

Parameters:
- `--num-samples`: Number of samples to generate
- `--output`: Output file path
- `--min-nodes`, `--max-nodes`: Range of graph node counts
- `--min-steps`, `--max-steps`: Range of DFS step counts
- `--graph-types`: Graph types (tree, sparse, dense)
- `--seed`: Random seed
- `--show-sample`: Display the first sample

### 2. Verify Dataset

Verify specific samples:
```bash
python scripts/verify_dfs_dataset.py \
    datasets/dfs_state_query_100.json \
    --sample-id 0 1 2 --verbose
```

Verify entire dataset:
```bash
python scripts/verify_dfs_dataset.py \
    datasets/dfs_state_query_100.json \
    --verify-all
```

## Evaluation Metrics

Recommended metrics:

1. **Node accuracy**: Whether `current_node` prediction is correct
2. **Stack accuracy**: Whether `stack` prediction exactly matches
3. **Partial stack accuracy**:
   - Stack depth correctness ratio
   - Stack top element correctness ratio
   - Stack node intersection ratio

## Dataset Statistics

Based on 100-sample dataset (seed=42):

- Mean node count: 24.9
- Mean target steps: 33.9
- Mean total DFS steps: 48.8
- Action distribution:
  - Backtrack: 78%
  - Visit: 22%

## Comparison with Existing DFS Benchmarks

| Feature | Existing DFS Benchmark | DFS State Query Dataset |
|---------|----------------------|------------------------|
| Interaction | Step-by-step | One-shot query |
| Model task | Choose next node | Predict state after k steps |
| Verification | Per-step DFS rule check | Final state verification |
| Step range | Full traversal (typically 15-30) | 20-50 steps |
| Difficulty | Medium | Higher (requires long-chain reasoning) |

## File List

- `../datasets/dfs_state_query_small.json`: Small-step dataset (for quick testing)
- `../datasets/legacy/dfs_state_query_100.json`: Main dataset (100 samples)
- `../datasets/legacy/dfs_state_query_test.json`: Test dataset (10 samples)
- `../scripts/generate_dfs_state_dataset.py`: Data generation script
- `../scripts/verify_dfs_dataset.py`: Verification script

## License

Same license as the main AQA-Bench project.
