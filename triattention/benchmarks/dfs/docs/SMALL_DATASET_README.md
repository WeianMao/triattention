# DFS State Query - Small Steps Dataset

**Created**: 2025-12-30
**Purpose**: Quick testing and debugging with shorter reasoning chains

## Dataset Overview

- **File**: `dfs_state_query/datasets/dfs_state_query_small.json`
- **Samples**: 30
- **Step range**: 5-19 steps
- **Graph size**: 10-20 nodes
- **Graph types**: Tree, Sparse
- **Validity**: ✅ 100% valid, 100% correct

## Key Features

### 1. Shorter Reasoning Chains
- **Original dataset**: 20-50 steps (mean: 33.7)
- **This dataset**: 5-19 steps (mean: 12.2)
- **Benefits**:
  - Faster model inference
  - Easier to debug and understand
  - Good for initial testing

### 2. Steps Distribution

```
[ 5- 9]:  8 samples  ████████
[10-14]: 12 samples  ████████████
[15-19]: 10 samples  ██████████
```

### 3. Graph Characteristics

- **Node count**: 10-20 (mean: 15.1)
- **Graph types**:
  - Tree: Random spanning tree
  - Sparse: Tree + 1-3 extra edges
- **Action distribution**:
  - Backtrack: 57%
  - Visit: 43%

## Use Cases

### 1. Quick Model Testing

Perfect for rapid iteration when developing or debugging:

```bash
# Test with 5 samples (< 1 minute)
pixi run python dfs_state_query/scripts/quick_test.py \
    --model-path Qwen/Qwen2.5-7B-Instruct \
    --dataset dfs_state_query/datasets/dfs_state_query_small.json \
    --num-samples 5
```

### 2. Full Small Dataset Evaluation

```bash
# Evaluate all 30 samples (2-3 minutes)
pixi run python dfs_state_query/scripts/eval_dfs_state_query.py \
    --dataset dfs_state_query/datasets/dfs_state_query_small.json \
    --model-path /path/to/model \
    --output dfs_state_query/analysis/dfs_small_eval.json \
    --verbose
```

### 3. CoT (Chain-of-Thought) Testing

```bash
# Test with CoT prompts (requires more tokens)
pixi run python dfs_state_query/scripts/eval_dfs_cot_english.py \
    --dataset dfs_state_query/datasets/dfs_state_query_small.json \
    --model-path /path/to/model \
    --output dfs_state_query/analysis/dfs_small_cot.json \
    --max-new-tokens 1024
```

## Expected Performance

Based on step count, expected difficulty:

- **5-9 steps** (Easy): High accuracy expected (>90%)
- **10-14 steps** (Medium): Moderate difficulty (~70-90%)
- **15-19 steps** (Harder): Lower accuracy (~50-70%)

## Comparison with Other Datasets

| Dataset | Samples | Steps Range | Mean Steps | Use Case |
|---------|---------|-------------|------------|----------|
| `dfs_state_query_small.json` | 30 | 5-19 | 12.2 | Quick testing |
| `dfs_state_query_test.json` | 10 | 20-50 | ~34 | Sanity check |
| `dfs_state_query_100_clean.json` | 98 | 20-50 | 33.7 | Full evaluation |

## Generation Details

Generated using:
```bash
pixi run python dfs_state_query/scripts/create_small_steps_subset.py \
    --num-samples 30 \
    --min-steps 5 \
    --max-steps 19 \
    --min-nodes 10 \
    --max-nodes 20 \
    --output dfs_state_query/datasets/dfs_state_query_small.json \
    --seed 42
```

### Generation Parameters

- **Random seed**: 42 (reproducible)
- **Graph generation**: NetworkX random trees/sparse graphs
- **Step selection**: Random step in valid range
- **Validation**: All samples verified for correctness

## Sample Example

```json
{
  "id": 9,
  "graph": {
    "nodes": [0, 1, 2, ..., 10],
    "edges": [[0, 1], [0, 2], ...]
  },
  "start_node": 0,
  "steps": 5,
  "answer": {
    "current_node": 3,
    "stack": [0, 1, 3],
    "visited_nodes": [0, 1, 2, 3, 4, 5]
  },
  "metadata": {
    "total_dfs_steps": 21,
    "graph_nodes": 11,
    "graph_edges": 11,
    "graph_type": "sparse",
    "action": "backtrack"
  }
}
```

## Verification

All samples verified:
```bash
pixi run python dfs_state_query/scripts/verify_dfs_dataset.py \
    dfs_state_query/datasets/dfs_state_query_small.json \
    --verify-all
```

**Result**: ✅ 30/30 valid, 30/30 correct

## Regeneration

To generate a new version with different parameters:

```bash
# More samples, wider range
pixi run python dfs_state_query/scripts/create_small_steps_subset.py \
    --num-samples 50 \
    --min-steps 3 \
    --max-steps 25 \
    --output dfs_state_query/datasets/dfs_state_query_small_v2.json \
    --show-stats

# Very easy (for debugging)
pixi run python dfs_state_query/scripts/create_small_steps_subset.py \
    --num-samples 10 \
    --min-steps 3 \
    --max-steps 10 \
    --min-nodes 5 \
    --max-nodes 10 \
    --output dfs_state_query/datasets/dfs_state_query_tiny.json \
    --show-stats
```

## Tips

1. **Start here**: Use this dataset for initial model testing
2. **Iterate quickly**: Small dataset = fast feedback loop
3. **Debug failures**: Shorter steps = easier to trace errors
4. **Scale up**: Move to `dfs_state_query/datasets/legacy/dfs_state_query_100_clean.json` after success

## Related Files

- `../scripts/create_small_steps_subset.py` - Generation script
- `../scripts/verify_dfs_dataset.py` - Verification script
- `../scripts/quick_test.py` - Quick evaluation script

---

**Note**: This dataset is automatically validated and guaranteed to be free of confusing samples (no empty stacks, steps within valid range).
