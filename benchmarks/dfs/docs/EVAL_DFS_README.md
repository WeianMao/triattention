# DFS State Query Evaluation Script Guide

## Overview

`eval_dfs_state_query.py` evaluates LLM performance on the DFS state prediction task.

## Quick Start

### 1. Basic Usage

```bash
python scripts/eval_dfs_state_query.py \
    --dataset datasets/legacy/dfs_state_query_test.json \
    --model-path /path/to/your/model \
    --output analysis/dfs_eval_test.json \
    --verbose
```

### 2. Test Mode (few samples)

```bash
python scripts/eval_dfs_state_query.py \
    --dataset datasets/legacy/dfs_state_query_100.json \
    --model-path /path/to/your/model \
    --output analysis/dfs_eval_sample.json \
    --max-samples 5 \
    --verbose
```

### 3. Full Evaluation

```bash
python scripts/eval_dfs_state_query.py \
    --dataset datasets/legacy/dfs_state_query_100.json \
    --model-path /path/to/your/model \
    --output analysis/dfs_eval_full.json \
    --max-new-tokens 512
```

## Parameters

- `--dataset`: Path to DFS state query dataset (JSON format)
- `--model-path`: HuggingFace model path (local or remote)
- `--output`: Output path for results (JSON format)
- `--max-samples`: Maximum number of samples to evaluate (for quick testing; defaults to all)
- `--max-new-tokens`: Maximum tokens for model generation (default: 512)
- `--verbose`: Show detailed output

## Output Format

Results are saved in JSON format:

```json
{
  "aggregate_metrics": {
    "current_node_correct": 0.85,
    "stack_exact_match": 0.72,
    "stack_depth_correct": 0.88,
    "stack_top_correct": 0.90,
    "stack_intersection_ratio": 0.85,
    "visited_exact_match": 0.68,
    "visited_precision": 0.92,
    "visited_recall": 0.89,
    "visited_f1": 0.90
  },
  "num_samples": 100,
  "num_evaluated": 95,
  "num_parse_errors": 5,
  "results": [...]
}
```

## Evaluation Metrics

### Core Metrics

1. **current_node_correct**: Current node prediction accuracy
2. **stack_exact_match**: Exact stack match accuracy (strictest)
3. **visited_exact_match**: Exact visited-node set match accuracy

### Partial Match Metrics

4. **stack_depth_correct**: Stack depth correctness ratio
5. **stack_top_correct**: Stack top element correctness ratio
6. **stack_intersection_ratio**: Stack node intersection ratio
7. **visited_precision**: Visited nodes precision
8. **visited_recall**: Visited nodes recall
9. **visited_f1**: Visited nodes F1 score

## Supported Models

The script uses the HuggingFace transformers library and supports all compatible models:

- Qwen2.5 series
- DeepSeek series
- Llama series
- Other chat models

Ensure the model supports `apply_chat_template`.

## Troubleshooting

### 1. Parse Errors

If model output cannot be parsed as JSON, it is recorded as `parse_error`. The script tries multiple extraction methods:
- Direct JSON parsing
- Markdown code block extraction
- Regex matching

### 2. Out of Memory

For large models, consider using quantized variants:
```bash
--model-path /path/to/model-int4
```

### 3. Truncated Generation

If model output is truncated, increase `--max-new-tokens`:
```bash
--max-new-tokens 1024
```

## License

Same license as the main AQA-Bench project.
