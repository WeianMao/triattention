# TriAttention vLLM Evaluation Framework

This directory contains the evaluation pipeline for TriAttention KV compression, adapted from the R-KV sharded dispatch framework. The key difference is using vLLM as the inference backend instead of HuggingFace.

## Architecture

```
evaluation/
├── dispatch/                      # Upstream: Task distribution
│   ├── triattention_sharded_dispatch.py
│   └── configs/
│       └── triattention_aime24.yaml
├── runner/                        # Midstream: vLLM inference
│   └── vllm_triattention_runner.py
├── merge/                         # Downstream: Result aggregation
│   └── merge_shards.py
└── eval/                          # Downstream: Accuracy calculation
    ├── eval_math_multi.py
    └── (supporting files)
```

## Quick Start

### Run Full Pipeline (dispatch + merge + eval)

```bash
# Activate the rkv environment
conda activate rkv

# Run with default config (AIME24 dataset)
python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
    --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml

# Dry-run to preview commands without execution
python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
    --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_aime24.yaml \
    --dry-run
```

### Run Single Shard Manually

```bash
python TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py \
    --dataset-path R-KV/HuggingFace/data/aime24.jsonl \
    --output-dir TriAttention_vLLM/evaluation/outputs/test \
    --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B \
    --sparse-stats-path R-KV/outputs/repository/sample8_fullkv_aime25_official_qwen/stats/deepseek_r1_qwen7b_plain_stats.pt \
    --shard-id 0 \
    --num-shards 1 \
    --num-samples 1 \
    --kv-budget 2048
```

### Run Without Compression (FullKV Baseline)

```bash
python TriAttention_vLLM/evaluation/runner/vllm_triattention_runner.py \
    --dataset-path R-KV/HuggingFace/data/aime24.jsonl \
    --output-dir TriAttention_vLLM/evaluation/outputs/fullkv \
    --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B \
    --shard-id 0 \
    --num-shards 1 \
    --num-samples 1 \
    --disable-compression true
```

## Configuration

### YAML Config Parameters

| Parameter | Description | Default |
|-----------|-------------|---------|
| `kv_budget` | Maximum KV cache size | 2048 |
| `window_size` | Recent token protection window | 128 |
| `divide_length` | Compression trigger interval | 128 |
| `pruning_mode` | Token selection strategy | "per_head" |
| `num_samples` | Samples per question | 8 |
| `num_shards` | Number of parallel shards | 8 |

### Key Paths

| Path | Description |
|------|-------------|
| `model_path` | DeepSeek-R1-Distill-Qwen-7B |
| `sparse_stats_path` | Precomputed frequency statistics |
| `dataset_path` | AIME24 evaluation dataset |

## Output Format

Each run produces JSONL files with records containing:

```json
{
    "index": 0,
    "question": "...",
    "answer": "...",
    "prompt": "...",
    "output": "...",
    "prefill_tokens": 123,
    "output_tokens": 4567,
    "total_tokens": 4690,
    "sample_idx": 0,
    "draw_idx": 0,
    "backend": "vllm_triattention",
    "kv_budget": 2048,
    "pruning_mode": "per_head"
}
```

## Comparison with R-KV Pipeline

| Aspect | R-KV | TriAttention vLLM |
|--------|------|-------------------|
| Backend | HuggingFace | vLLM |
| Compression | SpeckV (generate wrapper) | TriAttention (attention patch) |
| CUDA Graphs | N/A | Disabled (enforce_eager=True) |
| Interface | Same | Same |
| Output Format | Same | Same |

## Requirements

- vLLM >= 0.7.0
- CUDA (CUDA graphs disabled, eager mode required)
- Sufficient GPU memory (32GB+ recommended for Qwen-7B)

## Troubleshooting

### Process Name Masking

The dispatcher and runner use `PD-L1_binder` process name for cluster compatibility. This is set via `VLLM_PROCESS_NAME_PREFIX` environment variable.

### Memory Issues

If you encounter OOM errors:
1. Reduce `gpu_memory_utilization` (default: 0.9)
2. Reduce `num_samples` per question
3. Use fewer parallel shards

### Compression Not Triggering

1. Verify `sparse_stats_path` exists and is valid
2. Check that `disable_compression` is not set to true
3. Ensure sequence length exceeds `kv_budget + divide_length`
