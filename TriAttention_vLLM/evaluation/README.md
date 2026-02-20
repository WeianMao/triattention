# TriAttention vLLM Evaluation Framework

This directory contains the evaluation pipeline for TriAttention KV compression, adapted from the R-KV sharded dispatch framework. The key difference is using vLLM as the inference backend instead of HuggingFace.

## Architecture

```
evaluation/
├── dispatch/                      # Upstream: Task distribution
│   ├── triattention_sharded_dispatch.py
│   └── configs/
│       ├── triattention_v2_aime24.yaml
│       └── triattention_v2_aime24_quick.yaml
├── runner/                        # Midstream: vLLM inference
│   └── vllm_triattention_v2_runner.py     # V2 non-invasive path
├── merge/                         # Downstream: Result aggregation
│   └── merge_shards.py
└── eval/                          # Downstream: local eval helpers
    ├── eval_math_multi.py
    └── (supporting files)
```

## Quick Start

### V2 Quick Alignment (Recommended)

```bash
TriAttention_vLLM/evaluation/scripts/run_v2_hf_alignment_quick.sh

# Optional: dry-run only (no actual inference)
TriAttention_vLLM/evaluation/scripts/run_v2_hf_alignment_quick.sh --dry-run

# Optional: compare with HF merged result file
TriAttention_vLLM/evaluation/scripts/run_v2_hf_alignment_quick.sh /path/to/hf_merged.jsonl
```

V2 quick config:
`TriAttention_vLLM/evaluation/dispatch/configs/triattention_v2_aime24_quick.yaml`

### V2 HF-Aligned Sample8 (Strict Method Reuse)

```bash
TriAttention_vLLM/evaluation/scripts/run_v2_hf_alignment_sample8.sh

# Optional: dry-run
TriAttention_vLLM/evaluation/scripts/run_v2_hf_alignment_sample8.sh --dry-run

# Optional: compare with HF merged result file
TriAttention_vLLM/evaluation/scripts/run_v2_hf_alignment_sample8.sh /path/to/hf_merged.jsonl
```

HF-aligned config:
`TriAttention_vLLM/evaluation/dispatch/configs/triattention_v2_aime24_hf_strict.yaml`

Legacy per-head anchor config (for historical ~45% reproduction):
`TriAttention_vLLM/evaluation/dispatch/configs/triattention_v2_aime24_hf_perhead_anchor_legacy.yaml`

### Run Full Pipeline (dispatch + merge + eval)

```bash
# Activate the trivllm environment
conda activate trivllm

# Run with default config (V2 AIME24)
python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
    --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_v2_aime24.yaml

# Dry-run to preview commands without execution
python TriAttention_vLLM/evaluation/dispatch/triattention_sharded_dispatch.py \
    --config TriAttention_vLLM/evaluation/dispatch/configs/triattention_v2_aime24.yaml \
    --dry-run
```

Legacy V1 config is blocked by default in dispatcher. Use `--allow-legacy-v1` only when explicitly doing historical comparison.
Legacy V1 runner/config snapshot is archived under:
`TriAttention_vLLM/repository_archive/legacy_v0_v1_code_2026-02-20/`.

### Run Single Shard Manually (V2)

```bash
python TriAttention_vLLM/evaluation/runner/vllm_triattention_v2_runner.py \
    --dataset-path R-KV/HuggingFace/data/aime24.jsonl \
    --output-dir TriAttention_vLLM/evaluation/outputs/test \
    --model-path /data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-Distill-Qwen-7B \
    --shard-id 0 \
    --num-shards 1 \
    --num-samples 1 \
    --kv-budget 2048 \
    --enable-experimental-kv-compaction true
```

### Run Without Compression (FullKV Baseline, V2 Runner)

```bash
python TriAttention_vLLM/evaluation/runner/vllm_triattention_v2_runner.py \
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
| `divide_length` | Compression trigger interval | 128 |
| `protect_prefill` | Whether prefill is protected | true |
| `per_head_selection_semantics` | `per_head` 语义：`legacy_layer_local` / `hf_aligned_global_per_head` | legacy_layer_local |
| `enable_experimental_kv_compaction` | Whether to execute compaction (vs plan-only) | runner default: false (V2 YAML can override to true) |
| `num_samples` | Samples per question | 8 |
| `num_shards` | Number of parallel shards | 8 |

Sharding semantics (HF-aligned):
1. Every shard processes the full question set.
2. Draw ids (`run_id`) are partitioned across shards.
3. Example: `num_samples=8`, `num_shards=8` means 8 tasks and each shard handles 1 draw.

### Key Paths

| Path | Description |
|------|-------------|
| `model_path` | DeepSeek-R1-Distill-Qwen-7B |
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
    "backend": "vllm_triattention_v2",
    "kv_budget": 2048,
    "divide_length": 128,
    "protect_prefill": true,
    "enable_experimental_kv_compaction": true
}
```

## Comparison with R-KV Pipeline

| Aspect | R-KV | TriAttention vLLM |
|--------|------|-------------------|
| Backend | HuggingFace | vLLM |
| V2 Compression Path | N/A | Scheduler + Worker + Runner hook |
| CUDA Graphs | N/A | Disabled (enforce_eager=True) |
| Interface | Same | Same |
| Output Format | Same | Same |

## Requirements

- vLLM >= 0.15.0 (for V2 runner path)
- CUDA (CUDA graphs disabled, eager mode required)
- Sufficient GPU memory (32GB+ recommended for Qwen-7B)

## Troubleshooting

### Process Name Masking

The dispatcher and runner use `PD-L1_binder` process name for cluster compatibility. This is set via `VLLM_PROCESS_NAME_PREFIX` environment variable.

### HF Evaluation Reuse

Dispatch reuses HF official multi-sample evaluation script:
`R-KV/HuggingFace/evaluation/eval_math_multi.py`

### Memory Issues

If you encounter OOM errors:
1. Reduce `gpu_memory_utilization` (default: 0.9)
2. Reduce `num_samples` per question
3. Use fewer parallel shards

### Compression Not Triggering

1. Check that `disable_compression` is not set to true
2. Ensure sequence length exceeds `kv_budget + divide_length`
3. Confirm `enable_experimental_kv_compaction=true` if you expect actual KV mutation (runner default is false)
