# weian_script helpers

This directory contains convenience launchers for the HuggingFace-based math experiments.

### SDPA baselines (added for reproducibility)

- `run_rkv_aime24_single_sdpa.sh` – single-GPU DeepSeek R1 + R-KV run on AIME24 with `attn_implementation=sdpa`. Outputs to `R-KV/HuggingFace/outputs/output_sdpa.jsonl`.
- `run_rkv_aime24_sharded_sdpa.sh` – multi-GPU dispatcher that loads `configs/rkv_aime24_sharded_sdpa.yaml` (8 shards, sdpa) and handles auto GPU selection, merging, and evaluation.

Use these when you need deterministic attention for debugging/verification. All other scripts remain unchanged.
