# RKV LLaMA AIME24 Experiments

## Overview

This directory contains R-KV experiments on AIME24 dataset using DeepSeek-R1-Distill-Llama-8B model.

## Scripts

| Script | kv_budget | Description |
|--------|-----------|-------------|
| `run_rkv_aime24_llama.sh` | 2048 | Baseline R-KV LLaMA experiment |
| `run_rkv_aime24_llama_budget800.sh` | 800 | Low-budget comparison experiment |
| `run_rkv_aime24_qwen.sh` | 2048 | Qwen version (existing) |

## Common Parameters

| Parameter | Value |
|-----------|-------|
| Model | DeepSeek-R1-Distill-Llama-8B |
| Dataset | AIME24 (30 questions) |
| seed | 888 |
| num_samples | 8 |
| temperature | 0.6 |
| top_p | 0.95 |
| attn_implementation | flash_attention_2 |
| load_dtype | bfloat16 |

## Configs

- `configs/aime_sampled8_rkv_aime24_llama.yaml` (budget=2048)
- `configs/aime_sampled8_rkv_aime24_llama_budget800.yaml` (budget=800)

## Task Context

Created on 2025-01-09 to:
1. Migrate RKV LLaMA experiments from `aime24_official_sampled8/` to new directory structure
2. Add budget=800 experiments for low-budget algorithm comparison with SpeckV

## Notes

- Original `run_rkv_aime24_official_sampled8.sh` used seed=666, migrated scripts use seed=888 for consistency with other experiments
- `--dataset aime24` flag added to ensure correct eval output directory naming
