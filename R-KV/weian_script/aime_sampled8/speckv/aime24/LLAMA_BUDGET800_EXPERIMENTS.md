# SpeckV LLaMA AIME24 Budget=800 Experiments

## Overview

This document describes the SpeckV LLaMA budget=800 experiment added for low-budget algorithm comparison.

## New Script

| Script | kv_budget | Description |
|--------|-----------|-------------|
| `run_speckv_aime24_llama_norm_aligned_perhead_budget800.sh` | 800 | Low-budget SpeckV with per-head pruning |

## Existing Scripts (for reference)

| Script | kv_budget | Model |
|--------|-----------|-------|
| `run_speckv_aime24_llama_norm_aligned_perhead.sh` | 2048 | LLaMA |
| `run_speckv_aime24_qwen_norm_aligned_perhead.sh` | 2048 | Qwen |

## Parameters (budget=800 experiment)

| Parameter | Value |
|-----------|-------|
| Model | DeepSeek-R1-Distill-Llama-8B |
| Dataset | AIME24 (30 questions) |
| kv_budget | 800 |
| seed | 888 |
| num_samples | 8 |
| temperature | 0.6 |
| top_p | 0.95 |
| attn_implementation | flash_attention_2 |
| load_dtype | bfloat16 |
| sparse_stats_path | AIME25 LLaMA stats (cross-dataset) |

## SpeckV-specific Parameters

| Parameter | Value |
|-----------|-------|
| sparse_normalize_scores | true |
| include_prefill_in_budget | true |
| rkv_style_compression | true |
| rkv_style_slack_trigger | true |
| divide_length | 128 |
| per_head_pruning | true |
| sparse_round_window | 32 |

## Config

- `configs/aime_sampled8_speckv_aime24_llama_norm_aligned_budget800.yaml`

## Task Context

Created on 2025-01-09 for low-budget (kv_budget=800) algorithm comparison between R-KV and SpeckV on LLaMA setting.

## Notes

- SpeckV AIME25 LLaMA budget=800 is **NOT** included because AIME24 LLaMA `plain_stats.pt` does not exist (only `chat_stats.pt` available)
- Uses AIME25 LLaMA stats for AIME24 experiment (cross-dataset validation)
