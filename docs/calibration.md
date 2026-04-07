# Calibration Guide

TriAttention uses pre-computed statistics (Q/K centers and norms) for each model. Pre-computed stats for supported models are included in `triattention/calibration/`.

## Generating Stats for a Custom Model

```bash
python scripts/calibrate.py \
    --model <your-model-id-or-path> \
    --input <calibration_text.txt> \
    --output triattention/calibration/model_stats.pt
```

The calibration script runs a forward pass on plain text input, captures query states from every attention layer, inverts RoPE, and computes per-head frequency statistics. The resulting `.pt` file is loaded at inference time to score keys via the trigonometric series.

## Pre-computed Stats

Stats are organised by experiment target. Each sub-directory contains per-model `.pt` files:

**For AIME-24 experiments** (`triattention/calibration/for_aime24_experiment/`):

| Model | Stats Path |
|-------|-----------|
| Qwen3-8B | `triattention/calibration/for_aime24_experiment/qwen3_8b.pt` |
| DeepSeek-R1-Distill-Llama-8B | `triattention/calibration/for_aime24_experiment/ds_llama8b.pt` |
| DeepSeek-R1-Distill-Qwen-7B | `triattention/calibration/for_aime24_experiment/ds_qwen7b.pt` |

**For AIME-25 experiments** (`triattention/calibration/for_aime25_experiment/`):

| Model | Stats Path |
|-------|-----------|
| Qwen3-8B | `triattention/calibration/for_aime25_experiment/qwen3_8b.pt` |
| DeepSeek-R1-Distill-Llama-8B | `triattention/calibration/for_aime25_experiment/ds_llama8b.pt` |
| DeepSeek-R1-Distill-Qwen-7B | `triattention/calibration/for_aime25_experiment/ds_qwen7b.pt` |
