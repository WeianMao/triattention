# Calibration Guide

TriAttention uses pre-computed statistics (Q/K centers and norms) for each model. Pre-computed stats for supported models are included in `calibration/`.

## Generating Stats for a Custom Model

```bash
python scripts/calibrate.py \
    --model <your-model-id> \
    --calibration-data <your-data.jsonl> \
    --output-dir stats/
```

The calibration script computes Q/K distribution centers and Mean Resultant Length (R) values across all attention heads. These statistics are used at inference time to score keys via the trigonometric series.

## Pre-computed Stats

| Model | Stats Path |
|-------|-----------|
| Qwen3-8B | `calibration/qwen3-8b/` |
| DeepSeek-R1-Distill-Llama-8B | `calibration/ds-llama-8b/` |
| DeepSeek-R1-Distill-Qwen-7B | `calibration/ds-qwen-7b/` |
