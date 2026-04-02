# Calibration Guide

TriAttention uses pre-computed statistics (Q/K centers and norms) for each model. Pre-computed stats for supported models are included in `calibration/`.

## Generating Stats for a Custom Model

```bash
python scripts/calibrate.py \
    --model <your-model-id-or-path> \
    --input <calibration_text.txt> \
    --output calibration/model_stats.pt
```

The calibration script runs a forward pass on plain text input, captures query states from every attention layer, inverts RoPE, and computes per-head frequency statistics. The resulting `.pt` file is loaded at inference time to score keys via the trigonometric series.

## Pre-computed Stats

| Model | Stats Path |
|-------|-----------|
| Qwen3-8B | `calibration/qwen3_8b_stats.pt` |
| DeepSeek-R1-Distill-Llama-8B | `calibration/ds_llama_8b_stats.pt` |
| DeepSeek-R1-Distill-Qwen-7B | `calibration/ds_qwen_7b_stats.pt` |
