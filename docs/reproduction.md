# Reproduction Guide

## Prerequisites

```bash
git clone https://github.com/TODO/triattention.git
cd triattention
pip install -e .
pip install flash-attn --no-build-isolation  # recommended
```

## Experiments

Experiment configs and scripts are in `scripts/experiments/`.

### AIME24 with TriAttention (Qwen3-8B)

```bash
python scripts/cli.py \
    --model Qwen/Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --kv-budget 2048
```

### MATH-500 with DeepSeek-R1-Distill-Qwen-7B

```bash
python scripts/cli.py \
    --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B \
    --dataset math500 \
    --method triattention \
    --kv-budget 512
```

### Baseline Comparison

```bash
python scripts/cli.py \
    --model Qwen/Qwen3-8B \
    --dataset aime25 \
    --method full \
    --kv-budget 2048
```

### Running All Experiments

See `scripts/experiments/` for full experiment configurations covering all model-dataset-budget combinations reported in the paper.
