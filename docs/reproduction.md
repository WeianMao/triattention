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
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime24 \
    --method triattention \
    --budget 2048
```

### MATH-500 with DeepSeek-R1-Distill-Qwen-7B

```bash
python scripts/cli.py run-one \
    --model DeepSeek-R1-Distill-Qwen-7B \
    --dataset math500 \
    --method triattention \
    --budget 512
```

### Baseline Comparison

```bash
python scripts/cli.py run-one \
    --model Qwen3-8B \
    --dataset aime25 \
    --method fullkv
```

### Running All Experiments

See `scripts/experiments/` for full experiment configurations covering all model-dataset-budget combinations reported in the paper.
