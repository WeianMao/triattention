# Experiment Runner Usage Guide

## Overview

`run_pruning_experiment.py` provides automated experiment infrastructure for Module 1 Neural Network Key Pruning with:
- Configurable model architectures
- Automatic training from scratch
- Binary search for optimal threshold achieving target hit rate (>= 99.5%)
- Batch experiment execution support
- Result logging to JSON

## Quick Start

### Basic Usage

Run experiment with default parameters (3 kernels, 64 hidden dim, MLP):
```bash
python run_pruning_experiment.py
```

### Custom Configurations

```bash
# Experiment with 5 kernels
python run_pruning_experiment.py --num-kernels 5

# Experiment with 128 hidden dimensions
python run_pruning_experiment.py --mlp-hidden-dim 128

# Experiment with average pooling instead of MLP
python run_pruning_experiment.py --use-avg-pool

# Combined configuration
python run_pruning_experiment.py --num-kernels 5 --mlp-hidden-dim 128

# Custom experiment name
python run_pruning_experiment.py --num-kernels 3 --experiment-name baseline_exp
```

## Command-Line Arguments

- `--num-kernels`: Number of von Mises kernels per frequency band (default: 3)
- `--mlp-hidden-dim`: MLP hidden dimension (default: 64)
- `--use-avg-pool`: Use average pooling instead of MLP (flag, default: False)
- `--config`: Path to custom config YAML file (default: config.yaml)
- `--experiment-name`: Custom name for experiment (default: auto-generated)

## Batch Experiments

Create a bash script to run multiple experiments:

```bash
#!/bin/bash

# Experiment 1: Baseline (3 kernels, 64 hidden)
python run_pruning_experiment.py --num-kernels 3 --mlp-hidden-dim 64

# Experiment 2: More kernels (5 kernels, 64 hidden)
python run_pruning_experiment.py --num-kernels 5 --mlp-hidden-dim 64

# Experiment 3: Larger MLP (3 kernels, 128 hidden)
python run_pruning_experiment.py --num-kernels 3 --mlp-hidden-dim 128

# Experiment 4: Average pooling (3 kernels)
python run_pruning_experiment.py --num-kernels 3 --use-avg-pool

# Experiment 5: Minimal parameters (1 kernel, avg pool)
python run_pruning_experiment.py --num-kernels 1 --use-avg-pool
```

## Output Structure

Results are saved to `output/pruning_experiments/`:

```
output/
└── pruning_experiments/
    ├── k3_h64_mlp.json        # 3 kernels, 64 hidden, MLP
    ├── k5_h64_mlp.json        # 5 kernels, 64 hidden, MLP
    ├── k3_h128_mlp.json       # 3 kernels, 128 hidden, MLP
    └── k3_h64_avgpool.json    # 3 kernels, 64 hidden, avg pool
```

## Result Format

Each JSON file contains:

```json
{
  "experiment_name": "k3_h64_mlp",
  "config": {
    "num_kernels": 3,
    "mlp_hidden_dim": 64,
    "use_mlp": true
  },
  "metrics": {
    "param_count": 41156,
    "final_loss": 0.123456,
    "optimal_threshold": 0.542,
    "hit_rate": 0.9951,
    "keys_per_query": 45.32
  }
}
```

## Automatic Threshold Search

The script automatically searches for the optimal threshold that achieves:
- Target hit rate: >= 99.5%
- Method: Binary search
- Precision: 0.001
- Search range: [0.0, 1.0]

The algorithm finds the **smallest threshold** (most aggressive pruning) that still meets the hit rate target.

## Python API Usage

You can also import and use the functions directly:

```python
from run_pruning_experiment import run_pruning_experiment

# Run experiment programmatically
results = run_pruning_experiment(
    num_kernels=3,
    mlp_hidden_dim=64,
    use_mlp=True,
    experiment_name='my_experiment'
)

print(f"Hit rate: {results['metrics']['hit_rate']}")
print(f"Threshold: {results['metrics']['optimal_threshold']}")
print(f"Keys per query: {results['metrics']['keys_per_query']}")
```

## Validation

Run validation tests to ensure everything is set up correctly:

```bash
python test_experiment_runner.py
```

This will test:
- Module imports
- Output directory creation
- Model creation with different configs
- Config file loading

## Logs

Training and experiment logs are saved to `output/logs/pruning_experiments.log`.

## Notes

- Training uses config.yaml settings (100 epochs, lr=0.001, Adam optimizer)
- Model is trained from scratch for each experiment
- GPU is used if available, otherwise falls back to CPU
- Random seed is set to 42 for reproducibility
- Threshold search typically completes in 10-20 iterations
