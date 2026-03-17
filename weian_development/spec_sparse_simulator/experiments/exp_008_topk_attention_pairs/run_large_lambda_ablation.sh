#!/bin/bash
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate dc

EXP_DIR="/data/rbg/users/weian/project/rl/dc/weian_development/spec_sparse_simulator/experiments/exp_007_probe_activation_loss_ablation"
cd $EXP_DIR

# Using 7 GPUs (1-7), keeping GPU 0 free
# Total: 10 experiments, run in 2 phases (7 + 3)

run_single_experiment() {
    local lambda=$1
    local use_weighted=$2  # "True" or "False"
    local gpu=$3
    local version=$4       # "uniform" or "weighted"
    local exp_name="lambda_${lambda//./_}"
    local output_dir="output/ablation_large_${version}/$exp_name"

    echo "[GPU $gpu] Starting lambda=$lambda ($version)"

    # Create output directories
    mkdir -p $output_dir/checkpoints
    mkdir -p $output_dir/logs

    # Run training with specific lambda, use_weighted, and GPU
    CUDA_VISIBLE_DEVICES=$gpu python -c "
import sys
sys.path.insert(0, '$EXP_DIR')
import yaml
import json
from pathlib import Path
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify config for this experiment
config['training']['lambda_activation'] = $lambda
config['output']['checkpoints_dir'] = '$output_dir/checkpoints'
config['output']['logs_dir'] = '$output_dir/logs'

# Monkey-patch to control use_weighted
import train as train_module

_original_compute_probe_activation_loss = train_module.compute_probe_activation_loss

def patched_compute_probe_activation_loss(key_logits, key_probs, query_bin_probs, batch_argmax_keys, num_bins, alpha_dead_threshold=0.05, use_weighted=True):
    return _original_compute_probe_activation_loss(
        key_logits, key_probs, query_bin_probs, batch_argmax_keys,
        num_bins, alpha_dead_threshold, use_weighted=$use_weighted
    )

train_module.compute_probe_activation_loss = patched_compute_probe_activation_loss

from train import train
from evaluate import evaluate

# Train
logger.info('Training with lambda_activation=$lambda on GPU $gpu ($version)')
checkpoint_path = train(config, logger)

# Evaluate
logger.info('Evaluating...')
eval_results = evaluate(config, checkpoint_path, logger)

# Save results
results = {
    'lambda_activation': $lambda,
    'checkpoint_path': str(checkpoint_path),
    'evaluation': eval_results,
    'use_weighted': $use_weighted,
    'version': '$version'
}
results_path = Path('$output_dir/results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

logger.info('Completed lambda=$lambda ($version)')
" 2>&1 | tee $output_dir/run.log

    echo "[GPU $gpu] Finished lambda=$lambda ($version)"
}

echo "=== Phase 1: Running 7 experiments in parallel (GPU 1-7) ==="
# GPU 1: uniform lambda=1
# GPU 2: uniform lambda=2
# GPU 3: uniform lambda=5
# GPU 4: uniform lambda=10
# GPU 5: uniform lambda=20
# GPU 6: weighted lambda=1
# GPU 7: weighted lambda=2

run_single_experiment 1.0 False 1 uniform &
run_single_experiment 2.0 False 2 uniform &
run_single_experiment 5.0 False 3 uniform &
run_single_experiment 10.0 False 4 uniform &
run_single_experiment 20.0 False 5 uniform &
run_single_experiment 1.0 True 6 weighted &
run_single_experiment 2.0 True 7 weighted &

wait
echo "=== Phase 1 complete ==="

echo "=== Phase 2: Running remaining 3 experiments ==="
# GPU 1: weighted lambda=5
# GPU 2: weighted lambda=10
# GPU 3: weighted lambda=20
run_single_experiment 5.0 True 1 weighted &
run_single_experiment 10.0 True 2 weighted &
run_single_experiment 20.0 True 3 weighted &

wait
echo "=== Phase 2 complete ==="

echo "All large lambda experiments completed!"
