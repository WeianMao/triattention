#!/bin/bash
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate dc

EXP_DIR="/data/rbg/users/weian/project/rl/dc/weian_development/spec_sparse_simulator/experiments/exp_007_probe_activation_loss_ablation"
cd $EXP_DIR

# High lambda_balance values to overcome collapse
# Note: lambda_activation=0.0 (only testing balance loss)
LAMBDAS=(0.5 1.0 2.0 5.0 10.0)
GPUS=(0 1 2 3 4)

run_single_experiment() {
    local lambda=$1
    local gpu=$2
    local exp_name="balance_high_lambda_${lambda//./_}"

    echo "[GPU $gpu] Starting lambda_balance=$lambda"

    # Create output directories
    mkdir -p output/ablation_balance_high/$exp_name/checkpoints
    mkdir -p output/ablation_balance_high/$exp_name/logs

    # Run training with specific lambda_balance and GPU
    CUDA_VISIBLE_DEVICES=$gpu python -c "
import sys
sys.path.insert(0, '$EXP_DIR')
import yaml
import json
from pathlib import Path
from train import train
from evaluate import evaluate
import logging

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Load config
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

# Modify config for this experiment
config['training']['lambda_activation'] = 0.0
config['training']['lambda_balance'] = $lambda
config['output']['checkpoints_dir'] = 'output/ablation_balance_high/$exp_name/checkpoints'
config['output']['logs_dir'] = 'output/ablation_balance_high/$exp_name/logs'

# Train
logger.info('Training with lambda_balance=$lambda (lambda_activation=0.0) on GPU $gpu')
checkpoint_path = train(config, logger)

# Evaluate (includes collapse metrics)
logger.info('Evaluating...')
eval_results = evaluate(config, checkpoint_path, logger)

# Save results
results = {
    'lambda_activation': 0.0,
    'lambda_balance': $lambda,
    'checkpoint_path': str(checkpoint_path),
    'evaluation': eval_results
}
results_path = Path('output/ablation_balance_high/$exp_name/results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=lambda x: float(x) if hasattr(x, 'item') else str(x))

logger.info('Completed lambda_balance=$lambda')
" 2>&1 | tee output/ablation_balance_high/$exp_name/run.log

    echo "[GPU $gpu] Finished lambda_balance=$lambda"
}

# Run all experiments in parallel
for i in ${!LAMBDAS[@]}; do
    run_single_experiment ${LAMBDAS[$i]} ${GPUS[$i]} &
done

# Wait for all to complete
wait
echo "All high-weight balance loss experiments completed!"
