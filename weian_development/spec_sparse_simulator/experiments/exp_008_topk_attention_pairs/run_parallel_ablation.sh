#!/bin/bash
source /data/rbg/users/weian/env/miniconda3/etc/profile.d/conda.sh
conda activate dc

EXP_DIR="/data/rbg/users/weian/project/rl/dc/weian_development/spec_sparse_simulator/experiments/exp_007_probe_activation_loss_ablation"
cd $EXP_DIR

# Lambda values and GPU assignments
LAMBDAS=(0.0 0.01 0.05 0.1 0.5)
GPUS=(1 2 4 5 6)

run_single_experiment() {
    local lambda=$1
    local gpu=$2
    local exp_name="lambda_${lambda//./_}"
    
    echo "[GPU $gpu] Starting lambda=$lambda"
    
    # Create output directories
    mkdir -p output/ablation/$exp_name/checkpoints
    mkdir -p output/ablation/$exp_name/logs
    
    # Run training with specific lambda and GPU
    CUDA_VISIBLE_DEVICES=$gpu python -c "
import sys
sys.path.insert(0, '$EXP_DIR')
import yaml
import copy
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
config['training']['lambda_activation'] = $lambda
config['output']['checkpoints_dir'] = 'output/ablation/$exp_name/checkpoints'
config['output']['logs_dir'] = 'output/ablation/$exp_name/logs'

# Train
logger.info('Training with lambda_activation=$lambda on GPU $gpu')
checkpoint_path = train(config, logger)

# Evaluate  
logger.info('Evaluating...')
eval_results = evaluate(config, checkpoint_path, logger)

# Save results
results = {
    'lambda_activation': $lambda,
    'checkpoint_path': str(checkpoint_path),
    'evaluation': eval_results
}
results_path = Path('output/ablation/$exp_name/results.json')
with open(results_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

logger.info('Completed lambda=$lambda')
" 2>&1 | tee output/ablation/$exp_name/run.log
    
    echo "[GPU $gpu] Finished lambda=$lambda"
}

# Run all experiments in parallel
for i in ${!LAMBDAS[@]}; do
    run_single_experiment ${LAMBDAS[$i]} ${GPUS[$i]} &
done

# Wait for all to complete
wait
echo "All experiments completed!"
