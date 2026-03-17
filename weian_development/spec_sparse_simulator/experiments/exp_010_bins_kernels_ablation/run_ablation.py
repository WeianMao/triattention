"""
Ablation Study: Bins & Kernels Parameter Sweep

This script runs ablation experiments for:
- num_bins: [128, 64, 32, 16, 8]
- num_kernels: [3, 1]

Features:
- Multi-GPU parallel execution
- Independent output directories per configuration
- Aggregated summary results

Usage:
    # Run all configurations with auto GPU assignment
    python run_ablation.py --gpus 0,2,3,4,5

    # Run specific configurations
    python run_ablation.py --bins 64,32 --kernels 1 --gpus 0,2

    # Dry run (show configurations without running)
    python run_ablation.py --dry-run
"""

import argparse
import copy
import json
import logging
import multiprocessing as mp
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import torch
import yaml


def setup_logging(log_file=None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def calculate_params(num_bins, num_kernels, num_freqs=64):
    """
    Calculate total model parameters.

    Formula per KernelEncodingLayer:
        params = num_bins * num_freqs * num_kernels * 3 + num_bins

    Total (2 networks: key + query):
        total = 2 * (num_bins * num_freqs * num_kernels * 3 + num_bins)
    """
    per_layer = num_bins * num_freqs * num_kernels * 3 + num_bins
    total = 2 * per_layer
    return total


def run_single_config(args):
    """
    Run training and evaluation for a single configuration.

    Args:
        args: Tuple of (config_id, num_bins, num_kernels, gpu_id, base_config, exp_dir)

    Returns:
        Dict with results
    """
    config_id, num_bins, num_kernels, gpu_id, base_config, exp_dir = args

    # Set CUDA device
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Create modified config
    config = copy.deepcopy(base_config)
    config['model']['num_bins'] = num_bins
    config['model']['num_kernels'] = num_kernels

    # Create output directories for this configuration
    config_name = f"{num_bins}bins_{num_kernels}kernels"
    output_base = exp_dir / 'output' / f'ablation_{config_name}'

    config['output']['base_dir'] = str(output_base)
    config['output']['logs_dir'] = str(output_base / 'logs')
    config['output']['checkpoints_dir'] = str(output_base / 'checkpoints')
    config['output']['results_dir'] = str(output_base / 'results')

    # Create directories
    for dir_key in ['logs_dir', 'checkpoints_dir', 'results_dir']:
        Path(config['output'][dir_key]).mkdir(parents=True, exist_ok=True)

    # Setup logging
    log_file = Path(config['output']['logs_dir']) / 'ablation_run.log'
    logger = setup_logging(log_file)

    start_time = time.time()
    result = {
        'config_id': config_id,
        'num_bins': num_bins,
        'num_kernels': num_kernels,
        'gpu_id': gpu_id,
        'total_params': calculate_params(num_bins, num_kernels),
        'status': 'running',
        'start_time': datetime.now().isoformat()
    }

    logger.info(f"=" * 60)
    logger.info(f"Starting ablation config: {config_name}")
    logger.info(f"num_bins={num_bins}, num_kernels={num_kernels}, GPU={gpu_id}")
    logger.info(f"Total parameters: {result['total_params']:,}")
    logger.info(f"=" * 60)

    try:
        # Import here to avoid CUDA initialization issues
        from train import train
        from evaluate import evaluate

        # Train
        logger.info("Starting training...")
        checkpoint_path = train(config, logger)

        # Evaluate
        logger.info("Starting evaluation...")
        checkpoint_path = Path(config['output']['checkpoints_dir']) / 'best_model.pt'
        hit_rates = evaluate(config, checkpoint_path, logger)

        # Record results
        result['status'] = 'success'
        result['hit_rates'] = hit_rates
        result['checkpoint_path'] = str(checkpoint_path)

    except Exception as e:
        logger.error(f"Error in config {config_name}: {e}", exc_info=True)
        result['status'] = 'failed'
        result['error'] = str(e)

    end_time = time.time()
    result['duration_seconds'] = end_time - start_time
    result['end_time'] = datetime.now().isoformat()

    # Save individual result
    result_file = Path(config['output']['results_dir']) / 'ablation_result.json'
    with open(result_file, 'w') as f:
        json.dump(result, f, indent=2)

    logger.info(f"Config {config_name} completed in {result['duration_seconds']:.1f}s")

    return result


def run_ablation(bins_list, kernels_list, gpu_ids, base_config, exp_dir, logger, dry_run=False):
    """
    Run ablation study across all bin/kernel configurations.

    Args:
        bins_list: List of num_bins values to test
        kernels_list: List of num_kernels values to test
        gpu_ids: List of available GPU IDs
        base_config: Base configuration dict
        exp_dir: Experiment directory path
        logger: Logger instance
        dry_run: If True, only show configurations without running

    Returns:
        List of result dicts
    """
    # Generate all configurations
    configs = []
    config_id = 0
    for num_bins in bins_list:
        for num_kernels in kernels_list:
            configs.append({
                'id': config_id,
                'num_bins': num_bins,
                'num_kernels': num_kernels,
                'params': calculate_params(num_bins, num_kernels)
            })
            config_id += 1

    # Display configuration matrix
    logger.info("=" * 60)
    logger.info("Ablation Study Configuration Matrix")
    logger.info("=" * 60)

    baseline_params = calculate_params(128, 3)

    logger.info(f"\n{'ID':<4} {'Bins':<6} {'Kernels':<8} {'Params':<12} {'Relative':<10}")
    logger.info("-" * 50)
    for cfg in configs:
        relative = cfg['params'] / baseline_params * 100
        logger.info(f"{cfg['id']:<4} {cfg['num_bins']:<6} {cfg['num_kernels']:<8} {cfg['params']:<12,} {relative:.1f}%")

    logger.info(f"\nTotal configurations: {len(configs)}")
    logger.info(f"Available GPUs: {gpu_ids}")

    if dry_run:
        logger.info("\n[DRY RUN] Skipping actual execution")
        return []

    # Prepare tasks with GPU assignment
    tasks = []
    gpu_cycle = len(gpu_ids)

    for i, cfg in enumerate(configs):
        gpu_id = gpu_ids[i % gpu_cycle]
        tasks.append((
            cfg['id'],
            cfg['num_bins'],
            cfg['num_kernels'],
            gpu_id,
            base_config,
            exp_dir
        ))

    logger.info(f"\nStarting parallel execution on {len(gpu_ids)} GPUs...")

    # Run tasks in parallel (one per GPU at a time)
    results = []

    # Process in batches equal to GPU count
    for batch_start in range(0, len(tasks), gpu_cycle):
        batch_end = min(batch_start + gpu_cycle, len(tasks))
        batch_tasks = tasks[batch_start:batch_end]

        logger.info(f"\nBatch {batch_start // gpu_cycle + 1}: configs {batch_start} to {batch_end - 1}")

        # Use multiprocessing pool
        with mp.Pool(processes=len(batch_tasks)) as pool:
            batch_results = pool.map(run_single_config, batch_tasks)

        results.extend(batch_results)

        # Log batch summary
        for res in batch_results:
            status = res['status']
            if status == 'success':
                hit_rate_50 = res['hit_rates'].get('50', {}).get('hit_rate', 0)
                logger.info(f"  Config {res['config_id']}: {res['num_bins']}bins_{res['num_kernels']}kernels -> Hit@50={hit_rate_50:.2f}%")
            else:
                logger.info(f"  Config {res['config_id']}: {res['num_bins']}bins_{res['num_kernels']}kernels -> FAILED")

    return results


def aggregate_results(results, exp_dir, logger):
    """
    Aggregate all results into a summary file.

    Args:
        results: List of result dicts
        exp_dir: Experiment directory path
        logger: Logger instance
    """
    # Create summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'total_configs': len(results),
        'successful': sum(1 for r in results if r['status'] == 'success'),
        'failed': sum(1 for r in results if r['status'] == 'failed'),
        'baseline_params': calculate_params(128, 3),
        'configurations': []
    }

    baseline_params = summary['baseline_params']

    for result in sorted(results, key=lambda x: x['config_id']):
        cfg_summary = {
            'config_id': result['config_id'],
            'num_bins': result['num_bins'],
            'num_kernels': result['num_kernels'],
            'total_params': result['total_params'],
            'param_reduction_pct': (1 - result['total_params'] / baseline_params) * 100,
            'status': result['status'],
            'duration_seconds': result.get('duration_seconds', 0)
        }

        if result['status'] == 'success' and 'hit_rates' in result:
            cfg_summary['hit_rates'] = {
                k: v['hit_rate'] for k, v in result['hit_rates'].items()
            }
            cfg_summary['detailed_hit_rates'] = result['hit_rates']

        summary['configurations'].append(cfg_summary)

    # Save summary
    summary_file = exp_dir / 'output' / 'ablation_summary.json'
    summary_file.parent.mkdir(parents=True, exist_ok=True)

    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)

    logger.info(f"\nAblation summary saved to: {summary_file}")

    # Print summary table
    logger.info("\n" + "=" * 80)
    logger.info("ABLATION STUDY RESULTS SUMMARY")
    logger.info("=" * 80)

    logger.info(f"\n{'Bins':<6} {'Kernels':<8} {'Params':<12} {'Reduction':<10} {'Hit@50':<10} {'Hit@500':<10} {'Hit@1000':<10}")
    logger.info("-" * 80)

    for cfg in summary['configurations']:
        if cfg['status'] == 'success':
            hr = cfg.get('hit_rates', {})
            logger.info(
                f"{cfg['num_bins']:<6} {cfg['num_kernels']:<8} "
                f"{cfg['total_params']:<12,} {cfg['param_reduction_pct']:<10.1f}% "
                f"{hr.get('50', 0):<10.2f} {hr.get('500', 0):<10.2f} {hr.get('1000', 0):<10.2f}"
            )
        else:
            logger.info(
                f"{cfg['num_bins']:<6} {cfg['num_kernels']:<8} "
                f"{cfg['total_params']:<12,} {cfg['param_reduction_pct']:<10.1f}% "
                f"{'FAILED':<10}"
            )

    return summary


def main():
    parser = argparse.ArgumentParser(description='Ablation Study: Bins & Kernels Parameter Sweep')
    parser.add_argument(
        '--bins',
        type=str,
        default='128,64,32,16,8',
        help='Comma-separated list of num_bins values (default: 128,64,32,16,8)'
    )
    parser.add_argument(
        '--kernels',
        type=str,
        default='3,1',
        help='Comma-separated list of num_kernels values (default: 3,1)'
    )
    parser.add_argument(
        '--gpus',
        type=str,
        default='0',
        help='Comma-separated list of GPU IDs to use (default: 0)'
    )
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Show configurations without running experiments'
    )
    args = parser.parse_args()

    # Parse arguments
    bins_list = [int(x) for x in args.bins.split(',')]
    kernels_list = [int(x) for x in args.kernels.split(',')]
    gpu_ids = [int(x) for x in args.gpus.split(',')]

    # Setup paths
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    # Load base config
    with open(config_path, 'r') as f:
        base_config = yaml.safe_load(f)

    # Setup logging
    output_dir = exp_dir / 'output'
    output_dir.mkdir(parents=True, exist_ok=True)
    log_file = output_dir / 'ablation_main.log'
    logger = setup_logging(log_file)

    logger.info("=" * 60)
    logger.info("Ablation Study: Bins & Kernels Parameter Sweep")
    logger.info("=" * 60)
    logger.info(f"Bins to test: {bins_list}")
    logger.info(f"Kernels to test: {kernels_list}")
    logger.info(f"GPUs available: {gpu_ids}")

    try:
        # Run ablation
        results = run_ablation(
            bins_list, kernels_list, gpu_ids,
            base_config, exp_dir, logger,
            dry_run=args.dry_run
        )

        if results:
            # Aggregate results
            aggregate_results(results, exp_dir, logger)

            logger.info("\n" + "=" * 60)
            logger.info("Ablation study completed successfully!")
            logger.info("=" * 60)

    except Exception as e:
        logger.error(f"Ablation study failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
