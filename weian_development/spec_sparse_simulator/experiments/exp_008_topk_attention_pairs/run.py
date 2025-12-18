"""
Module 2 Experiment Runner with Top-K Attention Pairs Ablation

Modes:
- train: Train the model on single trace
- evaluate: Evaluate TopK Hit Rate for K=50/500/1000
- all: Run train then evaluate
- ablation: Run ablation study sweeping lambda_activation values
- ablation_topk: Run ablation study sweeping topk_gt values (1, 3, 5, 10)

Usage:
    python run.py --mode train
    python run.py --mode evaluate
    python run.py --mode all
    python run.py --mode ablation
    python run.py --mode ablation_topk
    python run.py --mode ablation_topk --topk_values 1,3,5,10
"""

import argparse
import copy
import json
import logging
import sys
from pathlib import Path

import yaml


def setup_logging(config, mode):
    """Setup logging configuration."""
    exp_dir = Path(__file__).parent
    log_dir = exp_dir / config['output']['logs_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'run_{mode}.log'

    handlers = [
        logging.FileHandler(log_file),
        logging.StreamHandler(sys.stdout)
    ]

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )

    return logging.getLogger(__name__)


def run_train(config, logger):
    """Run training mode."""
    logger.info("=" * 50)
    logger.info("Starting TRAINING mode")
    logger.info("=" * 50)

    from train import train
    checkpoint_path = train(config, logger)

    logger.info(f"Training completed. Checkpoint: {checkpoint_path}")
    return checkpoint_path


def run_evaluate(config, logger, checkpoint_path=None):
    """Run evaluation mode."""
    logger.info("=" * 50)
    logger.info("Starting EVALUATION mode")
    logger.info("=" * 50)

    from evaluate import evaluate

    # Determine checkpoint path
    exp_dir = Path(__file__).parent
    if checkpoint_path is None:
        checkpoint_path = exp_dir / config['output']['checkpoints_dir'] / 'best_model.pt'

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        checkpoints_dir = exp_dir / config['output']['checkpoints_dir']
        if checkpoints_dir.exists():
            available = list(checkpoints_dir.glob('*.pt'))
            if available:
                logger.info("Available checkpoints:")
                for ckpt in available:
                    logger.info(f"  - {ckpt.name}")
            else:
                logger.info("No checkpoints found. Run training first.")
        return None

    results = evaluate(config, checkpoint_path, logger)
    logger.info("Evaluation completed")
    return results


def run_ablation(config, logger, lambda_values=None):
    """
    Run ablation study sweeping lambda_activation values.

    Args:
        config: Base configuration dict
        logger: Logger instance
        lambda_values: List of lambda_activation values to test (default: [0.0, 0.01, 0.05, 0.1, 0.5])

    Returns:
        dict: Results for each lambda value
    """
    from train import train
    from evaluate import evaluate

    logger.info("=" * 50)
    logger.info("Starting ABLATION mode")
    logger.info("=" * 50)

    if lambda_values is None:
        lambda_values = [0.0, 0.01, 0.05, 0.1, 0.5]

    exp_dir = Path(__file__).parent
    results_dir = exp_dir / config['output']['results_dir'] / 'ablation'
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for i, lambda_val in enumerate(lambda_values):
        exp_name = f"lambda_{lambda_val:.2f}".replace('.', '_')
        logger.info("=" * 50)
        logger.info(f"Ablation experiment {i+1}/{len(lambda_values)}: lambda_activation={lambda_val}")
        logger.info("=" * 50)

        # Create modified config for this run
        run_config = copy.deepcopy(config)
        run_config['training']['lambda_activation'] = lambda_val

        # Use separate output directories for each run
        run_config['output']['checkpoints_dir'] = f"output/ablation/{exp_name}/checkpoints"
        run_config['output']['logs_dir'] = f"output/ablation/{exp_name}/logs"

        # Create directories
        (exp_dir / run_config['output']['checkpoints_dir']).mkdir(parents=True, exist_ok=True)
        (exp_dir / run_config['output']['logs_dir']).mkdir(parents=True, exist_ok=True)

        try:
            # Train
            logger.info(f"Training with lambda_activation={lambda_val}...")
            checkpoint_path = train(run_config, logger)

            # Evaluate
            logger.info(f"Evaluating with lambda_activation={lambda_val}...")
            eval_results = evaluate(run_config, checkpoint_path, logger)

            # Store results
            all_results[exp_name] = {
                'lambda_activation': lambda_val,
                'checkpoint_path': str(checkpoint_path),
                'evaluation': eval_results
            }

            logger.info(f"Completed experiment: lambda_activation={lambda_val}")
            if eval_results:
                for k, metrics in eval_results.items():
                    logger.info(f"  K={k}: Hit Rate = {metrics.get('hit_rate', 'N/A')}")

        except Exception as e:
            logger.error(f"Experiment failed for lambda={lambda_val}: {e}", exc_info=True)
            all_results[exp_name] = {
                'lambda_activation': lambda_val,
                'error': str(e)
            }

    # Save aggregated results
    results_path = results_dir / 'ablation_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Ablation results saved to: {results_path}")

    # Print summary table
    logger.info("=" * 50)
    logger.info("ABLATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"{'Lambda':<10} {'K=50 Hit%':<12} {'K=500 Hit%':<12} {'K=1000 Hit%':<12}")
    logger.info("-" * 50)

    for exp_name, result in all_results.items():
        if 'error' in result:
            logger.info(f"{result['lambda_activation']:<10} ERROR: {result['error']}")
        elif result.get('evaluation'):
            eval_data = result['evaluation']
            k50 = eval_data.get(50, {}).get('hit_rate', 'N/A')
            k500 = eval_data.get(500, {}).get('hit_rate', 'N/A')
            k1000 = eval_data.get(1000, {}).get('hit_rate', 'N/A')

            k50_str = f"{k50*100:.2f}%" if isinstance(k50, (int, float)) else str(k50)
            k500_str = f"{k500*100:.2f}%" if isinstance(k500, (int, float)) else str(k500)
            k1000_str = f"{k1000*100:.2f}%" if isinstance(k1000, (int, float)) else str(k1000)

            logger.info(f"{result['lambda_activation']:<10} {k50_str:<12} {k500_str:<12} {k1000_str:<12}")

    return all_results


def run_topk_ablation(config, logger, topk_values=None):
    """
    Run ablation study sweeping topk_gt values.

    Args:
        config: Base configuration dict
        logger: Logger instance
        topk_values: List of topk_gt values to test (default: [1, 3, 5, 10])

    Returns:
        dict: Results for each topk_gt value
    """
    from train import train
    from evaluate import evaluate

    logger.info("=" * 50)
    logger.info("Starting TOP-K ABLATION mode")
    logger.info("=" * 50)

    if topk_values is None:
        topk_values = [1, 3, 5, 10]

    exp_dir = Path(__file__).parent
    results_dir = exp_dir / config['output']['results_dir'] / 'ablation_topk'
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    for i, topk_val in enumerate(topk_values):
        exp_name = f"topk_{topk_val}"
        logger.info("=" * 50)
        logger.info(f"Top-K Ablation experiment {i+1}/{len(topk_values)}: topk_gt={topk_val}")
        logger.info("=" * 50)

        # Create modified config for this run
        run_config = copy.deepcopy(config)
        run_config['training']['topk_gt'] = topk_val

        # Use separate output directories for each run
        run_config['output']['checkpoints_dir'] = f"output/ablation_topk/{exp_name}/checkpoints"
        run_config['output']['logs_dir'] = f"output/ablation_topk/{exp_name}/logs"

        # Create directories
        (exp_dir / run_config['output']['checkpoints_dir']).mkdir(parents=True, exist_ok=True)
        (exp_dir / run_config['output']['logs_dir']).mkdir(parents=True, exist_ok=True)

        try:
            # Train
            logger.info(f"Training with topk_gt={topk_val}...")
            checkpoint_path = train(run_config, logger)

            # Evaluate
            logger.info(f"Evaluating with topk_gt={topk_val}...")
            eval_results = evaluate(run_config, checkpoint_path, logger)

            # Store results
            all_results[exp_name] = {
                'topk_gt': topk_val,
                'checkpoint_path': str(checkpoint_path),
                'evaluation': eval_results
            }

            # Save individual results
            individual_results_path = exp_dir / f"output/ablation_topk/{exp_name}/results.json"
            with open(individual_results_path, 'w') as f:
                json.dump(all_results[exp_name], f, indent=2, default=str)

            logger.info(f"Completed experiment: topk_gt={topk_val}")
            if eval_results:
                for k, metrics in eval_results.items():
                    logger.info(f"  K={k}: Hit Rate = {metrics.get('hit_rate', 'N/A')}")

        except Exception as e:
            logger.error(f"Experiment failed for topk_gt={topk_val}: {e}", exc_info=True)
            all_results[exp_name] = {
                'topk_gt': topk_val,
                'error': str(e)
            }

    # Save aggregated results
    results_path = results_dir / 'ablation_topk_results.json'
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"Top-K Ablation results saved to: {results_path}")

    # Print summary table
    logger.info("=" * 50)
    logger.info("TOP-K ABLATION SUMMARY")
    logger.info("=" * 50)
    logger.info(f"{'TopK_GT':<10} {'K=50 Hit%':<12} {'K=500 Hit%':<12} {'K=1000 Hit%':<12}")
    logger.info("-" * 50)

    for exp_name, result in all_results.items():
        if 'error' in result:
            logger.info(f"{result['topk_gt']:<10} ERROR: {result['error']}")
        elif result.get('evaluation'):
            eval_data = result['evaluation']
            k50 = eval_data.get(50, {}).get('hit_rate', 'N/A')
            k500 = eval_data.get(500, {}).get('hit_rate', 'N/A')
            k1000 = eval_data.get(1000, {}).get('hit_rate', 'N/A')

            k50_str = f"{k50*100:.2f}%" if isinstance(k50, (int, float)) else str(k50)
            k500_str = f"{k500*100:.2f}%" if isinstance(k500, (int, float)) else str(k500)
            k1000_str = f"{k1000*100:.2f}%" if isinstance(k1000, (int, float)) else str(k1000)

            logger.info(f"{result['topk_gt']:<10} {k50_str:<12} {k500_str:<12} {k1000_str:<12}")

    return all_results


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Module 2 Experiment Runner with Top-K Attention Pairs Ablation')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'all', 'ablation', 'ablation_topk'],
        default='all',
        help='Execution mode: train, evaluate, all, ablation, or ablation_topk (default: all)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation (relative to experiment dir)'
    )
    parser.add_argument(
        '--lambdas',
        type=str,
        default=None,
        help='Comma-separated lambda_activation values for ablation (default: 0.0,0.01,0.05,0.1,0.5)'
    )
    parser.add_argument(
        '--topk_values',
        type=str,
        default=None,
        help='Comma-separated topk_gt values for ablation_topk (default: 1,3,5,10)'
    )
    args = parser.parse_args()

    # Load configuration
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config, args.mode)

    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Mode: {args.mode}")

    try:
        if args.mode == 'train':
            run_train(config, logger)

        elif args.mode == 'evaluate':
            checkpoint_path = None
            if args.checkpoint:
                checkpoint_path = exp_dir / args.checkpoint
            run_evaluate(config, logger, checkpoint_path)

        elif args.mode == 'all':
            # Train first
            checkpoint_path = run_train(config, logger)

            # Then evaluate
            run_evaluate(config, logger, checkpoint_path)

        elif args.mode == 'ablation':
            # Parse lambda values
            lambda_values = None
            if args.lambdas:
                lambda_values = [float(x.strip()) for x in args.lambdas.split(',')]

            run_ablation(config, logger, lambda_values)

        elif args.mode == 'ablation_topk':
            # Parse topk values
            topk_values = None
            if args.topk_values:
                topk_values = [int(x.strip()) for x in args.topk_values.split(',')]

            run_topk_ablation(config, logger, topk_values)

        logger.info("=" * 50)
        logger.info("Experiment completed successfully")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
