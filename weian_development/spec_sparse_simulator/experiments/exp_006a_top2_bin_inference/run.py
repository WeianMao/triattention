"""
Module 2 Experiment Runner

Modes:
- train: Train the model on single trace
- evaluate: Evaluate TopK Hit Rate for K=50/500/1000
- all: Run train then evaluate

Usage:
    python run.py --mode train
    python run.py --mode evaluate
    python run.py --mode all
"""

import argparse
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


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(description='Module 2 Experiment Runner')
    parser.add_argument(
        '--mode',
        type=str,
        choices=['train', 'evaluate', 'all'],
        default='all',
        help='Execution mode: train, evaluate, or all (default: all)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint for evaluation (relative to experiment dir)'
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

        logger.info("=" * 50)
        logger.info("Experiment completed successfully")
        logger.info("=" * 50)

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
