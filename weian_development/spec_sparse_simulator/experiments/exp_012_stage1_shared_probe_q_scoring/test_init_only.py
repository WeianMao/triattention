"""
Test initialization only (no training) performance.

This script evaluates the model with only initialization (K-means probes + magnitude init)
without any training, to understand how much the initialization contributes.

Usage:
    python test_init_only.py

This script is designed for rapid iteration on initialization methods.
Modify the initialization logic here without changing train.py or other scripts.
"""
import sys
from pathlib import Path
import logging

import torch
import yaml

from model import create_model
from compute_kmeans_init import compute_kmeans_init, compute_magnitude_init
from evaluate import load_trace_data, compute_topk_hit_rate


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()

    # Load config
    exp_dir = Path(__file__).parent
    with open(exp_dir / "config.yaml") as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # =========================================================================
    # INITIALIZATION SECTION - Modify this section to test different init methods
    # =========================================================================

    # Compute K-means + magnitude initialization
    logger.info("Computing K-means initialization with magnitude init...")
    init_probes, cluster_labels, Q_relatives_unnorm = compute_kmeans_init(
        config, logger, n_clusters=config['model']['num_bins'], return_extras=True
    )
    logger.info(f"K-means initialization completed. Probe shape: {init_probes.shape}")

    num_freqs = config['model'].get('num_freqs', 64)
    magnitude_init = compute_magnitude_init(
        Q_relatives_unnorm, cluster_labels, config['model']['num_bins'], num_freqs
    )
    logger.info(f"Magnitude initialization computed. Shape: {magnitude_init.shape}")
    logger.info(f"Magnitude init stats: mean={magnitude_init.mean():.6f}, "
               f"std={magnitude_init.std():.6f}, min={magnitude_init.min():.6f}, "
               f"max={magnitude_init.max():.6f}")

    # Create model with initialization (NO TRAINING)
    model = create_model(config, init_probes=init_probes)

    # Apply magnitude initialization
    with torch.no_grad():
        model.key_network.k_magnitude_weights.copy_(magnitude_init)
        model.query_network.distance_scorer.q_magnitude_weights.copy_(magnitude_init)
    logger.info("Applied magnitude initialization to k_magnitude_weights and q_magnitude_weights")

    # =========================================================================
    # END INITIALIZATION SECTION
    # =========================================================================

    model = model.to(device)
    model.eval()

    # Load test data
    logger.info("Loading test data...")
    trace_data = load_trace_data(config, logger, trace_type='test')

    # Evaluate
    K_values = config['evaluation']['topk_K']
    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    logger.info(f"Evaluating TopK Hit Rate for K={K_values}")
    logger.info("=" * 60)
    logger.info("INITIALIZATION ONLY - NO TRAINING")
    logger.info("=" * 60)

    hit_rates = compute_topk_hit_rate(
        model, trace_data, K_values, round_window, exclude_tail, device, logger
    )

    logger.info("\n" + "=" * 60)
    logger.info("=== Initialization Only (No Training) Results ===")
    logger.info("=" * 60)
    for k_val, metrics in hit_rates.items():
        logger.info(f"K={k_val}: {metrics['hit_rate']:.2f}% hit rate "
                   f"(Recent: {metrics['recent_hit_rate']:.2f}%, Bin: {metrics['bin_hit_rate']:.2f}%)")


if __name__ == '__main__':
    main()
