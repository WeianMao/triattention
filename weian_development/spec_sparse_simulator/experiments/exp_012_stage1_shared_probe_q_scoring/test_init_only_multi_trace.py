"""
Test multi-trace initialization only (no training) performance.

This script evaluates the model with K-means initialization from ALL training traces
without any training, to compare with single-trace initialization.
"""
import sys
from pathlib import Path
import logging

import torch
import yaml

from model import create_model
from compute_kmeans_init import compute_magnitude_init
from compute_kmeans_init_multi_trace_v2 import compute_kmeans_init_multi_trace
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

    # Fixed settings
    num_bins = 128
    use_l2_norm = False
    invert_to_origin = False

    logger.info(f"Using num_bins = {num_bins}")
    logger.info(f"Using L2 normalization: {use_l2_norm}")
    logger.info(f"Using invert_to_origin: {invert_to_origin}")

    # Create a modified config for model creation
    test_config = config.copy()
    test_config['model'] = config['model'].copy()
    test_config['model']['num_bins'] = num_bins

    # Compute K-means + magnitude initialization from ALL training traces
    logger.info("Computing K-means initialization from ALL training traces...")
    init_probes, cluster_labels, Q_relatives_unnorm = compute_kmeans_init_multi_trace(
        config, logger, n_clusters=num_bins, return_extras=True,
        use_l2_norm=use_l2_norm, invert_to_origin=invert_to_origin
    )
    logger.info(f"K-means cluster centers shape: {init_probes.shape}")

    num_freqs = config['model'].get('num_freqs', 64)
    magnitude_init = compute_magnitude_init(
        Q_relatives_unnorm, cluster_labels, num_bins, num_freqs
    )
    logger.info(f"Magnitude initialization computed. Shape: {magnitude_init.shape}")
    logger.info(f"Magnitude init stats: mean={magnitude_init.mean():.6f}, "
               f"std={magnitude_init.std():.6f}, min={magnitude_init.min():.6f}, "
               f"max={magnitude_init.max():.6f}")

    # Create model with initialization (NO TRAINING)
    model = create_model(test_config, init_probes=init_probes, use_l2_norm=use_l2_norm)

    # Apply magnitude initialization
    with torch.no_grad():
        model.key_network.k_magnitude_weights.copy_(magnitude_init)
        model.query_network.distance_scorer.q_magnitude_weights.copy_(magnitude_init)
    logger.info("Applied magnitude initialization to k_magnitude_weights and q_magnitude_weights")

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
    logger.info("MULTI-TRACE INITIALIZATION ONLY - NO TRAINING")
    logger.info("=" * 60)

    hit_rates = compute_topk_hit_rate(
        model, trace_data, K_values, round_window, exclude_tail, device, logger
    )

    logger.info("\n" + "=" * 60)
    logger.info("=== Multi-Trace Initialization Only (No Training) Results ===")
    logger.info("=" * 60)
    for k_val, metrics in hit_rates.items():
        logger.info(f"K={k_val}: {metrics['hit_rate']:.2f}% hit rate "
                   f"(Recent: {metrics['recent_hit_rate']:.2f}%, Bin: {metrics['bin_hit_rate']:.2f}%)")


if __name__ == '__main__':
    main()
