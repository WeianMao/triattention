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
from analyze_miss_cases import analyze_miss_cases, load_trace_data as load_trace_data_for_analysis


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

    # Override num_bins for this experiment
    # Set to 1 to test single-bin case (no bin partitioning, pure key scoring)
    num_bins = 128  # <-- MODIFY THIS TO TEST DIFFERENT BIN COUNTS
    logger.info(f"Using num_bins = {num_bins} (override from config's {config['model']['num_bins']})")

    # L2 normalization toggle
    # Set to False to test without L2 normalization (preserve amplitude info like Hybrid Frequency)
    use_l2_norm = False  # <-- MODIFY THIS TO TEST WITH/WITHOUT L2 NORM
    logger.info(f"Using L2 normalization: {use_l2_norm}")

    # Invert to origin toggle
    # Set to True to invert each Q to position 0 (like Hybrid Frequency)
    # Set to False to rotate all Q in round to reference position (original Module 2 behavior)
    invert_to_origin = False  # <-- MODIFY THIS TO TEST HYBRID FREQ STYLE INIT
    logger.info(f"Using invert_to_origin: {invert_to_origin}")

    # Probe initialization method toggle
    # "kmeans_center": Use K-means cluster centers directly
    # "class_mean": Use K-means labels, then compute mean of unnormalized vectors per class
    # "l2_cluster_unnorm_mean": K-means on L2-normalized vectors, but init with unnormalized class means
    probe_init_method = "kmeans_center"  # <-- MODIFY THIS TO TEST DIFFERENT INIT METHODS
    logger.info(f"Using probe_init_method: {probe_init_method}")

    # Create a modified config for model creation
    test_config = config.copy()
    test_config['model'] = config['model'].copy()
    test_config['model']['num_bins'] = num_bins

    # Compute K-means + magnitude initialization
    logger.info("Computing K-means initialization with magnitude init...")
    init_probes, cluster_labels, Q_relatives_unnorm = compute_kmeans_init(
        config, logger, n_clusters=num_bins, return_extras=True,
        use_l2_norm=use_l2_norm, invert_to_origin=invert_to_origin
    )
    logger.info(f"K-means cluster centers shape: {init_probes.shape}")

    # Choose probe initialization method
    if probe_init_method == "class_mean":
        # Compute class mean from unnormalized vectors (same clustering as above)
        logger.info("Computing class means from unnormalized vectors...")
        import numpy as np
        cluster_labels_tensor = torch.from_numpy(cluster_labels) if isinstance(cluster_labels, np.ndarray) else cluster_labels
        class_means = torch.zeros_like(init_probes)
        for b in range(num_bins):
            mask = cluster_labels_tensor == b
            if mask.sum() > 0:
                class_means[b] = Q_relatives_unnorm[mask].mean(dim=0)
        init_probes = class_means
        logger.info(f"Using class mean initialization. Probe shape: {init_probes.shape}")
    elif probe_init_method == "l2_cluster_unnorm_mean":
        # K-means on L2-normalized vectors, but init probes with unnormalized class means
        logger.info("Re-running K-means on L2-normalized vectors...")
        from sklearn.cluster import KMeans
        import numpy as np

        # L2 normalize for clustering
        norm = torch.norm(Q_relatives_unnorm, p=2, dim=-1, keepdim=True)
        Q_relatives_l2 = Q_relatives_unnorm / (norm + 1e-8)

        # Run K-means on L2-normalized vectors
        kmeans_l2 = KMeans(n_clusters=num_bins, random_state=42, n_init=10)
        kmeans_l2.fit(Q_relatives_l2.numpy())
        l2_labels = kmeans_l2.labels_

        # Compute unnormalized class means using L2-based labels
        l2_labels_tensor = torch.from_numpy(l2_labels)
        class_means = torch.zeros(num_bins, Q_relatives_unnorm.shape[1], dtype=Q_relatives_unnorm.dtype)
        for b in range(num_bins):
            mask = l2_labels_tensor == b
            if mask.sum() > 0:
                class_means[b] = Q_relatives_unnorm[mask].mean(dim=0)
        init_probes = class_means
        cluster_labels = l2_labels  # Update labels for magnitude init
        logger.info(f"Using L2-clustered unnormalized mean initialization. Probe shape: {init_probes.shape}")
    else:
        logger.info(f"Using K-means cluster center initialization. Probe shape: {init_probes.shape}")

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

    # =========================================================================
    # END INITIALIZATION SECTION
    # =========================================================================

    model = model.to(device)
    model.eval()

    # Save initialized model for verification
    exp_dir = Path(__file__).parent
    init_checkpoint_path = exp_dir / 'output' / 'checkpoints' / 'init_only_model.pt'
    init_checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        'epoch': 0,
        'model_state_dict': model.state_dict(),
        'loss': 0.0,
        'config': test_config,
        'note': 'Initialization only, no training'
    }, init_checkpoint_path)
    logger.info(f"Saved initialized model to: {init_checkpoint_path}")

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

    # =========================================================================
    # MISS CASE ANALYSIS SECTION
    # =========================================================================
    import numpy as np

    logger.info("\n" + "=" * 60)
    logger.info("MISS CASE ANALYSIS (K=50)")
    logger.info("=" * 60)

    # Load trace data in the format expected by analyze_miss_cases
    trace_data_for_analysis = load_trace_data_for_analysis(config, logger, use_test=True)

    K_val = 50
    stats, miss_cases, best_ranks, selected_ranks = analyze_miss_cases(
        model, trace_data_for_analysis, K_val, round_window, exclude_tail, device, logger
    )

    # Print summary
    total = stats['total_queries']
    logger.info(f"Total queries: {total}")
    logger.info(f"Recent hits: {stats['recent_hits']} ({100*stats['recent_hits']/total:.2f}%)")
    logger.info(f"Bin hits: {stats['bin_hits']} ({100*stats['bin_hits']/total:.2f}%)")
    logger.info(f"Misses: {stats['misses']} ({100*stats['misses']/total:.2f}%)")

    if stats['misses'] > 0:
        logger.info(f"\n--- Miss Type Breakdown ---")
        logger.info(f"Type A (Key Network - argmax ranks poorly in ALL bins): "
                   f"{stats['type_a_misses']} ({100*stats['type_a_misses']/stats['misses']:.2f}% of misses)")
        logger.info(f"Type B (Query Network - argmax ranks well in another bin): "
                   f"{stats['type_b_misses']} ({100*stats['type_b_misses']/stats['misses']:.2f}% of misses)")

        best_ranks_arr = np.array(best_ranks)
        logger.info(f"\n--- Best Rank Distribution (for missed cases) ---")
        thresholds = [50, 100, 200, 500, 1000]
        for t in thresholds:
            count = np.sum(best_ranks_arr < t)
            logger.info(f"  Best rank < {t}: {count} ({100*count/len(best_ranks_arr):.2f}%)")


if __name__ == '__main__':
    main()
