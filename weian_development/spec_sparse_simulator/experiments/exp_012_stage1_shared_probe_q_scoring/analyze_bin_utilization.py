"""
Bin Utilization Analysis for Module 2

Analyzes:
1. Key Network: How keys are distributed across bins (per round)
2. Query Network: How queries route to bins
3. Bin collapse detection: Are only a few bins used?
4. Cross-round consistency: Do bins have consistent utilization?

Metrics:
- Entropy (higher = more uniform distribution)
- Gini coefficient (higher = more inequality)
- Effective number of bins (exp(entropy))
- Top-K bin concentration
"""

import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np

from model import Module2Network, create_model


def setup_logging():
    """Setup logging configuration."""
    exp_dir = Path(__file__).parent
    log_dir = exp_dir / 'output' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'analyze_bin_utilization.log'

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


def load_config():
    """Load experiment configuration."""
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    return config


def load_trace_data(config, logger, use_test=True):
    """Load trace data from qk.pt file."""
    exp_dir = Path(__file__).parent

    if use_test and 'test_trace_path' in config['data']:
        trace_path = exp_dir / config['data']['test_trace_path']
        logger.info("Using TEST trace for analysis")
    else:
        trace_path = exp_dir / config['data']['trace_path']
        logger.info("Using TRAINING trace for analysis")

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    logger.info(f"Loading trace data from: {trace_path}")
    qk_data = torch.load(trace_path, map_location='cpu')

    # Load head sample file
    head_sample_path = exp_dir / config['data']['head_sample_file']
    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    Q = qk_data['q'][layer, head]
    K = qk_data['k'][layer, head]

    seq_len, head_dim = Q.shape
    logger.info(f"Trace shape: Q={Q.shape}, K={K.shape}")

    return {
        'Q': Q,
        'K': K,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'layer': layer,
        'head': head
    }


def load_checkpoint(config, device, logger):
    """Load model from checkpoint."""
    exp_dir = Path(__file__).parent
    checkpoint_path = exp_dir / config['output']['checkpoints_dir'] / 'best_model.pt'

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}")

    return model


def compute_entropy(counts):
    """Compute entropy from counts (higher = more uniform)."""
    if counts.sum() == 0:
        return 0.0
    probs = counts / counts.sum()
    probs = probs[probs > 0]  # Remove zeros
    return -np.sum(probs * np.log(probs))


def compute_gini(counts):
    """Compute Gini coefficient (0 = perfect equality, 1 = maximum inequality)."""
    if counts.sum() == 0:
        return 0.0
    sorted_counts = np.sort(counts)
    n = len(sorted_counts)
    cumulative = np.cumsum(sorted_counts)
    gini = (n + 1 - 2 * np.sum(cumulative) / cumulative[-1]) / n
    return gini


def analyze_bin_utilization(model, trace_data, round_window, exclude_tail, device, logger):
    """
    Comprehensive bin utilization analysis.

    Returns:
    - Key network bin assignment statistics
    - Query network bin routing statistics
    - Per-round breakdown
    - Cross-round consistency metrics
    """
    model.eval()

    Q = trace_data['Q'].to(device)
    K = trace_data['K'].to(device)
    seq_len = trace_data['seq_len']
    num_bins = model.num_bins

    valid_end = min(seq_len - exclude_tail, seq_len)

    # Global counters
    query_bin_counts = np.zeros(num_bins, dtype=np.int64)
    key_bin_counts = np.zeros(num_bins, dtype=np.int64)  # argmax bin per key

    # Per-round statistics
    round_stats = []

    # Key assignment details (which bin each key is assigned to, per round)
    # This tracks the argmax bin assignment for each key
    key_assignments_per_round = []

    with torch.no_grad():
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            if round_start == 0:
                continue

            historical_keys = K[:round_start]
            num_historical = round_start

            reference_angles = model.compute_reference_angles(round_start, round_window)

            # Get key probabilities
            key_probs = model.forward_keys(historical_keys, reference_angles)  # (num_keys, num_bins)

            # Key bin assignment: argmax bin for each key
            # key_probs has softmax over keys (dim=0), so each column sums to 1
            # For each key, its "score" in each bin is key_probs[key, bin]
            # The key is assigned to its highest-scoring bin
            key_argmax_bins = key_probs.argmax(dim=1).cpu().numpy()  # (num_keys,)

            round_key_bin_counts = np.bincount(key_argmax_bins, minlength=num_bins)
            key_assignments_per_round.append(round_key_bin_counts)

            # Accumulate global key counts (use the last round's assignment for total)
            # Actually, for global we should look at how keys are distributed in the final context
            # Let's track per-round and also do a weighted average

            # Query routing for this round
            round_query_bin_counts = np.zeros(num_bins, dtype=np.int64)

            for q_idx in range(round_start, min(round_end, valid_end)):
                query = Q[q_idx:q_idx + 1]

                empty_bin_mask = key_probs.sum(dim=0) == 0
                query_bin_probs = model.forward_queries(query, reference_angles, empty_bin_mask)

                selected_bin = query_bin_probs.argmax(dim=-1).item()
                round_query_bin_counts[selected_bin] += 1
                query_bin_counts[selected_bin] += 1

            # Compute round statistics
            round_stat = {
                'round_start': round_start,
                'num_historical_keys': num_historical,
                'num_queries': round_query_bin_counts.sum(),
                'key_entropy': compute_entropy(round_key_bin_counts),
                'key_gini': compute_gini(round_key_bin_counts),
                'query_entropy': compute_entropy(round_query_bin_counts),
                'query_gini': compute_gini(round_query_bin_counts),
                'key_bins_used': np.sum(round_key_bin_counts > 0),
                'query_bins_used': np.sum(round_query_bin_counts > 0),
                'key_bin_counts': round_key_bin_counts.tolist(),
                'query_bin_counts': round_query_bin_counts.tolist(),
            }
            round_stats.append(round_stat)

    # Compute global statistics
    # For keys: use the last round's distribution (full context)
    if key_assignments_per_round:
        key_bin_counts = key_assignments_per_round[-1]

    global_stats = {
        'num_bins': num_bins,
        'total_queries': int(query_bin_counts.sum()),
        'total_keys_in_final_round': int(key_bin_counts.sum()),

        # Key network metrics
        'key_entropy': compute_entropy(key_bin_counts),
        'key_max_entropy': np.log(num_bins),  # Maximum possible entropy
        'key_effective_bins': np.exp(compute_entropy(key_bin_counts)),
        'key_gini': compute_gini(key_bin_counts),
        'key_bins_used': int(np.sum(key_bin_counts > 0)),
        'key_empty_bins': int(np.sum(key_bin_counts == 0)),

        # Query network metrics
        'query_entropy': compute_entropy(query_bin_counts),
        'query_max_entropy': np.log(num_bins),
        'query_effective_bins': np.exp(compute_entropy(query_bin_counts)),
        'query_gini': compute_gini(query_bin_counts),
        'query_bins_used': int(np.sum(query_bin_counts > 0)),
        'query_empty_bins': int(np.sum(query_bin_counts == 0)),

        # Raw counts
        'key_bin_counts': key_bin_counts.tolist(),
        'query_bin_counts': query_bin_counts.tolist(),
    }

    # Top-K bin concentration
    for topk in [5, 10, 20]:
        # Keys
        top_key_bins = np.argsort(key_bin_counts)[-topk:]
        top_key_count = key_bin_counts[top_key_bins].sum()
        global_stats[f'key_top{topk}_concentration'] = float(top_key_count / key_bin_counts.sum()) if key_bin_counts.sum() > 0 else 0.0

        # Queries
        top_query_bins = np.argsort(query_bin_counts)[-topk:]
        top_query_count = query_bin_counts[top_query_bins].sum()
        global_stats[f'query_top{topk}_concentration'] = float(top_query_count / query_bin_counts.sum()) if query_bin_counts.sum() > 0 else 0.0

    return global_stats, round_stats


def main():
    """Main entry point for bin utilization analysis."""
    logger = setup_logging()
    config = load_config()

    logger.info("=" * 60)
    logger.info("Bin Utilization Analysis")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    trace_data = load_trace_data(config, logger, use_test=True)
    model = load_checkpoint(config, device, logger)

    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    logger.info(f"\nAnalyzing bin utilization...")

    global_stats, round_stats = analyze_bin_utilization(
        model, trace_data, round_window, exclude_tail, device, logger
    )

    # Print summary
    logger.info("\n" + "=" * 60)
    logger.info("GLOBAL BIN UTILIZATION STATISTICS")
    logger.info("=" * 60)

    logger.info(f"\n--- Configuration ---")
    logger.info(f"Number of bins: {global_stats['num_bins']}")
    logger.info(f"Total queries analyzed: {global_stats['total_queries']}")
    logger.info(f"Total keys in final round: {global_stats['total_keys_in_final_round']}")

    logger.info(f"\n--- Key Network (Bin Assignment) ---")
    logger.info(f"Bins used: {global_stats['key_bins_used']} / {global_stats['num_bins']} ({100*global_stats['key_bins_used']/global_stats['num_bins']:.1f}%)")
    logger.info(f"Empty bins: {global_stats['key_empty_bins']}")
    logger.info(f"Entropy: {global_stats['key_entropy']:.4f} / {global_stats['key_max_entropy']:.4f} (max)")
    logger.info(f"Effective bins (exp(entropy)): {global_stats['key_effective_bins']:.2f}")
    logger.info(f"Gini coefficient: {global_stats['key_gini']:.4f} (0=equal, 1=max inequality)")
    logger.info(f"Top 5 bin concentration: {100*global_stats['key_top5_concentration']:.2f}%")
    logger.info(f"Top 10 bin concentration: {100*global_stats['key_top10_concentration']:.2f}%")
    logger.info(f"Top 20 bin concentration: {100*global_stats['key_top20_concentration']:.2f}%")

    logger.info(f"\n--- Query Network (Bin Routing) ---")
    logger.info(f"Bins used: {global_stats['query_bins_used']} / {global_stats['num_bins']} ({100*global_stats['query_bins_used']/global_stats['num_bins']:.1f}%)")
    logger.info(f"Empty bins: {global_stats['query_empty_bins']}")
    logger.info(f"Entropy: {global_stats['query_entropy']:.4f} / {global_stats['query_max_entropy']:.4f} (max)")
    logger.info(f"Effective bins (exp(entropy)): {global_stats['query_effective_bins']:.2f}")
    logger.info(f"Gini coefficient: {global_stats['query_gini']:.4f} (0=equal, 1=max inequality)")
    logger.info(f"Top 5 bin concentration: {100*global_stats['query_top5_concentration']:.2f}%")
    logger.info(f"Top 10 bin concentration: {100*global_stats['query_top10_concentration']:.2f}%")
    logger.info(f"Top 20 bin concentration: {100*global_stats['query_top20_concentration']:.2f}%")

    # Bin collapse detection
    logger.info(f"\n--- Bin Collapse Assessment ---")
    key_collapse = global_stats['key_effective_bins'] < global_stats['num_bins'] * 0.3
    query_collapse = global_stats['query_effective_bins'] < global_stats['num_bins'] * 0.3

    if key_collapse:
        logger.info(f"WARNING: Key network shows bin collapse! Effective bins ({global_stats['key_effective_bins']:.1f}) < 30% of total ({global_stats['num_bins']})")
    else:
        logger.info(f"Key network: No significant bin collapse detected")

    if query_collapse:
        logger.info(f"WARNING: Query network shows bin collapse! Effective bins ({global_stats['query_effective_bins']:.1f}) < 30% of total ({global_stats['num_bins']})")
    else:
        logger.info(f"Query network: No significant bin collapse detected")

    # Top bins detail
    logger.info(f"\n--- Top 10 Most Used Bins ---")
    key_counts = np.array(global_stats['key_bin_counts'])
    query_counts = np.array(global_stats['query_bin_counts'])

    top_key_bins = np.argsort(key_counts)[-10:][::-1]
    logger.info(f"Key Network:")
    for rank, bin_idx in enumerate(top_key_bins):
        logger.info(f"  #{rank+1}: Bin {bin_idx}, {key_counts[bin_idx]} keys ({100*key_counts[bin_idx]/key_counts.sum():.2f}%)")

    top_query_bins = np.argsort(query_counts)[-10:][::-1]
    logger.info(f"Query Network:")
    for rank, bin_idx in enumerate(top_query_bins):
        logger.info(f"  #{rank+1}: Bin {bin_idx}, {query_counts[bin_idx]} queries ({100*query_counts[bin_idx]/query_counts.sum():.2f}%)")

    # Check key-query alignment (do queries route to bins that have keys?)
    logger.info(f"\n--- Key-Query Bin Alignment ---")
    bins_with_keys = set(np.where(key_counts > 0)[0])
    bins_with_queries = set(np.where(query_counts > 0)[0])
    overlap = bins_with_keys & bins_with_queries
    queries_to_empty_bins = sum(query_counts[i] for i in bins_with_queries - bins_with_keys)

    logger.info(f"Bins with both keys and queries: {len(overlap)}")
    logger.info(f"Queries routed to empty bins: {queries_to_empty_bins} ({100*queries_to_empty_bins/query_counts.sum():.2f}%)")

    # Save results
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / 'output' / 'analysis'
    results_dir.mkdir(parents=True, exist_ok=True)

    # Convert numpy types to Python types for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(i) for i in obj]
        return obj

    results = {
        'global_stats': convert_to_serializable(global_stats),
        'round_stats': convert_to_serializable(round_stats)
    }

    results_file = results_dir / 'bin_utilization_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info("=" * 60)
    logger.info("Analysis completed")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
