"""
Combined Model Analysis for Module 2 v5

Combines bin utilization analysis and miss case diagnosis.

Bin Utilization Analysis:
- Key Network: How keys are distributed across bins (per round)
- Query Network: How queries route to bins
- Bin collapse detection: Are only a few bins used?

Miss Case Diagnosis:
- Type A (Key Network Issue): argmax key ranks poorly in ALL bins
- Type B (Query Network Issue): argmax key ranks well in some bin, but query selected wrong bin
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml
import numpy as np

from model_v5 import Module2Network, create_model, load_model_inv_freq


def setup_logging(log_name='analyze_model_v5'):
    """Setup logging configuration."""
    exp_dir = Path(__file__).parent
    log_dir = exp_dir / 'output' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / f'{log_name}.log'

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

    # Compute attention for miss case analysis
    scale = head_dim ** -0.5
    attention_logits = Q @ K.T * scale

    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_logits.masked_fill_(causal_mask, float('-inf'))

    attention = F.softmax(attention_logits, dim=-1)

    return {
        'Q': Q,
        'K': K,
        'attention': attention,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'layer': layer,
        'head': head
    }


def load_checkpoint_v5(config, device, logger, checkpoint_name='final_model_multi_trace_v5.pt', use_error_term=None):
    """Load model from v5 checkpoint."""
    exp_dir = Path(__file__).parent
    checkpoint_path = exp_dir / config['output']['checkpoints_dir'] / checkpoint_name

    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Auto-detect use_error_term from checkpoint if not specified
    if use_error_term is None:
        use_error_term = 'query_network.distance_scorer.q_error_weights' in checkpoint['model_state_dict']
        logger.info(f"Auto-detected use_error_term={use_error_term}")

    # Load inv_freq from model
    model_path = config.get('model', {}).get('model_path',
        "/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B")
    inv_freq = load_model_inv_freq(model_path, logger)

    # Create model with v5 API
    model = create_model(config, init_probes=None, use_l2_norm=True, inv_freq=inv_freq, use_error_term=use_error_term)
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
    """
    model.eval()

    Q = trace_data['Q'].to(device)
    K = trace_data['K'].to(device)
    seq_len = trace_data['seq_len']
    num_bins = model.num_bins

    valid_end = min(seq_len - exclude_tail, seq_len)

    # Global counters
    query_bin_counts = np.zeros(num_bins, dtype=np.int64)
    key_bin_counts = np.zeros(num_bins, dtype=np.int64)

    # Per-round statistics
    round_stats = []
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
            key_probs = model.forward_keys(historical_keys, reference_angles)

            # Key bin assignment: argmax bin for each key
            key_argmax_bins = key_probs.argmax(dim=1).cpu().numpy()
            round_key_bin_counts = np.bincount(key_argmax_bins, minlength=num_bins)
            key_assignments_per_round.append(round_key_bin_counts)

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
                'num_queries': int(round_query_bin_counts.sum()),
                'key_entropy': compute_entropy(round_key_bin_counts),
                'key_gini': compute_gini(round_key_bin_counts),
                'query_entropy': compute_entropy(round_query_bin_counts),
                'query_gini': compute_gini(round_query_bin_counts),
                'key_bins_used': int(np.sum(round_key_bin_counts > 0)),
                'query_bins_used': int(np.sum(round_query_bin_counts > 0)),
            }
            round_stats.append(round_stat)

    # Compute global statistics
    if key_assignments_per_round:
        key_bin_counts = key_assignments_per_round[-1]

    global_stats = {
        'num_bins': num_bins,
        'total_queries': int(query_bin_counts.sum()),
        'total_keys_in_final_round': int(key_bin_counts.sum()),

        # Key network metrics
        'key_entropy': compute_entropy(key_bin_counts),
        'key_max_entropy': np.log(num_bins),
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


def analyze_miss_cases(model, trace_data, K_val, round_window, exclude_tail, device, logger):
    """
    Detailed analysis of miss cases.

    Classifies misses into:
    - Type A (Key Network Issue): argmax key ranks poorly in ALL bins
    - Type B (Query Network Issue): argmax key ranks well in some bin, wrong routing
    """
    model.eval()

    Q = trace_data['Q'].to(device)
    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']
    num_bins = model.num_bins

    valid_end = min(seq_len - exclude_tail, seq_len)

    # Collect detailed miss case info
    miss_cases = []

    # Statistics
    stats = {
        'total_queries': 0,
        'recent_hits': 0,
        'bin_hits': 0,
        'misses': 0,
        'type_a_misses': 0,  # Key network issue: argmax ranks poorly everywhere
        'type_b_misses': 0,  # Query network issue: argmax ranks well in another bin
    }

    # For analyzing rank distributions
    argmax_best_ranks = []
    argmax_selected_ranks = []

    with torch.no_grad():
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            if round_start == 0:
                continue

            historical_keys = K[:round_start]
            num_historical = round_start

            reference_angles = model.compute_reference_angles(round_start, round_window)

            # Get key probabilities (used for ranking)
            key_probs = model.forward_keys(historical_keys, reference_angles)

            for q_idx in range(round_start, min(round_end, valid_end)):
                attn_weights = attention[q_idx, :q_idx + 1]
                argmax_key = attn_weights.argmax().item()

                argmax_in_recent = argmax_key >= round_start

                query = Q[q_idx:q_idx + 1]

                empty_bin_mask = key_probs.sum(dim=0) == 0
                query_bin_probs = model.forward_queries(query, reference_angles, empty_bin_mask)

                selected_bin = query_bin_probs.argmax(dim=-1).item()

                stats['total_queries'] += 1

                if argmax_in_recent:
                    stats['recent_hits'] += 1
                    continue

                # Get scores for selected bin
                bin_scores = key_probs[:, selected_bin]
                actual_k = min(K_val, num_historical)
                _, topk_indices = torch.topk(bin_scores, actual_k)

                if argmax_key in topk_indices:
                    stats['bin_hits'] += 1
                else:
                    # This is a miss - analyze it
                    stats['misses'] += 1

                    # Rank of argmax key in selected bin
                    sorted_indices = torch.argsort(bin_scores, descending=True)
                    rank_in_selected = (sorted_indices == argmax_key).nonzero(as_tuple=True)[0].item()

                    # Find best rank across ALL bins
                    best_rank = num_historical
                    best_bin = -1

                    for b in range(num_bins):
                        if empty_bin_mask[b]:
                            continue
                        bin_b_scores = key_probs[:, b]
                        sorted_b = torch.argsort(bin_b_scores, descending=True)
                        rank_in_b = (sorted_b == argmax_key).nonzero(as_tuple=True)[0].item()

                        if rank_in_b < best_rank:
                            best_rank = rank_in_b
                            best_bin = b

                    argmax_best_ranks.append(best_rank)
                    argmax_selected_ranks.append(rank_in_selected)

                    # Classification
                    # Type A: Best rank is still >= K (argmax key ranks poorly everywhere)
                    # Type B: Best rank < K (argmax key ranks well in some bin, wrong routing)
                    if best_rank >= K_val:
                        miss_type = 'A'
                        stats['type_a_misses'] += 1
                    else:
                        miss_type = 'B'
                        stats['type_b_misses'] += 1

                    miss_cases.append({
                        'query_idx': q_idx,
                        'argmax_key': argmax_key,
                        'selected_bin': selected_bin,
                        'best_bin': best_bin,
                        'rank_in_selected': rank_in_selected,
                        'best_rank': best_rank,
                        'miss_type': miss_type,
                        'num_historical': num_historical
                    })

    return stats, miss_cases, argmax_best_ranks, argmax_selected_ranks


def print_bin_utilization_summary(global_stats, logger):
    """Print bin utilization summary."""
    logger.info("\n" + "=" * 60)
    logger.info("BIN UTILIZATION STATISTICS")
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

    # Key-query alignment
    logger.info(f"\n--- Key-Query Bin Alignment ---")
    bins_with_keys = set(np.where(key_counts > 0)[0])
    bins_with_queries = set(np.where(query_counts > 0)[0])
    overlap = bins_with_keys & bins_with_queries
    queries_to_empty_bins = sum(query_counts[i] for i in bins_with_queries - bins_with_keys)

    logger.info(f"Bins with both keys and queries: {len(overlap)}")
    logger.info(f"Queries routed to empty bins: {queries_to_empty_bins} ({100*queries_to_empty_bins/query_counts.sum():.2f}%)")


def print_miss_case_summary(stats, miss_cases, best_ranks, selected_ranks, K_val, logger):
    """Print miss case summary."""
    logger.info("\n" + "=" * 60)
    logger.info(f"MISS CASE ANALYSIS (K={K_val})")
    logger.info("=" * 60)

    total = stats['total_queries']
    logger.info(f"Total queries: {total}")
    logger.info(f"Recent hits (auto): {stats['recent_hits']} ({100*stats['recent_hits']/total:.2f}%)")
    logger.info(f"Bin hits: {stats['bin_hits']} ({100*stats['bin_hits']/total:.2f}%)")
    logger.info(f"Misses: {stats['misses']} ({100*stats['misses']/total:.2f}%)")

    if stats['misses'] > 0:
        logger.info(f"\n--- Miss Type Breakdown ---")
        logger.info(f"Type A (Key Network Issue - argmax ranks poorly in ALL bins):")
        logger.info(f"  Count: {stats['type_a_misses']} ({100*stats['type_a_misses']/stats['misses']:.2f}% of misses)")
        logger.info(f"Type B (Query Network Issue - argmax ranks well in another bin):")
        logger.info(f"  Count: {stats['type_b_misses']} ({100*stats['type_b_misses']/stats['misses']:.2f}% of misses)")

        # Rank statistics
        best_ranks_arr = np.array(best_ranks)
        selected_ranks_arr = np.array(selected_ranks)

        logger.info(f"\n--- Rank Statistics (for missed cases) ---")
        logger.info(f"Best rank across all bins:")
        logger.info(f"  Min: {best_ranks_arr.min()}, Max: {best_ranks_arr.max()}")
        logger.info(f"  Mean: {best_ranks_arr.mean():.2f}, Median: {np.median(best_ranks_arr):.2f}")

        logger.info(f"Rank in selected bin:")
        logger.info(f"  Min: {selected_ranks_arr.min()}, Max: {selected_ranks_arr.max()}")
        logger.info(f"  Mean: {selected_ranks_arr.mean():.2f}, Median: {np.median(selected_ranks_arr):.2f}")

        # Distribution of best ranks
        logger.info(f"\n--- Best Rank Distribution (for missed cases) ---")
        thresholds = [50, 100, 200, 500, 1000]
        for t in thresholds:
            count = np.sum(best_ranks_arr < t)
            logger.info(f"  Best rank < {t}: {count} ({100*count/len(best_ranks_arr):.2f}%)")

        # Show some example miss cases
        logger.info(f"\n--- Sample Miss Cases (first 10) ---")
        for i, case in enumerate(miss_cases[:10]):
            logger.info(f"  Case {i+1}:")
            logger.info(f"    Query idx: {case['query_idx']}, Argmax key: {case['argmax_key']}")
            logger.info(f"    Selected bin: {case['selected_bin']}, Best bin: {case['best_bin']}")
            logger.info(f"    Rank in selected: {case['rank_in_selected']}, Best rank: {case['best_rank']}")
            logger.info(f"    Type: {case['miss_type']} ({'Key Network' if case['miss_type'] == 'A' else 'Query Network'} issue)")


def convert_to_serializable(obj):
    """Convert numpy types to Python types for JSON serialization."""
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


def main():
    """Main entry point for combined model analysis."""
    parser = argparse.ArgumentParser(description='Analyze Module 2 v5 model')
    parser.add_argument('--checkpoint', type=str, default='final_model_multi_trace_v5.pt',
                        help='Checkpoint filename to load (default: final_model_multi_trace_v5.pt)')
    parser.add_argument('--k-values', type=int, nargs='+', default=[50],
                        help='K values for miss case analysis (default: 50)')
    parser.add_argument('--skip-bin-analysis', action='store_true',
                        help='Skip bin utilization analysis')
    parser.add_argument('--skip-miss-analysis', action='store_true',
                        help='Skip miss case analysis')
    args = parser.parse_args()

    logger = setup_logging()
    config = load_config()

    logger.info("=" * 60)
    logger.info("Combined Model Analysis for Module 2 v5")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    trace_data = load_trace_data(config, logger, use_test=True)
    model = load_checkpoint_v5(config, device, logger, checkpoint_name=args.checkpoint)

    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    exp_dir = Path(__file__).parent
    results_dir = exp_dir / 'output' / 'analysis'
    results_dir.mkdir(parents=True, exist_ok=True)

    all_results = {}

    # Bin Utilization Analysis
    if not args.skip_bin_analysis:
        logger.info(f"\n{'='*60}")
        logger.info("Running Bin Utilization Analysis...")
        logger.info("=" * 60)

        global_stats, round_stats = analyze_bin_utilization(
            model, trace_data, round_window, exclude_tail, device, logger
        )

        print_bin_utilization_summary(global_stats, logger)

        all_results['bin_utilization'] = {
            'global_stats': convert_to_serializable(global_stats),
            'round_stats': convert_to_serializable(round_stats)
        }

    # Miss Case Analysis
    if not args.skip_miss_analysis:
        all_results['miss_case'] = {}

        for K_val in args.k_values:
            logger.info(f"\n{'='*60}")
            logger.info(f"Running Miss Case Analysis for K={K_val}...")
            logger.info("=" * 60)

            stats, miss_cases, best_ranks, selected_ranks = analyze_miss_cases(
                model, trace_data, K_val, round_window, exclude_tail, device, logger
            )

            print_miss_case_summary(stats, miss_cases, best_ranks, selected_ranks, K_val, logger)

            all_results['miss_case'][f'K_{K_val}'] = {
                'stats': stats,
                'best_ranks': best_ranks,
                'selected_ranks': selected_ranks,
                'miss_cases': miss_cases
            }

    # Save combined results
    results_file = results_dir / 'model_analysis_v5.json'
    with open(results_file, 'w') as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"\n{'='*60}")
    logger.info(f"All results saved to: {results_file}")
    logger.info("Analysis completed")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
