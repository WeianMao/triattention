"""
Miss Case Diagnosis Analysis for Module 2

Analyzes why certain queries miss (argmax key not in selected bin's TopK).
Classifies misses into:
- Type A (Key Network Issue): argmax key ranks poorly in ALL bins
- Type B (Query Network Issue): argmax key ranks well in some bin, but query selected wrong bin

Also analyzes the distribution of argmax key ranks across bins.
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

    log_file = log_dir / 'analyze_miss_cases.log'

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

    # Compute attention
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


def analyze_miss_cases(model, trace_data, K_val, round_window, exclude_tail, device, logger):
    """
    Detailed analysis of miss cases.

    For each miss, determines:
    - Rank of argmax key in selected bin
    - Best rank of argmax key across ALL bins
    - Which bin would have been optimal
    - Classification: Type A (key network issue) or Type B (query network issue)

    Returns detailed statistics.
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
    argmax_best_ranks = []  # Best rank across all bins for each miss
    argmax_selected_ranks = []  # Rank in selected bin for each miss

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
                    best_rank = num_historical  # worst possible
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


def main():
    """Main entry point for miss case analysis."""
    logger = setup_logging()
    config = load_config()

    logger.info("=" * 60)
    logger.info("Miss Case Diagnosis Analysis")
    logger.info("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    trace_data = load_trace_data(config, logger, use_test=True)
    model = load_checkpoint(config, device, logger)

    K_val = 50  # Analyze for K=50
    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    logger.info(f"\nAnalyzing miss cases for K={K_val}...")

    stats, miss_cases, best_ranks, selected_ranks = analyze_miss_cases(
        model, trace_data, K_val, round_window, exclude_tail, device, logger
    )

    # Print summary statistics
    logger.info("\n" + "=" * 60)
    logger.info("SUMMARY STATISTICS")
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

    # Save detailed results
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / 'output' / 'analysis'
    results_dir.mkdir(parents=True, exist_ok=True)

    results = {
        'K': K_val,
        'stats': stats,
        'best_ranks': best_ranks,
        'selected_ranks': selected_ranks,
        'miss_cases': miss_cases
    }

    results_file = results_dir / 'miss_case_analysis.json'
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nDetailed results saved to: {results_file}")
    logger.info("=" * 60)
    logger.info("Analysis completed")
    logger.info("=" * 60)


if __name__ == '__main__':
    main()
