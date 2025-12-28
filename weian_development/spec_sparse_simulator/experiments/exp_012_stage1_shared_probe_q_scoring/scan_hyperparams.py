#!/usr/bin/env python3
"""
Hyperparameter Scanning Script for Sparse Attention Simulation.

Scans different (top_bins, keys_per_bin) combinations to find optimal settings
for a given budget (total keys per query = top_bins * keys_per_bin).

Uses the fast evaluation code from train_multi_trace_v5.py for efficient scanning.
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import torch
import torch.nn.functional as F
import yaml

from model_v5 import create_model
from train_multi_trace_v5 import (
    compute_attention_matrix,
    extract_round_labels,
    preload_test_trace,
)


def setup_logging():
    """Setup logging to stdout."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger(__name__)


def compute_hit_rate_for_config(
    model,
    test_trace_data: Dict,
    top_bins: int,
    keys_per_bin: int,
    device: torch.device,
) -> Dict:
    """
    Compute hit rate for a specific (top_bins, keys_per_bin) configuration.

    Uses vectorized operations from train_multi_trace_v5.py for efficiency.

    Args:
        model: Module2Network instance (in eval mode)
        test_trace_data: Preloaded test trace dict with Q, K, cached_rounds
        top_bins: Number of top bins to use
        keys_per_bin: Number of keys to select per bin (K value)
        device: torch.device

    Returns:
        Dict with hit rate statistics
    """
    Q = test_trace_data['Q']
    K = test_trace_data['K']
    cached_rounds = test_trace_data['cached_rounds']
    round_window = test_trace_data['round_window']

    total_hits = 0
    total_queries = 0
    recent_hits = 0
    bin_hits = 0

    # Track unique keys statistics
    total_unique_keys = 0
    total_non_recent_queries = 0

    with torch.no_grad():
        for round_info in cached_rounds:
            round_start = round_info['round_start']
            labels = round_info['labels']

            query_indices = labels['query_indices']
            argmax_keys = labels['argmax_keys'].to(device)
            argmax_in_recent = labels['argmax_in_recent'].to(device)

            num_queries = len(query_indices)
            if num_queries == 0:
                continue

            # Historical keys
            historical_keys = K[:round_start]
            num_historical = round_start

            if num_historical == 0:
                continue

            # Compute reference angles
            reference_angles = model.compute_reference_angles(round_start, round_window)

            # Forward pass: Key network (once per round)
            key_probs = model.forward_keys(historical_keys, reference_angles)

            # Detect empty bins
            empty_bin_mask = (key_probs.sum(dim=0) == 0).detach()

            # Forward pass: Query network (batched for all queries in this round)
            queries = Q[query_indices]
            query_bin_probs = model.forward_queries(queries, reference_angles, empty_bin_mask)

            # Pre-compute topk keys for all bins at once
            actual_k = min(keys_per_bin, num_historical)
            _, topk_keys_per_bin = torch.topk(key_probs, actual_k, dim=0)
            # topk_keys_per_bin: (actual_k, num_bins)

            # Get top bins for each query
            actual_top_bins = min(top_bins, model.num_bins)
            _, top_bin_indices = torch.topk(query_bin_probs, actual_top_bins, dim=1)
            # top_bin_indices: (num_queries, top_bins)

            # Count recent hits (vectorized)
            num_recent = argmax_in_recent.sum().item()
            recent_hits += num_recent
            total_hits += num_recent
            total_queries += num_queries

            # For non-recent queries, check if argmax is in selected bins' topk
            non_recent_mask = ~argmax_in_recent
            num_non_recent = non_recent_mask.sum().item()

            if num_non_recent > 0:
                non_recent_indices = torch.where(non_recent_mask)[0]
                non_recent_argmax = argmax_keys[non_recent_mask]
                non_recent_top_bins = top_bin_indices[non_recent_mask]  # (num_non_recent, top_bins)

                # Gather topk keys for selected bins
                # topk_keys_per_bin: (actual_k, num_bins)
                # non_recent_top_bins: (num_non_recent, top_bins)
                expanded_bins = non_recent_top_bins.unsqueeze(0).expand(actual_k, -1, -1)
                topk_expanded = topk_keys_per_bin.unsqueeze(1).expand(-1, num_non_recent, -1)
                selected_keys = topk_expanded.gather(2, expanded_bins)
                # selected_keys: (actual_k, num_non_recent, top_bins)

                # Count unique keys per query
                # selected_keys: (actual_k, num_non_recent, top_bins)
                # For each query, flatten and count unique
                for q_idx in range(num_non_recent):
                    query_selected = selected_keys[:, q_idx, :].flatten()  # (actual_k * top_bins,)
                    unique_keys = torch.unique(query_selected)
                    total_unique_keys += len(unique_keys)
                total_non_recent_queries += num_non_recent

                # Check if argmax matches any selected key
                argmax_expanded = non_recent_argmax.view(1, num_non_recent, 1)
                matches = (selected_keys == argmax_expanded)
                query_hits = matches.any(dim=0).any(dim=1)  # (num_non_recent,)

                round_bin_hits = query_hits.sum().item()
                bin_hits += round_bin_hits
                total_hits += round_bin_hits

    # Compute rates
    if total_queries > 0:
        hit_rate = total_hits / total_queries * 100
        recent_rate = recent_hits / total_queries * 100
        bin_rate = bin_hits / total_queries * 100
    else:
        hit_rate = recent_rate = bin_rate = 0.0

    # Compute average unique keys per query
    if total_non_recent_queries > 0:
        avg_unique_keys = total_unique_keys / total_non_recent_queries
    else:
        avg_unique_keys = 0.0

    return {
        'hit_rate': hit_rate,
        'recent_rate': recent_rate,
        'bin_rate': bin_rate,
        'total_queries': total_queries,
        'total_hits': total_hits,
        'recent_hits': recent_hits,
        'bin_hits': bin_hits,
        'top_bins': top_bins,
        'keys_per_bin': keys_per_bin,
        'total_keys_per_query': top_bins * keys_per_bin,
        'avg_unique_keys': avg_unique_keys,
        'unique_ratio': avg_unique_keys / (top_bins * keys_per_bin) if top_bins * keys_per_bin > 0 else 0.0,
    }


def generate_configs_for_budget(budget: int, max_bins: int = 128) -> List[Tuple[int, int]]:
    """
    Generate all valid (top_bins, keys_per_bin) combinations for a given budget.

    Args:
        budget: Total keys per query budget
        max_bins: Maximum number of bins in the model

    Returns:
        List of (top_bins, keys_per_bin) tuples
    """
    configs = []

    # Find all divisors of budget that make sense
    for top_bins in range(1, min(budget + 1, max_bins + 1)):
        if budget % top_bins == 0:
            keys_per_bin = budget // top_bins
            if keys_per_bin >= 1:
                configs.append((top_bins, keys_per_bin))

    return configs


def load_model_and_verify(
    config: Dict,
    checkpoint_path: Path,
    device: torch.device,
    logger,
) -> torch.nn.Module:
    """
    Load model from checkpoint and verify it works.

    Args:
        config: Configuration dict
        checkpoint_path: Path to checkpoint
        device: torch device
        logger: Logger instance

    Returns:
        Loaded model
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get model parameters from checkpoint config if available
    ckpt_config = checkpoint.get('config', config)

    # Determine network type
    network_type = ckpt_config.get('model', {}).get('network_type', 'shared_probe')

    # Create model with appropriate settings
    model = create_model(
        ckpt_config,
        use_l2_norm=True,
        use_key_temperature=True,
        use_position_scaling=True,
        network_type=network_type,
    )

    # Load state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded from epoch {checkpoint['epoch']}, loss: {checkpoint.get('loss', 'N/A')}")

    return model


def scan_hyperparams(
    model,
    test_trace_data: Dict,
    budgets: List[int],
    device: torch.device,
    logger,
    max_bins: int = 128,
) -> Dict:
    """
    Scan hyperparameters for multiple budgets.

    Args:
        model: Module2Network instance
        test_trace_data: Preloaded test trace
        budgets: List of budget values to scan
        device: torch device
        logger: Logger instance
        max_bins: Maximum bins in model

    Returns:
        Dict with results for each budget
    """
    results = {}

    for budget in budgets:
        logger.info(f"\n{'='*60}")
        logger.info(f"Scanning budget: {budget} keys/query")
        logger.info(f"{'='*60}")

        configs = generate_configs_for_budget(budget, max_bins)
        logger.info(f"Found {len(configs)} valid configurations")

        budget_results = []
        best_hit_rate = 0
        best_config = None

        for top_bins, keys_per_bin in configs:
            start_time = time.time()

            result = compute_hit_rate_for_config(
                model, test_trace_data, top_bins, keys_per_bin, device
            )

            elapsed = time.time() - start_time

            budget_results.append(result)

            logger.info(
                f"  top_bins={top_bins:3d}, keys_per_bin={keys_per_bin:4d}: "
                f"Hit={result['hit_rate']:5.2f}% | UniqueKeys={result['avg_unique_keys']:6.1f} "
                f"({result['unique_ratio']*100:4.1f}%) [{elapsed:.1f}s]"
            )

            if result['hit_rate'] > best_hit_rate:
                best_hit_rate = result['hit_rate']
                best_config = (top_bins, keys_per_bin)

        # Sort by hit rate descending
        budget_results.sort(key=lambda x: x['hit_rate'], reverse=True)

        results[budget] = {
            'configs': budget_results,
            'best_config': best_config,
            'best_hit_rate': best_hit_rate,
        }

        # Find best config's unique keys
        best_result = next(r for r in budget_results if r['top_bins'] == best_config[0] and r['keys_per_bin'] == best_config[1])
        logger.info(f"\nBest for budget {budget}: top_bins={best_config[0]}, "
                   f"keys_per_bin={best_config[1]}, hit_rate={best_hit_rate:.2f}%, "
                   f"unique_keys={best_result['avg_unique_keys']:.1f}")

    return results


def main():
    parser = argparse.ArgumentParser(description='Hyperparameter Scanning for Sparse Attention')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='output/checkpoints/best_model_multi_trace_v5.pt',
        help='Path to checkpoint (relative to experiment dir)'
    )
    parser.add_argument(
        '--budgets',
        type=str,
        default='50,100,200,400,800',
        help='Comma-separated list of budget values to scan'
    )
    parser.add_argument(
        '--verify-baseline',
        action='store_true',
        help='First verify baseline performance with top_bins=1, K=50'
    )
    parser.add_argument(
        '--output',
        type=str,
        default='output/results/hyperparam_scan.json',
        help='Output file for results'
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Parse budgets
    budgets = [int(x.strip()) for x in args.budgets.split(',')]

    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load checkpoint
    checkpoint_path = exp_dir / args.checkpoint
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    model = load_model_and_verify(config, checkpoint_path, device, logger)

    # Get head info for preloading
    head_sample_path = exp_dir / config['data']['head_sample_file']
    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    # Preload test trace
    test_trace_path = exp_dir / config['data']['test_trace_path']
    logger.info(f"Preloading test trace from: {test_trace_path}")

    test_trace_data = preload_test_trace(
        test_trace_path, layer, head, config, logger, device
    )

    # Verify baseline if requested
    if args.verify_baseline:
        logger.info("\n" + "="*60)
        logger.info("Verifying baseline performance (top_bins=1, keys_per_bin=50)")
        logger.info("="*60)

        baseline_result = compute_hit_rate_for_config(
            model, test_trace_data, top_bins=1, keys_per_bin=50, device=device
        )

        logger.info(f"Baseline hit rate: {baseline_result['hit_rate']:.2f}%")
        logger.info(f"  Recent: {baseline_result['recent_rate']:.2f}%")
        logger.info(f"  Bin: {baseline_result['bin_rate']:.2f}%")

        if abs(baseline_result['hit_rate'] - 61) > 5:
            logger.warning(f"Baseline hit rate ({baseline_result['hit_rate']:.2f}%) differs "
                          f"significantly from expected (~61%)")

    # Run hyperparameter scan
    results = scan_hyperparams(
        model, test_trace_data, budgets, device, logger,
        max_bins=config['model']['num_bins']
    )

    # Print summary
    logger.info("\n" + "="*60)
    logger.info("SUMMARY")
    logger.info("="*60)

    for budget, budget_result in results.items():
        best = budget_result['best_config']
        hit_rate = budget_result['best_hit_rate']
        # Find best config's unique keys from configs list
        best_result = next(r for r in budget_result['configs'] if r['top_bins'] == best[0] and r['keys_per_bin'] == best[1])
        unique_keys = best_result['avg_unique_keys']
        logger.info(f"Budget {budget:4d}: Best = (bins={best[0]:3d}, K={best[1]:4d}) -> {hit_rate:.2f}%, unique={unique_keys:.1f}")

    # Save results
    output_path = exp_dir / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    serializable_results = {}
    for budget, budget_result in results.items():
        serializable_results[str(budget)] = {
            'best_config': {
                'top_bins': budget_result['best_config'][0],
                'keys_per_bin': budget_result['best_config'][1],
            },
            'best_hit_rate': budget_result['best_hit_rate'],
            'all_configs': budget_result['configs'],
        }

    with open(output_path, 'w') as f:
        json.dump(serializable_results, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
