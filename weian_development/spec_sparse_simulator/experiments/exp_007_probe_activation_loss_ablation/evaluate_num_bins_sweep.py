#!/usr/bin/env python3
"""
Evaluate Hit Rate with different numbers of selected bins.

This script tests how hit rate changes when we select different numbers of bins
(e.g., top-1, top-2, top-4, top-8, ..., top-128).

This helps understand:
1. Is the Query Network selecting the correct bins?
2. What's the upper bound if we use all bins?
3. How many bins do we need to achieve good hit rate?

Usage:
    python evaluate_num_bins_sweep.py --checkpoint output/ablation/lambda_0_0/checkpoints/best_model.pt
    python evaluate_num_bins_sweep.py --checkpoint output/ablation/lambda_0_1/checkpoints/best_model.pt --num-bins 1 2 4 8 16 32 64 128
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model import create_model


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)],
        force=True
    )
    return logging.getLogger(__name__)


def load_trace_data(config, logger, exp_dir):
    """Load trace data from qk.pt file."""
    # Use test_trace_path for evaluation if available, otherwise fall back to trace_path
    if 'test_trace_path' in config['data']:
        trace_path = exp_dir / config['data']['test_trace_path']
        logger.info("Using TEST trace for evaluation (cross-trace validation mode)")
    else:
        trace_path = exp_dir / config['data']['trace_path']
        logger.info("Using TRAINING trace for evaluation (overfit validation mode)")

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")

    logger.info(f"Loading trace data from: {trace_path}")
    qk_data = torch.load(trace_path, map_location='cpu')

    # Load head sample file
    head_sample_path = exp_dir / config['data']['head_sample_file']
    if not head_sample_path.exists():
        raise FileNotFoundError(f"Head sample file not found: {head_sample_path}")

    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    Q = qk_data['q'][layer, head]
    K = qk_data['k'][layer, head]

    seq_len, head_dim = Q.shape
    logger.info(f"Trace shape: Q={Q.shape}, K={K.shape}")

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
    }


def load_checkpoint(checkpoint_path, config, device, logger):
    """Load model from checkpoint."""
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    epoch = checkpoint.get('epoch', 'N/A')
    loss = checkpoint.get('loss', 'N/A')
    logger.info(f"Loaded model from epoch {epoch}, loss: {loss}")

    return model


def compute_hit_rate_with_num_bins(
    model,
    trace_data,
    config,
    topk_per_bin,
    num_bins_to_select,
    device,
    logger
):
    """
    Compute hit rate when selecting a specific number of bins.

    Args:
        model: Module2Network instance
        trace_data: Dict with Q, K, attention, seq_len
        config: Configuration dict
        topk_per_bin: Number of keys to select from each bin (e.g., 50)
        num_bins_to_select: Number of bins to select (1, 2, 4, ..., 128)
        device: torch.device
        logger: Logger instance

    Returns:
        Dict with hit rate statistics
    """
    model.eval()

    Q = trace_data['Q'].to(device)
    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']
    total_bins = config['model']['num_bins']

    valid_end = min(seq_len - exclude_tail, seq_len)

    hits = 0
    total = 0
    recent_hits = 0
    bin_hits = 0

    with torch.no_grad():
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            if round_start == 0:
                continue

            historical_keys = K[:round_start]
            num_historical = round_start

            reference_angles = model.compute_reference_angles(round_start, round_window)
            key_probs = model.forward_keys(historical_keys, reference_angles)

            for q_idx in range(round_start, min(round_end, valid_end)):
                attn_weights = attention[q_idx, :q_idx + 1]
                argmax_key = attn_weights.argmax().item()
                argmax_in_recent = argmax_key >= round_start

                query = Q[q_idx:q_idx + 1]
                empty_bin_mask = key_probs.sum(dim=0) == 0
                query_bin_probs = model.forward_queries(query, reference_angles, empty_bin_mask)

                # Select top-N bins
                actual_num_bins = min(num_bins_to_select, total_bins)
                _, selected_bins = torch.topk(query_bin_probs.squeeze(0), actual_num_bins)
                selected_bins = selected_bins.tolist()

                total += 1

                if argmax_in_recent:
                    hits += 1
                    recent_hits += 1
                else:
                    actual_k = min(topk_per_bin, num_historical)

                    all_topk_indices = set()
                    for bin_idx in selected_bins:
                        bin_scores = key_probs[:, bin_idx]
                        _, topk_indices = torch.topk(bin_scores, actual_k)
                        all_topk_indices.update(topk_indices.tolist())

                    if argmax_key in all_topk_indices:
                        hits += 1
                        bin_hits += 1

    hit_rate = hits / total * 100 if total > 0 else 0
    recent_rate = recent_hits / total * 100 if total > 0 else 0
    bin_rate = bin_hits / total * 100 if total > 0 else 0

    # Calculate total keys considered
    # For each query: num_bins_to_select * topk_per_bin + recent_keys (approx round_window)
    avg_keys_per_query = num_bins_to_select * topk_per_bin + round_window

    return {
        'num_bins': num_bins_to_select,
        'topk_per_bin': topk_per_bin,
        'hit_rate': round(hit_rate, 2),
        'recent_hit_rate': round(recent_rate, 2),
        'bin_hit_rate': round(bin_rate, 2),
        'total_queries': total,
        'total_hits': hits,
        'recent_hits': recent_hits,
        'bin_hits': bin_hits,
        'avg_keys_per_query': avg_keys_per_query
    }


def main():
    parser = argparse.ArgumentParser(description='Evaluate Hit Rate with different numbers of bins')
    parser.add_argument(
        '--checkpoint',
        type=str,
        required=True,
        help='Path to checkpoint (relative to experiment dir)'
    )
    parser.add_argument(
        '--num-bins',
        type=int,
        nargs='+',
        default=[1, 2, 4, 8, 16, 32, 64, 128],
        help='Number of bins to select (default: 1 2 4 8 16 32 64 128)'
    )
    parser.add_argument(
        '--topk',
        type=int,
        default=50,
        help='Top-K keys per bin (default: 50)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path'
    )
    args = parser.parse_args()

    logger = setup_logging()

    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    checkpoint_path = exp_dir / args.checkpoint
    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        return

    trace_data = load_trace_data(config, logger, exp_dir)
    model = load_checkpoint(checkpoint_path, config, device, logger)

    results = []

    logger.info(f"\nEvaluating with Top-{args.topk} keys per bin")
    logger.info(f"Testing bin counts: {args.num_bins}")
    logger.info("=" * 80)

    for num_bins in args.num_bins:
        logger.info(f"\nEvaluating with {num_bins} bins...")
        result = compute_hit_rate_with_num_bins(
            model, trace_data, config, args.topk, num_bins, device, logger
        )
        results.append(result)

        logger.info(f"  Hit Rate: {result['hit_rate']:.2f}% "
                   f"(Recent: {result['recent_hit_rate']:.2f}%, Bin: {result['bin_hit_rate']:.2f}%)")
        logger.info(f"  Avg keys per query: {result['avg_keys_per_query']}")

    # Print summary table
    logger.info("\n" + "=" * 100)
    logger.info("SUMMARY: Hit Rate vs Number of Bins")
    logger.info("=" * 100)
    logger.info(f"{'#Bins':<10} {'HitRate%':<12} {'BinHitRate%':<14} {'RecentRate%':<14} {'AvgKeys/Query':<15}")
    logger.info("-" * 100)

    for r in results:
        logger.info(f"{r['num_bins']:<10} {r['hit_rate']:<12.2f} {r['bin_hit_rate']:<14.2f} "
                   f"{r['recent_hit_rate']:<14.2f} {r['avg_keys_per_query']:<15}")

    # Save results
    if args.output:
        output_path = Path(args.output)
    else:
        output_path = checkpoint_path.parent.parent / 'num_bins_sweep_results.json'

    output_data = {
        'checkpoint': str(checkpoint_path),
        'topk_per_bin': args.topk,
        'results': results
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    logger.info(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
