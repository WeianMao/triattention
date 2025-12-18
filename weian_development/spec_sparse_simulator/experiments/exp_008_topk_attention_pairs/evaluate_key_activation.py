#!/usr/bin/env python3
"""
Evaluate Key Network Activation Metrics for Loss1 (Probe Activation Loss).

This script measures whether Loss1 is successfully "activating" bins by making
the Key Network place Ground Truth keys (keys that will be attended to) into
the top-K of each bin.

Key Metrics:
1. Total Recall@K: What fraction of GT keys appear in at least one bin's Top-K?
2. Average GT per Bin: How many GT keys does each bin contain in its Top-K?
3. Bins with GT > 0: How many bins have at least 1 GT key in Top-K?
4. GT Distribution across Bins: How evenly distributed are GT keys across bins?

Ground Truth Definition:
- For each round, GT keys = unique argmax keys (within historical keys) that
  are the attention target for queries in that round.

Usage:
    python evaluate_key_activation.py --checkpoint output/ablation/lambda_0_1/checkpoints/best_model.pt
    python evaluate_key_activation.py --ablation-dir output/ablation --topk 50
"""

import argparse
import json
import logging
import math
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

    # Select first head for POC
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    # Extract Q, K for selected head
    Q = qk_data['q'][layer, head]  # (seq_len, head_dim)
    K = qk_data['k'][layer, head]  # (seq_len, head_dim)

    seq_len, head_dim = Q.shape
    logger.info(f"Trace shape: Q={Q.shape}, K={K.shape}")

    # Compute attention matrix with causal mask
    scale = head_dim ** -0.5
    attention_logits = Q @ K.T * scale

    # Apply causal mask
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


def compute_key_activation_metrics(
    model,
    trace_data,
    config,
    topk_values,
    device,
    logger
):
    """
    Compute Key Network activation metrics.

    For each round:
    1. Identify Ground Truth keys (argmax keys within historical range)
    2. Get Key Network's probability distribution over bins
    3. For each bin, get Top-K keys
    4. Compute how many GT keys are in each bin's Top-K

    Args:
        model: Module2Network instance (eval mode)
        trace_data: Dict with Q, K, attention, seq_len
        config: Configuration dict
        topk_values: List of K values for Top-K analysis (e.g., [50, 100, 200])
        device: torch.device
        logger: Logger instance

    Returns:
        Dict with metrics for each topk value
    """
    model.eval()

    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']
    num_bins = config['model']['num_bins']

    valid_end = min(seq_len - exclude_tail, seq_len)

    # Initialize results structure for each topk
    results = {k: {
        'total_gt_keys': 0,
        'gt_in_any_bin_topk': 0,
        'gt_per_bin': [],  # List of (round, bin_idx, gt_count) for analysis
        'bins_with_gt': [],  # Number of bins with at least 1 GT key per round
        'round_metrics': []  # Detailed per-round metrics
    } for k in topk_values}

    with torch.no_grad():
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            # Skip first round
            if round_start == 0:
                continue

            # Historical keys
            historical_keys = K[:round_start]
            num_historical = round_start

            # Compute reference angles
            reference_angles = model.compute_reference_angles(round_start, round_window)

            # Forward pass: Key network
            key_probs = model.forward_keys(historical_keys, reference_angles)
            # key_probs: (num_historical, num_bins)

            # Collect Ground Truth keys for this round
            gt_keys_set = set()
            round_queries_end = min(round_end, valid_end)

            for q_idx in range(round_start, round_queries_end):
                attn_weights = attention[q_idx, :q_idx + 1]
                argmax_key = attn_weights.argmax().item()

                # Only include if argmax is in historical keys
                if argmax_key < round_start:
                    gt_keys_set.add(argmax_key)

            if len(gt_keys_set) == 0:
                continue

            gt_keys = list(gt_keys_set)
            gt_keys_tensor = torch.tensor(gt_keys, device=device)

            # For each topk value, compute metrics
            for topk in topk_values:
                actual_topk = min(topk, num_historical)

                # For each bin, get its top-K keys
                # key_probs: (num_historical, num_bins)
                # We want top-K keys for each bin
                topk_per_bin = []
                gt_in_bin = []

                for bin_idx in range(num_bins):
                    bin_probs = key_probs[:, bin_idx]
                    _, topk_indices = torch.topk(bin_probs, actual_topk)
                    topk_set = set(topk_indices.cpu().tolist())
                    topk_per_bin.append(topk_set)

                    # Count GT keys in this bin's top-K
                    gt_count = len(gt_keys_set.intersection(topk_set))
                    gt_in_bin.append(gt_count)

                # Compute round-level metrics
                # 1. Total GT keys covered by any bin's top-K
                all_topk_keys = set()
                for topk_set in topk_per_bin:
                    all_topk_keys.update(topk_set)

                gt_covered = len(gt_keys_set.intersection(all_topk_keys))

                # 2. Number of bins with at least 1 GT key
                bins_with_gt = sum(1 for c in gt_in_bin if c > 0)

                # 3. Average GT per bin
                avg_gt_per_bin = sum(gt_in_bin) / num_bins

                # 4. Max GT in any single bin
                max_gt_in_bin = max(gt_in_bin)

                # Store results
                results[topk]['total_gt_keys'] += len(gt_keys_set)
                results[topk]['gt_in_any_bin_topk'] += gt_covered
                results[topk]['bins_with_gt'].append(bins_with_gt)

                results[topk]['round_metrics'].append({
                    'round_start': round_start,
                    'num_gt_keys': len(gt_keys_set),
                    'gt_covered': gt_covered,
                    'recall': gt_covered / len(gt_keys_set) if gt_keys_set else 0,
                    'bins_with_gt': bins_with_gt,
                    'avg_gt_per_bin': avg_gt_per_bin,
                    'max_gt_in_bin': max_gt_in_bin,
                    'gt_distribution': gt_in_bin  # Full distribution for analysis
                })

    # Compute summary metrics
    summary = {}
    for topk in topk_values:
        r = results[topk]
        total_gt = r['total_gt_keys']
        covered = r['gt_in_any_bin_topk']

        recall = covered / total_gt if total_gt > 0 else 0
        avg_bins_with_gt = sum(r['bins_with_gt']) / len(r['bins_with_gt']) if r['bins_with_gt'] else 0

        # Compute per-round average recall
        per_round_recalls = [rm['recall'] for rm in r['round_metrics']]
        avg_per_round_recall = sum(per_round_recalls) / len(per_round_recalls) if per_round_recalls else 0

        # Compute average GT per bin (across all rounds)
        all_avg_gt = [rm['avg_gt_per_bin'] for rm in r['round_metrics']]
        overall_avg_gt_per_bin = sum(all_avg_gt) / len(all_avg_gt) if all_avg_gt else 0

        summary[topk] = {
            'total_recall': round(recall * 100, 2),
            'avg_per_round_recall': round(avg_per_round_recall * 100, 2),
            'avg_bins_with_gt': round(avg_bins_with_gt, 2),
            'avg_gt_per_bin': round(overall_avg_gt_per_bin, 4),
            'total_gt_keys': total_gt,
            'gt_covered': covered,
            'num_rounds': len(r['round_metrics'])
        }

        logger.info(f"Top-{topk}: Recall={recall*100:.2f}%, "
                   f"Avg Bins with GT={avg_bins_with_gt:.2f}/{num_bins}, "
                   f"Avg GT/Bin={overall_avg_gt_per_bin:.4f}")

    return summary, results


def analyze_gt_distribution(results, topk, num_bins, logger):
    """
    Analyze how GT keys are distributed across bins.

    Returns distribution statistics like:
    - How many bins typically have GT keys
    - Is GT concentrated in few bins or spread out?
    """
    all_distributions = []
    for rm in results[topk]['round_metrics']:
        all_distributions.append(rm['gt_distribution'])

    if not all_distributions:
        return None

    # Aggregate across all rounds
    num_rounds = len(all_distributions)
    bin_totals = [0] * num_bins

    for dist in all_distributions:
        for bin_idx, count in enumerate(dist):
            bin_totals[bin_idx] += count

    # Compute statistics
    non_zero_bins = sum(1 for b in bin_totals if b > 0)
    total_gt_assignments = sum(bin_totals)

    # Gini coefficient for GT distribution
    sorted_totals = sorted(bin_totals)
    cumulative = 0
    gini_sum = 0
    for i, t in enumerate(sorted_totals):
        cumulative += t
        gini_sum += cumulative

    if total_gt_assignments > 0:
        gini = 1 - 2 * gini_sum / (num_bins * total_gt_assignments) + 1 / num_bins
    else:
        gini = 1.0

    # Entropy
    if total_gt_assignments > 0:
        entropy = 0
        for b in bin_totals:
            if b > 0:
                p = b / total_gt_assignments
                entropy -= p * math.log(p)
        max_entropy = math.log(num_bins)
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0
    else:
        normalized_entropy = 0

    logger.info(f"GT Distribution Analysis (Top-{topk}):")
    logger.info(f"  Bins ever containing GT: {non_zero_bins}/{num_bins}")
    logger.info(f"  Gini coefficient: {gini:.4f} (0=uniform, 1=concentrated)")
    logger.info(f"  Entropy (normalized): {normalized_entropy:.4f} (1=uniform)")

    # Top bins by GT count
    bin_counts = [(i, c) for i, c in enumerate(bin_totals)]
    bin_counts.sort(key=lambda x: -x[1])
    top5 = bin_counts[:5]
    logger.info(f"  Top 5 bins: {[(b, c) for b, c in top5]}")

    return {
        'bins_with_any_gt': non_zero_bins,
        'gini': round(gini, 4),
        'entropy_normalized': round(normalized_entropy, 4),
        'top5_bins': top5,
        'bin_totals': bin_totals
    }


def evaluate_single_checkpoint(checkpoint_path, config, exp_dir, topk_values, device, logger):
    """Evaluate a single checkpoint."""
    trace_data = load_trace_data(config, logger, exp_dir)
    model = load_checkpoint(checkpoint_path, config, device, logger)

    summary, detailed = compute_key_activation_metrics(
        model, trace_data, config, topk_values, device, logger
    )

    # Analyze GT distribution for primary topk
    primary_topk = topk_values[0]
    dist_analysis = analyze_gt_distribution(detailed, primary_topk, config['model']['num_bins'], logger)

    return {
        'summary': summary,
        'distribution_analysis': dist_analysis,
        'checkpoint': str(checkpoint_path)
    }


def evaluate_ablation_dir(ablation_dir, config, exp_dir, topk_values, device, logger):
    """Evaluate all checkpoints in ablation directory."""
    ablation_path = exp_dir / ablation_dir

    if not ablation_path.exists():
        logger.error(f"Ablation directory not found: {ablation_path}")
        return None

    # Find all lambda_* subdirectories
    lambda_dirs = sorted([d for d in ablation_path.iterdir() if d.is_dir() and d.name.startswith('lambda_')])

    if not lambda_dirs:
        logger.error(f"No lambda_* directories found in {ablation_path}")
        return None

    logger.info(f"Found {len(lambda_dirs)} ablation experiments")

    # Load trace data once
    trace_data = load_trace_data(config, logger, exp_dir)

    all_results = {}

    for lambda_dir in lambda_dirs:
        checkpoint_path = lambda_dir / 'checkpoints' / 'best_model.pt'
        if not checkpoint_path.exists():
            checkpoint_path = lambda_dir / 'checkpoints' / 'final_model.pt'

        if not checkpoint_path.exists():
            logger.warning(f"No checkpoint found for {lambda_dir.name}")
            continue

        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {lambda_dir.name}")
        logger.info(f"{'='*60}")

        model = load_checkpoint(checkpoint_path, config, device, logger)

        summary, detailed = compute_key_activation_metrics(
            model, trace_data, config, topk_values, device, logger
        )

        primary_topk = topk_values[0]
        dist_analysis = analyze_gt_distribution(detailed, primary_topk, config['model']['num_bins'], logger)

        all_results[lambda_dir.name] = {
            'summary': summary,
            'distribution_analysis': dist_analysis,
            'checkpoint': str(checkpoint_path)
        }

        # Free memory
        del model
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return all_results


def print_comparison_table(all_results, topk, logger):
    """Print comparison table for ablation results."""
    if not all_results:
        return

    logger.info(f"\n{'='*100}")
    logger.info(f"KEY ACTIVATION METRICS COMPARISON (Top-{topk})")
    logger.info(f"{'='*100}")

    header = f"{'Experiment':<20} {'Recall%':<12} {'AvgBinsGT':<12} {'AvgGT/Bin':<12} {'BinsAnyGT':<12} {'Entropy':<12} {'Gini':<12}"
    logger.info(header)
    logger.info("-" * 100)

    for exp_name in sorted(all_results.keys()):
        result = all_results[exp_name]
        s = result['summary'].get(topk, {})
        d = result.get('distribution_analysis', {}) or {}

        recall = f"{s.get('total_recall', 'N/A'):.2f}%" if isinstance(s.get('total_recall'), (int, float)) else 'N/A'
        avg_bins = f"{s.get('avg_bins_with_gt', 'N/A'):.2f}" if isinstance(s.get('avg_bins_with_gt'), (int, float)) else 'N/A'
        avg_gt = f"{s.get('avg_gt_per_bin', 'N/A'):.4f}" if isinstance(s.get('avg_gt_per_bin'), (int, float)) else 'N/A'
        bins_any = str(d.get('bins_with_any_gt', 'N/A'))
        entropy = f"{d.get('entropy_normalized', 'N/A'):.4f}" if isinstance(d.get('entropy_normalized'), (int, float)) else 'N/A'
        gini = f"{d.get('gini', 'N/A'):.4f}" if isinstance(d.get('gini'), (int, float)) else 'N/A'

        logger.info(f"{exp_name:<20} {recall:<12} {avg_bins:<12} {avg_gt:<12} {bins_any:<12} {entropy:<12} {gini:<12}")

    logger.info(f"\nLegend:")
    logger.info(f"  Recall%: Percentage of GT keys found in any bin's Top-{topk}")
    logger.info(f"  AvgBinsGT: Average number of bins containing GT keys per round")
    logger.info(f"  AvgGT/Bin: Average GT keys per bin (higher = more activation)")
    logger.info(f"  BinsAnyGT: Total bins that ever contained a GT key")
    logger.info(f"  Entropy: Distribution uniformity (1=uniform, 0=concentrated)")
    logger.info(f"  Gini: Inequality measure (0=uniform, 1=concentrated)")


def main():
    parser = argparse.ArgumentParser(description='Evaluate Key Network Activation Metrics')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to single checkpoint (relative to experiment dir)'
    )
    parser.add_argument(
        '--ablation-dir',
        type=str,
        default=None,
        help='Path to ablation directory containing lambda_* subdirs'
    )
    parser.add_argument(
        '--topk',
        type=int,
        nargs='+',
        default=[50, 100, 200],
        help='Top-K values to evaluate (default: 50 100 200)'
    )
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output JSON file path'
    )
    args = parser.parse_args()

    logger = setup_logging()

    # Load configuration
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    topk_values = args.topk
    logger.info(f"Evaluating Top-K values: {topk_values}")

    try:
        if args.ablation_dir:
            # Evaluate all experiments in ablation directory
            all_results = evaluate_ablation_dir(
                args.ablation_dir, config, exp_dir, topk_values, device, logger
            )

            if all_results:
                print_comparison_table(all_results, topk_values[0], logger)

                # Save results
                output_path = args.output or (exp_dir / args.ablation_dir / 'key_activation_metrics.json')
                with open(output_path, 'w') as f:
                    # Convert non-serializable items
                    json.dump(all_results, f, indent=2, default=str)
                logger.info(f"\nResults saved to: {output_path}")

        elif args.checkpoint:
            # Evaluate single checkpoint
            checkpoint_path = exp_dir / args.checkpoint
            if not checkpoint_path.exists():
                logger.error(f"Checkpoint not found: {checkpoint_path}")
                return

            result = evaluate_single_checkpoint(
                checkpoint_path, config, exp_dir, topk_values, device, logger
            )

            # Save results
            if args.output:
                output_path = Path(args.output)
            else:
                output_path = checkpoint_path.parent.parent / 'key_activation_metrics.json'

            with open(output_path, 'w') as f:
                json.dump(result, f, indent=2, default=str)
            logger.info(f"\nResults saved to: {output_path}")

        else:
            # Default: evaluate both ablation directories
            for ablation_dir in ['output/ablation', 'output/ablation_weighted']:
                ablation_path = exp_dir / ablation_dir
                if ablation_path.exists():
                    logger.info(f"\n{'#'*80}")
                    logger.info(f"Evaluating: {ablation_dir}")
                    logger.info(f"{'#'*80}")

                    all_results = evaluate_ablation_dir(
                        ablation_dir, config, exp_dir, topk_values, device, logger
                    )

                    if all_results:
                        print_comparison_table(all_results, topk_values[0], logger)

                        output_path = ablation_path / 'key_activation_metrics.json'
                        with open(output_path, 'w') as f:
                            json.dump(all_results, f, indent=2, default=str)
                        logger.info(f"Results saved to: {output_path}")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
