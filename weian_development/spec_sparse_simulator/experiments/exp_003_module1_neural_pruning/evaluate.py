"""
Evaluation logic for Module 1 Neural Network Key Pruning.

Implements metrics: Argmax Hit Rate, Keys per Query, Computation Reduction.
Reuses reference implementation's inference method.
"""

import logging
from pathlib import Path
import json
import sys
import torch
import torch.nn.functional as F
import yaml
import matplotlib.pyplot as plt
import numpy as np

from model import Module1KeyPruningNetwork
from train import load_trace_data, setup_logging, extract_pruning_labels


def compute_module1_metrics(drop_probs, labels, query_argmax_in_history, threshold=0.5):
    """
    Compute Module 1 evaluation metrics.

    Args:
        drop_probs: (num_keys,) - model predicted drop probabilities
        labels: (num_keys,) - ground truth labels (0=should retain, 1=should drop)
        query_argmax_in_history: (num_queries,) - bool, whether each query's argmax is in historical keys
        threshold: drop threshold (default 0.5)

    Hit determination rules:
    - argmax in history → check if that key is retained
    - argmax in current round new Keys → direct hit (Full Attention)

    Returns:
        dict with 5 metrics: argmax_hit_rate, keys_per_query, computation_reduction,
                             retention_rate, false_negative_rate
    """
    # Compute retain mask: drop_probs < threshold means retain
    retain_mask = drop_probs < threshold

    # Retention Rate
    retention_rate = retain_mask.sum().item() / len(drop_probs)

    # Argmax Hit Rate (critical metric)
    # Component 1: argmax in current round new Keys → direct hit
    hits_from_recent = (~query_argmax_in_history).sum().item()

    # Component 2: argmax in history Keys (label=0) → check if retained
    should_retain = (labels == 0)
    hits_from_history = (retain_mask & should_retain).sum().item()

    total_queries = len(query_argmax_in_history)
    argmax_hit_rate = (hits_from_recent + hits_from_history) / total_queries if total_queries > 0 else 0.0

    # Keys per Query
    keys_per_query = retain_mask.sum().item()

    # Computation Reduction
    total_keys = len(drop_probs)
    computation_reduction = 1.0 - (keys_per_query / total_keys) if total_keys > 0 else 0.0

    # False Negative Rate (auxiliary metric)
    false_negatives = (~retain_mask & should_retain).sum().item()
    false_negative_rate = false_negatives / should_retain.sum().item() if should_retain.sum().item() > 0 else 0.0

    return {
        'argmax_hit_rate': argmax_hit_rate,
        'keys_per_query': keys_per_query,
        'computation_reduction': computation_reduction,
        'retention_rate': retention_rate,
        'false_negative_rate': false_negative_rate,
    }


def evaluate_on_trace(model, trace_data, config, threshold=0.5, round_window=128, logger=None):
    """
    Evaluate trained model on full trace across all rounds.

    Args:
        model: Module1KeyPruningNetwork instance
        trace_data: Dict with Q, K, attention, seq_len
        config: Configuration dict
        threshold: drop threshold (default 0.5)
        round_window: round window size (default 128)
        logger: Logger instance

    Returns:
        dict with per_round_metrics and overall_metrics
    """
    model.eval()

    K = trace_data['K']
    attention = trace_data['attention']
    seq_len = trace_data['seq_len']

    device = next(model.parameters()).device

    per_round_metrics = []

    # Accumulators for overall metrics
    total_hits_recent = 0
    total_hits_history = 0
    total_queries = 0
    total_retained_keys = 0
    total_keys = 0
    total_false_negatives = 0
    total_should_retain = 0

    # Iterate over rounds
    for round_start in range(0, seq_len, round_window):
        round_end = min(round_start + round_window, seq_len)

        # Skip if round_start is 0 (no historical keys)
        if round_start == 0:
            continue

        # Extract labels for this round
        labels = extract_pruning_labels(attention, round_start, round_end, seq_len)
        labels = labels.to(device)

        # Get historical keys
        keys = K[:round_start].to(device)

        # Create position indices
        key_positions = torch.arange(round_start, device=device)

        # Compute reference angles
        reference_angles = model.kernel_layer._compute_reference_angles(
            round_start,
            round_window=round_window
        )

        # Forward pass
        with torch.no_grad():
            drop_probs = model(keys, key_positions, reference_angles)

        # Determine query_argmax_in_history for current round queries
        # Check if argmax for queries in [round_start, round_end) is in historical keys
        query_argmax_in_history = torch.zeros(round_end - round_start, dtype=torch.bool, device=device)

        for q_idx in range(round_start, round_end):
            # Get attention weights for this query
            attn_weights = attention[q_idx]
            argmax_key = attn_weights.argmax().item()

            # Check if argmax is in historical keys (< round_start)
            if argmax_key < round_start:
                query_argmax_in_history[q_idx - round_start] = True

        # Compute metrics for this round
        metrics = compute_module1_metrics(drop_probs, labels, query_argmax_in_history, threshold)

        # Store per-round metrics
        per_round_metrics.append({
            'round_start': round_start,
            'round_end': round_end,
            **metrics
        })

        # Accumulate for overall metrics
        retain_mask = drop_probs < threshold
        should_retain = (labels == 0)

        hits_recent = (~query_argmax_in_history).sum().item()
        hits_history = (retain_mask & should_retain).sum().item()

        total_hits_recent += hits_recent
        total_hits_history += hits_history
        total_queries += len(query_argmax_in_history)
        total_retained_keys += retain_mask.sum().item()
        total_keys += len(drop_probs)
        total_false_negatives += (~retain_mask & should_retain).sum().item()
        total_should_retain += should_retain.sum().item()

    # Compute overall metrics
    overall_argmax_hit_rate = (total_hits_recent + total_hits_history) / total_queries if total_queries > 0 else 0.0
    overall_keys_per_query = total_retained_keys / len(per_round_metrics) if len(per_round_metrics) > 0 else 0.0
    overall_computation_reduction = 1.0 - (total_retained_keys / total_keys) if total_keys > 0 else 0.0
    overall_retention_rate = total_retained_keys / total_keys if total_keys > 0 else 0.0
    overall_false_negative_rate = total_false_negatives / total_should_retain if total_should_retain > 0 else 0.0

    overall_metrics = {
        'argmax_hit_rate': overall_argmax_hit_rate,
        'keys_per_query': overall_keys_per_query,
        'computation_reduction': overall_computation_reduction,
        'retention_rate': overall_retention_rate,
        'false_negative_rate': overall_false_negative_rate,
    }

    if logger:
        logger.info(f"Overall Metrics:")
        logger.info(f"  Argmax Hit Rate: {overall_argmax_hit_rate:.4f}")
        logger.info(f"  Keys per Query: {overall_keys_per_query:.2f}")
        logger.info(f"  Computation Reduction: {overall_computation_reduction:.4f}")
        logger.info(f"  Retention Rate: {overall_retention_rate:.4f}")
        logger.info(f"  False Negative Rate: {overall_false_negative_rate:.4f}")

    return {
        'per_round_metrics': per_round_metrics,
        'overall_metrics': overall_metrics
    }


def plot_drop_prob_distribution(drop_probs, labels, save_path, threshold=0.5):
    """
    Plot drop probability distribution colored by labels.

    Args:
        drop_probs: (num_keys,) drop probabilities
        labels: (num_keys,) ground truth labels (0=retain, 1=drop)
        save_path: Path to save figure
        threshold: drop threshold (default 0.5)
    """
    plt.figure(figsize=(10, 6))

    # Separate by labels
    retain_probs = drop_probs[labels == 0].cpu().numpy()
    drop_probs_true = drop_probs[labels == 1].cpu().numpy()

    # Plot histograms
    plt.hist(retain_probs, bins=50, alpha=0.6, color='blue', label='Should Retain (label=0)')
    plt.hist(drop_probs_true, bins=50, alpha=0.6, color='red', label='Should Drop (label=1)')

    # Add vertical line at threshold
    plt.axvline(x=threshold, color='black', linestyle='--', linewidth=2, label=f'Threshold={threshold}')

    plt.xlabel('Drop Probability')
    plt.ylabel('Frequency')
    plt.title('Drop Probability Distribution by Ground Truth Labels')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metrics_vs_threshold(model, trace_data, config, save_path, logger=None):
    """
    Plot metrics vs threshold sweep.

    Args:
        model: Module1KeyPruningNetwork instance
        trace_data: Dict with Q, K, attention, seq_len
        config: Configuration dict
        save_path: Path to save figure
        logger: Logger instance
    """
    # Sweep threshold from 0.0 to 1.0
    thresholds = np.arange(0.0, 1.05, 0.05)

    hit_rates = []
    keys_per_query = []
    retention_rates = []

    round_window = config['data']['round_window']

    if logger:
        logger.info("Sweeping thresholds for metrics vs threshold plot...")

    for threshold in thresholds:
        results = evaluate_on_trace(model, trace_data, config, threshold=threshold, round_window=round_window)
        overall = results['overall_metrics']

        hit_rates.append(overall['argmax_hit_rate'])
        keys_per_query.append(overall['keys_per_query'])
        retention_rates.append(overall['retention_rate'])

    # Create plot
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12))

    # Plot 1: Argmax Hit Rate
    ax1.plot(thresholds, hit_rates, 'b-', linewidth=2, label='Argmax Hit Rate')
    ax1.axhline(y=0.99, color='red', linestyle='--', linewidth=1.5, label='99% Target')
    ax1.set_xlabel('Threshold')
    ax1.set_ylabel('Argmax Hit Rate')
    ax1.set_title('Argmax Hit Rate vs Threshold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0.0, 1.05])

    # Plot 2: Keys per Query
    ax2.plot(thresholds, keys_per_query, 'g-', linewidth=2, label='Keys per Query')
    ax2.set_xlabel('Threshold')
    ax2.set_ylabel('Keys per Query')
    ax2.set_title('Keys per Query vs Threshold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Retention Rate
    ax3.plot(thresholds, retention_rates, 'orange', linewidth=2, label='Retention Rate')
    ax3.set_xlabel('Threshold')
    ax3.set_ylabel('Retention Rate')
    ax3.set_title('Retention Rate vs Threshold')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0.0, 1.05])

    plt.tight_layout()

    # Save figure
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()

    if logger:
        logger.info(f"Metrics vs threshold plot saved to: {save_path}")


def evaluate(checkpoint_path, config, logger):
    """
    Main evaluation entry point.

    Args:
        checkpoint_path: Path to model checkpoint
        config: Configuration dict
        logger: Logger instance

    Returns:
        dict: Evaluation results
    """
    logger.info("Initializing evaluation...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    model_cfg = config['model']
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Get head_dim from checkpoint config if available
    if 'config' in checkpoint and 'data' in checkpoint['config']:
        trace_config = checkpoint['config']
    else:
        trace_config = config

    # Load trace data to get head_dim
    trace_data = load_trace_data(config, logger)
    head_dim = trace_data['head_dim']

    model = Module1KeyPruningNetwork(
        num_bins=model_cfg['num_bins'],
        num_freqs=head_dim // 2,
        num_kernels=model_cfg['num_kernels'],
        mlp_hidden=model_cfg['mlp_hidden_dim'],
        anchor_positions=model_cfg['position_anchors']
    )

    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    logger.info(f"Model loaded from: {checkpoint_path}")

    # Get threshold
    threshold = config['evaluation']['threshold_value']
    round_window = config['data']['round_window']

    # Run evaluation on trace
    results = evaluate_on_trace(model, trace_data, config, threshold=threshold,
                               round_window=round_window, logger=logger)

    # Setup output directories
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / config['output']['results_dir']
    figures_dir = exp_dir / config['output']['figures_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Save metrics to JSON
    metrics_path = results_dir / 'metrics.json'
    with open(metrics_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"Metrics saved to: {metrics_path}")

    # Generate visualizations
    logger.info("Generating visualizations...")

    # Collect all drop_probs and labels across rounds for distribution plot
    K = trace_data['K']
    attention = trace_data['attention']
    seq_len = trace_data['seq_len']

    all_drop_probs = []
    all_labels = []

    for round_start in range(0, seq_len, round_window):
        round_end = min(round_start + round_window, seq_len)

        if round_start == 0:
            continue

        labels = extract_pruning_labels(attention, round_start, round_end, seq_len)
        labels = labels.to(device)

        keys = K[:round_start].to(device)
        key_positions = torch.arange(round_start, device=device)

        reference_angles = model.kernel_layer._compute_reference_angles(
            round_start,
            round_window=round_window
        )

        with torch.no_grad():
            drop_probs = model(keys, key_positions, reference_angles)

        all_drop_probs.append(drop_probs)
        all_labels.append(labels)

    all_drop_probs = torch.cat(all_drop_probs)
    all_labels = torch.cat(all_labels)

    # Plot 1: Drop probability distribution
    dist_plot_path = figures_dir / 'drop_prob_distribution.png'
    plot_drop_prob_distribution(all_drop_probs, all_labels, dist_plot_path, threshold=threshold)
    logger.info(f"Drop probability distribution plot saved to: {dist_plot_path}")

    # Plot 2: Metrics vs threshold
    metrics_plot_path = figures_dir / 'metrics_vs_threshold.png'
    plot_metrics_vs_threshold(model, trace_data, config, metrics_plot_path, logger=logger)

    logger.info("Evaluation completed successfully!")
    return results


def main():
    """Main entry point."""
    # Load configuration
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config)

    # Get checkpoint path
    checkpoints_dir = exp_dir / config['output']['checkpoints_dir']
    checkpoint_path = checkpoints_dir / 'final_model.pt'

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.error("Please train the model first using train.py")
        sys.exit(1)

    try:
        # Run evaluation
        results = evaluate(checkpoint_path, config, logger)
        logger.info("Evaluation successful!")
    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
