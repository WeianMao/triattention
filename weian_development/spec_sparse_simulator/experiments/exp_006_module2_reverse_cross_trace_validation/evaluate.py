"""
Module 2 Evaluation Script

Evaluation Metrics:
- TopK Hit Rate: argmax key in selected bin's TopK keys
- Keys per Query: TopK + num_recent_keys
- Handle recent keys as auto-hit
- Handle empty bins (mask with -inf)
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import yaml

from model import Module2Network, create_model


def setup_logging(config):
    """Setup logging configuration."""
    exp_dir = Path(__file__).parent
    log_dir = exp_dir / config['output']['logs_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'evaluate.log'

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


def load_trace_data(config, logger):
    """
    Load trace data from qk.pt file.

    Args:
        config: Configuration dict
        logger: Logger instance

    Returns:
        dict with Q, K tensors and metadata
    """
    exp_dir = Path(__file__).parent

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
    logger.info(f"Using head [layer={layer}, head={head}] for POC evaluation")

    # Extract Q, K for selected head
    Q = qk_data['q'][layer, head]  # (seq_len, head_dim)
    K = qk_data['k'][layer, head]  # (seq_len, head_dim)

    seq_len, head_dim = Q.shape
    logger.info(f"Trace shape: Q={Q.shape}, K={K.shape}")

    # Compute attention matrix with causal mask
    scale = head_dim ** -0.5
    attention_logits = Q @ K.T * scale  # (seq_len, seq_len)

    # Apply causal mask: keys must be <= query position
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_logits.masked_fill_(causal_mask, float('-inf'))

    attention = F.softmax(attention_logits, dim=-1)  # (seq_len, seq_len)

    return {
        'Q': Q,
        'K': K,
        'attention': attention,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'layer': layer,
        'head': head
    }


def load_checkpoint(checkpoint_path, config, device, logger):
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        config: Configuration dict
        device: torch.device
        logger: Logger instance

    Returns:
        Loaded model
    """
    logger.info(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = create_model(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded model from epoch {checkpoint['epoch']}, loss: {checkpoint['loss']:.6f}")

    return model


def compute_topk_hit_rate(
    model,
    trace_data,
    K_values,
    round_window,
    exclude_tail,
    device,
    logger
):
    """
    Compute TopK Hit Rate for different K values.

    TopK Hit Rate: Percentage of queries whose argmax key is either:
    1. In the selected bin's TopK keys (historical)
    2. In recent keys (auto-hit, always included via full attention)

    Args:
        model: Module2Network instance (in eval mode)
        trace_data: Dict with Q, K, attention, seq_len
        K_values: List of K values to evaluate (e.g., [50, 500, 1000])
        round_window: Size of each round
        exclude_tail: Number of tail queries to exclude
        device: torch.device
        logger: Logger instance

    Returns:
        Dict with hit rates for each K value and detailed statistics
    """
    model.eval()

    Q = trace_data['Q'].to(device)
    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']

    # Initialize counters for each K value
    results = {k: {'hits': 0, 'total': 0, 'recent_hits': 0, 'bin_hits': 0} for k in K_values}

    # Determine valid query range (exclude tail)
    valid_end = min(seq_len - exclude_tail, seq_len)

    with torch.no_grad():
        # Iterate over rounds
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            # Skip first round (no historical keys)
            if round_start == 0:
                continue

            # Historical keys (< round_start)
            historical_keys = K[:round_start]  # (round_start, head_dim)
            num_historical = round_start

            # Compute reference angles for this round
            reference_angles = model.compute_reference_angles(round_start, round_window)

            # Forward pass: Key network on historical keys
            key_probs = model.forward_keys(historical_keys, reference_angles)  # (num_historical, num_bins)

            # Iterate over queries in this round
            for q_idx in range(round_start, min(round_end, valid_end)):
                # Get attention weights for all keys <= q_idx (causal)
                attn_weights = attention[q_idx, :q_idx + 1]

                # Find argmax key
                argmax_key = attn_weights.argmax().item()

                # Check if argmax is in recent keys (>= round_start)
                argmax_in_recent = argmax_key >= round_start

                # Get query vector
                query = Q[q_idx:q_idx + 1]  # (1, head_dim)

                # Detect empty bins (for masking)
                empty_bin_mask = key_probs.sum(dim=0) == 0  # (num_bins,)

                # Forward pass: Query network
                query_bin_probs = model.forward_queries(query, reference_angles, empty_bin_mask)  # (1, num_bins)

                # Select bin (argmax of query bin probabilities)
                selected_bin = query_bin_probs.argmax(dim=-1).item()

                # Get key scores for selected bin
                # key_probs: (num_historical, num_bins)
                bin_scores = key_probs[:, selected_bin]  # (num_historical,)

                # For each K value, check if argmax key is hit
                for k_val in K_values:
                    results[k_val]['total'] += 1

                    if argmax_in_recent:
                        # Auto-hit: argmax is in recent keys (always included via full attention)
                        results[k_val]['hits'] += 1
                        results[k_val]['recent_hits'] += 1
                    else:
                        # Check if argmax key is in TopK of selected bin
                        # Limit k to available historical keys
                        actual_k = min(k_val, num_historical)

                        # Get TopK key indices from bin scores
                        _, topk_indices = torch.topk(bin_scores, actual_k)

                        if argmax_key in topk_indices:
                            results[k_val]['hits'] += 1
                            results[k_val]['bin_hits'] += 1

    # Compute hit rates
    hit_rates = {}
    for k_val in K_values:
        total = results[k_val]['total']
        if total > 0:
            hit_rate = results[k_val]['hits'] / total * 100
            recent_rate = results[k_val]['recent_hits'] / total * 100
            bin_rate = results[k_val]['bin_hits'] / total * 100
        else:
            hit_rate = 0.0
            recent_rate = 0.0
            bin_rate = 0.0

        hit_rates[k_val] = {
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_rate,
            'bin_hit_rate': bin_rate,
            'total_queries': total,
            'total_hits': results[k_val]['hits'],
            'recent_hits': results[k_val]['recent_hits'],
            'bin_hits': results[k_val]['bin_hits']
        }

        logger.info(
            f"K={k_val}: Hit Rate = {hit_rate:.2f}% "
            f"(Recent: {recent_rate:.2f}%, Bin: {bin_rate:.2f}%, "
            f"Total: {results[k_val]['hits']}/{total})"
        )

    return hit_rates


def evaluate(config, checkpoint_path, logger):
    """
    Main evaluation function.

    Args:
        config: Configuration dict
        checkpoint_path: Path to model checkpoint
        logger: Logger instance

    Returns:
        Evaluation results dict
    """
    logger.info("Starting Module 2 evaluation...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load trace data
    trace_data = load_trace_data(config, logger)

    # Load model
    model = load_checkpoint(checkpoint_path, config, device, logger)

    # Get evaluation parameters
    K_values = config['evaluation']['topk_K']
    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = 0  # Align with baseline: evaluate ALL queries (no tail exclusion)

    logger.info(f"Evaluating TopK Hit Rate for K={K_values}")

    # Compute hit rates
    hit_rates = compute_topk_hit_rate(
        model, trace_data, K_values, round_window, exclude_tail, device, logger
    )

    # Save results
    exp_dir = Path(__file__).parent
    results_dir = exp_dir / config['output']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / 'evaluation_results.json'
    with open(results_file, 'w') as f:
        json.dump(hit_rates, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    return hit_rates


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Module 2 Evaluation')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='output/checkpoints/best_model.pt',
        help='Path to checkpoint (relative to experiment dir)'
    )
    args = parser.parse_args()

    # Load configuration
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config)

    # Resolve checkpoint path
    checkpoint_path = exp_dir / args.checkpoint

    if not checkpoint_path.exists():
        logger.error(f"Checkpoint not found: {checkpoint_path}")
        logger.info("Available checkpoints:")
        checkpoints_dir = exp_dir / config['output']['checkpoints_dir']
        if checkpoints_dir.exists():
            for ckpt in checkpoints_dir.glob('*.pt'):
                logger.info(f"  - {ckpt.name}")
        return

    try:
        # Run evaluation
        results = evaluate(config, checkpoint_path, logger)
        logger.info("Evaluation completed successfully")

        # Print summary
        logger.info("\n=== Evaluation Summary ===")
        for k_val, metrics in results.items():
            logger.info(f"K={k_val}: {metrics['hit_rate']:.2f}% hit rate")

    except Exception as e:
        logger.error(f"Evaluation failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
