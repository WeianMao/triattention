"""
Module 2 Evaluation Script

Evaluation Metrics:
- TopK Hit Rate: argmax key in selected bin's TopK keys
- Keys per Query: TopK + num_recent_keys
- Handle recent keys as auto-hit
- Handle empty bins (mask with -inf)

Memory-optimized version: filters unused heads after loading
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


def load_qk_filtered_by_heads(trace_path, head_samples, logger):
    """
    Load QK data from a trace file and immediately filter to only the required heads.

    This saves memory by discarding unused head data right after loading.

    Args:
        trace_path: Path to the qk.pt file
        head_samples: List of (layer, head) tuples to keep
        logger: Logger instance

    Returns:
        dict with 'q' and 'k' tensors of shape (num_heads, seq_len, head_dim)
    """
    logger.info(f"Loading trace data from: {trace_path}")
    qk_data = torch.load(trace_path, map_location='cpu')

    # Get original shape info
    orig_shape = qk_data['q'].shape  # (num_layers, num_heads_per_layer, seq_len, head_dim)
    seq_len = orig_shape[2]
    head_dim = orig_shape[3]
    logger.info(f"Original trace shape: {orig_shape}, seq_len={seq_len}")

    # Extract only the required heads
    q_list = []
    k_list = []
    for layer, head in head_samples:
        q_list.append(qk_data['q'][layer, head])  # (seq_len, head_dim)
        k_list.append(qk_data['k'][layer, head])

    # Stack into (num_heads, seq_len, head_dim)
    q_filtered = torch.stack(q_list, dim=0)
    k_filtered = torch.stack(k_list, dim=0)

    # Delete original data to free memory
    del qk_data

    logger.info(f"Filtered to {len(head_samples)} heads: Q={q_filtered.shape}, K={k_filtered.shape}")

    return {
        'q': q_filtered,
        'k': k_filtered,
        'seq_len': seq_len,
        'head_dim': head_dim
    }


def load_test_trace_data(config, logger):
    """
    Load test trace data, filtering to only required heads.

    Args:
        config: Configuration dict
        logger: Logger instance

    Returns:
        List of dicts, each containing Q, K, attention for one head
    """
    exp_dir = Path(__file__).parent

    # Load head sample file
    head_sample_path = exp_dir / config['data']['head_sample_file']
    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    # Get test trace path
    test_trace_path = exp_dir / config['data']['test_trace_path']
    if not test_trace_path.exists():
        raise FileNotFoundError(f"Test trace file not found: {test_trace_path}")

    logger.info("Using TEST trace for evaluation (cross-trace validation mode)")

    # Load and filter
    filtered_data = load_qk_filtered_by_heads(test_trace_path, head_samples, logger)

    test_data_list = []

    for head_idx, (layer, head) in enumerate(head_samples):
        Q = filtered_data['q'][head_idx]
        K = filtered_data['k'][head_idx]
        seq_len = filtered_data['seq_len']
        head_dim = filtered_data['head_dim']

        # Compute attention matrix
        scale = head_dim ** -0.5
        attention_logits = Q @ K.T * scale
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
        attention_logits.masked_fill_(causal_mask, float('-inf'))
        attention = F.softmax(attention_logits, dim=-1)

        test_data_list.append({
            'Q': Q,
            'K': K,
            'attention': attention,
            'seq_len': seq_len,
            'head_dim': head_dim,
            'layer': layer,
            'head': head
        })

    del filtered_data
    logger.info(f"Loaded {len(test_data_list)} test samples")
    return test_data_list


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


def compute_topk_hit_rate_single(
    model,
    trace_data,
    K_values,
    round_window,
    exclude_tail,
    device,
    logger
):
    """
    Compute TopK Hit Rate for a single trace/head.

    Returns:
        Dict with hit counts for each K value
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

    return results


def compute_topk_hit_rate(
    model,
    all_test_data,
    K_values,
    round_window,
    exclude_tail,
    device,
    logger
):
    """
    Compute TopK Hit Rate for all test traces/heads.

    Args:
        model: Module2Network instance (in eval mode)
        all_test_data: List of trace data dicts
        K_values: List of K values to evaluate (e.g., [50, 500, 1000])
        round_window: Size of each round
        exclude_tail: Number of tail queries to exclude
        device: torch.device
        logger: Logger instance

    Returns:
        Dict with hit rates for each K value and detailed statistics
    """
    # Aggregate results across all test samples
    aggregate_results = {k: {'hits': 0, 'total': 0, 'recent_hits': 0, 'bin_hits': 0} for k in K_values}

    for idx, trace_data in enumerate(all_test_data):
        layer, head = trace_data['layer'], trace_data['head']
        logger.info(f"Evaluating head {idx+1}/{len(all_test_data)}: layer={layer}, head={head}")

        results = compute_topk_hit_rate_single(
            model, trace_data, K_values, round_window, exclude_tail, device, logger
        )

        # Aggregate
        for k_val in K_values:
            aggregate_results[k_val]['hits'] += results[k_val]['hits']
            aggregate_results[k_val]['total'] += results[k_val]['total']
            aggregate_results[k_val]['recent_hits'] += results[k_val]['recent_hits']
            aggregate_results[k_val]['bin_hits'] += results[k_val]['bin_hits']

    # Compute hit rates
    hit_rates = {}
    for k_val in K_values:
        total = aggregate_results[k_val]['total']
        if total > 0:
            hit_rate = aggregate_results[k_val]['hits'] / total * 100
            recent_rate = aggregate_results[k_val]['recent_hits'] / total * 100
            bin_rate = aggregate_results[k_val]['bin_hits'] / total * 100
        else:
            hit_rate = 0.0
            recent_rate = 0.0
            bin_rate = 0.0

        hit_rates[k_val] = {
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_rate,
            'bin_hit_rate': bin_rate,
            'total_queries': total,
            'total_hits': aggregate_results[k_val]['hits'],
            'recent_hits': aggregate_results[k_val]['recent_hits'],
            'bin_hits': aggregate_results[k_val]['bin_hits']
        }

        logger.info(
            f"K={k_val}: Hit Rate = {hit_rate:.2f}% "
            f"(Recent: {recent_rate:.2f}%, Bin: {bin_rate:.2f}%, "
            f"Total: {aggregate_results[k_val]['hits']}/{total})"
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

    # Load test trace data
    all_test_data = load_test_trace_data(config, logger)

    # Load model
    model = load_checkpoint(checkpoint_path, config, device, logger)

    # Get evaluation parameters
    K_values = config['evaluation']['topk_K']
    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    logger.info(f"Evaluating TopK Hit Rate for K={K_values}")

    # Compute hit rates
    hit_rates = compute_topk_hit_rate(
        model, all_test_data, K_values, round_window, exclude_tail, device, logger
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
