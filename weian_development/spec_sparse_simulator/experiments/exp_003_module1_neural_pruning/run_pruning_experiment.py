"""
Experiment runner for Module 1 Neural Network Key Pruning.

Supports configurable model architectures and automatic threshold search
to find optimal threshold achieving target hit rate.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import yaml
import torch
import torch.optim as optim
import torch.nn.functional as F

# Ensure experiment directory is in path
EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))

from model import Module1KeyPruningNetwork
from train import (
    load_trace_data,
    train_epoch,
    setup_logging as train_setup_logging
)
from run import run_evaluate_mode, compute_argmax_hit_rate_from_prune_mask


def setup_logging(log_file=None):
    """Setup logging configuration."""
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def train_model(config, trace_data, model, device, logger):
    """
    Train model from scratch.

    Args:
        config: Configuration dict
        trace_data: Dict with Q, K, attention, seq_len
        model: Module1KeyPruningNetwork instance
        device: torch.device
        logger: Logger instance

    Returns:
        final_loss: Final training loss
    """
    logger.info("Starting model training...")

    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # Training loop
    num_epochs = config['training']['epochs']
    final_loss = 0.0

    for epoch in range(1, num_epochs + 1):
        avg_loss = train_epoch(model, trace_data, optimizer, config, device, logger)
        final_loss = avg_loss

        if epoch % 10 == 0:
            logger.info(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}")

    logger.info(f"Training completed. Final loss: {final_loss:.6f}")
    return final_loss


def evaluate_model_at_threshold(model, trace_data, config, threshold, device):
    """
    Evaluate model at a specific threshold using the correct prune_mask approach.

    This mirrors the evaluation logic in run.py's run_evaluate_mode:
    - Creates prune_mask (seq_len, seq_len)
    - Always retains current round keys (Module 1 only predicts historical keys)
    - Computes argmax hit rate correctly

    Args:
        model: Module1KeyPruningNetwork instance
        trace_data: Dict with Q, K, attention, seq_len
        config: Configuration dict
        threshold: Drop probability threshold
        device: torch.device

    Returns:
        dict with hit_rate, keys_per_query
    """
    model.eval()

    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']
    round_window = config['data']['round_window']

    # Create prune mask
    prune_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    with torch.no_grad():
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            if round_start == 0:
                # No historical keys yet - retain all within causal mask
                for q_idx in range(round_end):
                    prune_mask[q_idx, :q_idx+1] = True
                continue

            # Get historical keys
            keys = K[:round_start]
            key_positions = torch.arange(round_start, device=device)

            # Compute reference angles
            reference_angles = model.kernel_layer._compute_reference_angles(
                round_start,
                round_window=round_window
            )

            # Predict drop probabilities
            drop_probs = model(keys, key_positions, reference_angles)

            # Apply threshold to get retention mask
            retain_mask = drop_probs < threshold

            # Update prune mask for queries in this round
            for q_idx in range(round_start, round_end):
                # Historical keys: apply model's retention mask
                prune_mask[q_idx, :round_start] = retain_mask.squeeze()

                # Current round keys (round_start to q_idx): ALWAYS retain
                # Module 1 only predicts for historical keys; current round uses full attention
                prune_mask[q_idx, round_start:q_idx+1] = True

    # Compute metrics
    hit_rate = compute_argmax_hit_rate_from_prune_mask(prune_mask, attention)
    keys_per_query = prune_mask.sum(dim=1).float().mean().item()

    return {
        'hit_rate': hit_rate,
        'keys_per_query': keys_per_query
    }


def evaluate_model_with_fixed_k(model, trace_data, config, fixed_k, device, logger=None):
    """
    Evaluate model with fixed number of retained keys per query.

    Instead of using threshold-based pruning, this function:
    - For each round, computes drop probabilities for all historical keys
    - Retains the top-K keys with LOWEST drop probability (most important)
    - Always retains current round keys (same as threshold mode)

    Args:
        model: Module1KeyPruningNetwork instance
        trace_data: Dict with Q, K, attention, seq_len
        config: Configuration dict
        fixed_k: Fixed number of keys to retain per query (excluding current round)
        device: torch.device
        logger: Logger instance

    Returns:
        dict with hit_rate, keys_per_query, fixed_k
    """
    if logger:
        logger.info(f"Evaluating with fixed K={fixed_k} keys per query...")

    model.eval()

    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']
    round_window = config['data']['round_window']

    # Create prune mask
    prune_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)

    with torch.no_grad():
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            if round_start == 0:
                # No historical keys yet - retain all within causal mask
                for q_idx in range(round_end):
                    prune_mask[q_idx, :q_idx+1] = True
                continue

            # Get historical keys
            keys = K[:round_start]
            key_positions = torch.arange(round_start, device=device)

            # Compute reference angles
            reference_angles = model.kernel_layer._compute_reference_angles(
                round_start,
                round_window=round_window
            )

            # Predict drop probabilities
            drop_probs = model(keys, key_positions, reference_angles).squeeze()

            # Fixed-K selection: keep top-K keys with LOWEST drop probability
            num_historical = round_start
            k_to_keep = min(fixed_k, num_historical)

            if k_to_keep < num_historical:
                # Get indices of top-K keys with lowest drop probability
                _, topk_indices = torch.topk(drop_probs, k_to_keep, largest=False)
                retain_mask = torch.zeros(num_historical, dtype=torch.bool, device=device)
                retain_mask[topk_indices] = True
            else:
                # Keep all historical keys if fixed_k >= num_historical
                retain_mask = torch.ones(num_historical, dtype=torch.bool, device=device)

            # Update prune mask for queries in this round
            for q_idx in range(round_start, round_end):
                # Historical keys: apply fixed-K retention mask
                prune_mask[q_idx, :round_start] = retain_mask

                # Current round keys (round_start to q_idx): ALWAYS retain
                prune_mask[q_idx, round_start:q_idx+1] = True

    # Compute metrics
    hit_rate = compute_argmax_hit_rate_from_prune_mask(prune_mask, attention)
    keys_per_query = prune_mask.sum(dim=1).float().mean().item()

    if logger:
        logger.info(f"Fixed-K evaluation results:")
        logger.info(f"  Fixed K: {fixed_k}")
        logger.info(f"  Hit rate: {hit_rate*100:.2f}%")
        logger.info(f"  Actual keys per query: {keys_per_query:.2f}")

    return {
        'hit_rate': hit_rate,
        'keys_per_query': keys_per_query,
        'fixed_k': fixed_k
    }


def auto_threshold_search(model, trace_data, config, target_hit_rate=0.995,
                          precision=0.001, logger=None):
    """
    Binary search for optimal threshold achieving target hit rate.

    Semantics: drop_probs < threshold → retain key
    - Higher threshold → retain more keys → higher hit rate (but less pruning)
    - Lower threshold → drop more keys → lower hit rate (but more pruning)

    Goal: Find LOWEST threshold that achieves hit_rate >= target_hit_rate
    (maximize pruning while meeting hit rate constraint)

    Args:
        model: Trained Module1KeyPruningNetwork instance
        trace_data: Dict with Q, K, attention, seq_len
        config: Configuration dict
        target_hit_rate: Target hit rate (default: 0.995 = 99.5%)
        precision: Search precision (default: 0.001)
        logger: Logger instance

    Returns:
        dict with optimal_threshold, achieved_hit_rate, keys_per_query
    """
    if logger:
        logger.info(f"Starting auto threshold search (target: {target_hit_rate*100:.1f}%)...")

    model.eval()
    device = next(model.parameters()).device

    # Binary search bounds
    low = 0.0
    high = 1.0
    best_threshold = None
    best_hit_rate = 0.0
    best_keys_per_query = float('inf')

    iteration = 0
    max_iterations = 50  # Safety limit

    while (high - low) > precision and iteration < max_iterations:
        mid = (low + high) / 2.0
        iteration += 1

        # Evaluate at mid threshold using correct evaluation method
        results = evaluate_model_at_threshold(model, trace_data, config, mid, device)

        hit_rate = results['hit_rate']
        keys_per_query = results['keys_per_query']

        if logger and iteration % 5 == 0:
            logger.info(f"  Iter {iteration}: threshold={mid:.4f}, hit_rate={hit_rate*100:.2f}%, keys={keys_per_query:.2f}")

        if hit_rate >= target_hit_rate:
            # Target achieved at this threshold
            # Try LOWER threshold (more aggressive pruning)
            best_threshold = mid
            best_hit_rate = hit_rate
            best_keys_per_query = keys_per_query
            high = mid  # Search lower half
        else:
            # Target not achieved, need HIGHER threshold (retain more keys)
            low = mid  # Search upper half

    # Final evaluation at best threshold
    if best_threshold is None:
        # No threshold achieved target, use highest (threshold=1.0 retains all)
        best_threshold = 1.0
        results = evaluate_model_at_threshold(model, trace_data, config, best_threshold, device)
        best_hit_rate = results['hit_rate']
        best_keys_per_query = results['keys_per_query']

    if logger:
        logger.info(f"Search completed in {iteration} iterations")
        logger.info(f"  Optimal threshold: {best_threshold:.4f}")
        logger.info(f"  Achieved hit rate: {best_hit_rate*100:.2f}%")
        logger.info(f"  Keys per query: {best_keys_per_query:.2f}")

    return {
        'optimal_threshold': best_threshold,
        'achieved_hit_rate': best_hit_rate,
        'keys_per_query': best_keys_per_query
    }


def run_pruning_experiment(
    num_kernels=3,
    mlp_hidden_dim=64,
    use_mlp=True,
    config_path=None,
    experiment_name=None,
    checkpoint_path=None,
    kappa_init=None,
    fixed_keys=None,
    logger=None
):
    """
    Run complete pruning experiment with model training and threshold search.

    Args:
        num_kernels: Number of von Mises kernels per frequency band (default: 3)
        mlp_hidden_dim: MLP hidden dimension (default: 64)
        use_mlp: If False, use average pooling instead of MLP (default: True)
        config_path: Path to config YAML file (default: config.yaml)
        experiment_name: Name for experiment output (default: auto-generated)
        checkpoint_path: Path to existing checkpoint (skip training if provided)
        kappa_init: Initial value for kappa (for num_kernels=1 experiments)
        fixed_keys: If set, use fixed-K evaluation instead of threshold search (default: None)
        logger: Logger instance

    Returns:
        dict with all experiment results and metrics
    """
    # Setup logger if not provided
    if logger is None:
        logger = setup_logging()

    # Load base configuration
    if config_path is None:
        config_path = EXP_DIR / 'config.yaml'
    else:
        config_path = Path(config_path)

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override model config with experiment parameters
    config['model']['num_kernels'] = num_kernels
    config['model']['mlp_hidden_dim'] = mlp_hidden_dim

    # Generate experiment name if not provided
    if experiment_name is None:
        mlp_type = 'mlp' if use_mlp else 'avgpool'
        if fixed_keys is not None:
            experiment_name = f'k{num_kernels}_h{mlp_hidden_dim}_{mlp_type}_fixedK{fixed_keys}'
        else:
            experiment_name = f'k{num_kernels}_h{mlp_hidden_dim}_{mlp_type}'

    logger.info("=" * 70)
    logger.info(f"EXPERIMENT: {experiment_name}")
    logger.info("=" * 70)
    logger.info(f"Configuration:")
    logger.info(f"  num_kernels: {num_kernels}")
    logger.info(f"  mlp_hidden_dim: {mlp_hidden_dim}")
    logger.info(f"  use_mlp: {use_mlp}")
    if fixed_keys is not None:
        logger.info(f"  fixed_keys: {fixed_keys} (fixed-K mode)")
    else:
        logger.info(f"  mode: auto-threshold search")
    logger.info("=" * 70)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['experiment']['seed'])

    # Load trace data
    trace_data = load_trace_data(config, logger)
    head_dim = trace_data['head_dim']

    # Create model
    if use_mlp:
        model = Module1KeyPruningNetwork(
            num_bins=config['model']['num_bins'],
            num_freqs=head_dim // 2,
            num_kernels=num_kernels,
            mlp_hidden=mlp_hidden_dim,
            anchor_positions=config['model']['position_anchors']
        )
    else:
        # For average pooling, create model with minimal MLP (1 hidden unit)
        # and rely on kernel encoding averaging
        logger.info("Using average pooling mode (MLP with hidden_dim=1)")
        model = Module1KeyPruningNetwork(
            num_bins=config['model']['num_bins'],
            num_freqs=head_dim // 2,
            num_kernels=num_kernels,
            mlp_hidden=1,  # Minimal MLP acts as weighted average
            anchor_positions=config['model']['position_anchors']
        )

    model = model.to(device)

    # Apply custom kappa initialization if specified (for num_kernels=1 experiments)
    if kappa_init is not None:
        with torch.no_grad():
            model.kernel_layer.kappa.fill_(kappa_init)
        logger.info(f"Applied custom kappa initialization: {kappa_init}")

    # Get parameter count
    param_counts = model.get_param_count()
    total_params = param_counts['total']
    logger.info(f"Model parameters: {param_counts}")
    logger.info(f"Total parameters: {total_params:,}")

    # Load checkpoint or train model
    if checkpoint_path is not None:
        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            logger.info(f"Loading checkpoint from: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            final_loss = checkpoint.get('loss', 0.0)
            logger.info(f"Checkpoint loaded. Loss from checkpoint: {final_loss:.6f}")
        else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
        # Train model
        final_loss = train_model(config, trace_data, model, device, logger)

    # Run evaluation based on mode
    if fixed_keys is not None:
        # Fixed-K evaluation mode
        eval_results = evaluate_model_with_fixed_k(
            model,
            trace_data,
            config,
            fixed_k=fixed_keys,
            device=device,
            logger=logger
        )

        # Compile results for fixed-K mode
        results = {
            'experiment_name': experiment_name,
            'config': {
                'num_kernels': num_kernels,
                'mlp_hidden_dim': mlp_hidden_dim if use_mlp else 1,
                'use_mlp': use_mlp,
                'fixed_keys': fixed_keys,
            },
            'metrics': {
                'param_count': total_params,
                'final_loss': final_loss,
                'fixed_k': fixed_keys,
                'hit_rate': eval_results['hit_rate'],
                'keys_per_query': eval_results['keys_per_query'],
            }
        }

        # Save results
        output_dir = EXP_DIR / 'output' / 'pruning_experiments'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{experiment_name}.json'

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 70)
        logger.info("EXPERIMENT RESULTS (Fixed-K Mode)")
        logger.info("=" * 70)
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Parameter count: {total_params:,}")
        logger.info(f"Final loss: {final_loss:.6f}")
        logger.info(f"Fixed K: {fixed_keys}")
        logger.info(f"Hit rate: {eval_results['hit_rate']*100:.2f}%")
        logger.info(f"Keys per query: {eval_results['keys_per_query']:.2f}")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 70)

    else:
        # Auto threshold search mode (default)
        threshold_results = auto_threshold_search(
            model,
            trace_data,
            config,
            target_hit_rate=0.995,
            precision=0.001,
            logger=logger
        )

        # Compile results
        results = {
            'experiment_name': experiment_name,
            'config': {
                'num_kernels': num_kernels,
                'mlp_hidden_dim': mlp_hidden_dim if use_mlp else 1,
                'use_mlp': use_mlp,
            },
            'metrics': {
                'param_count': total_params,
                'final_loss': final_loss,
                'optimal_threshold': threshold_results['optimal_threshold'],
                'hit_rate': threshold_results['achieved_hit_rate'],
                'keys_per_query': threshold_results['keys_per_query'],
            }
        }

        # Save results to JSON
        output_dir = EXP_DIR / 'output' / 'pruning_experiments'
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f'{experiment_name}.json'

        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)

        logger.info("=" * 70)
        logger.info("EXPERIMENT RESULTS")
        logger.info("=" * 70)
        logger.info(f"Experiment: {experiment_name}")
        logger.info(f"Parameter count: {total_params:,}")
        logger.info(f"Final loss: {final_loss:.6f}")
        logger.info(f"Optimal threshold: {threshold_results['optimal_threshold']:.4f}")
        logger.info(f"Hit rate: {threshold_results['achieved_hit_rate']*100:.2f}%")
        logger.info(f"Keys per query: {threshold_results['keys_per_query']:.2f}")
        logger.info(f"Results saved to: {output_path}")
        logger.info("=" * 70)

    return results


def main():
    """Main entry point for command-line execution."""
    parser = argparse.ArgumentParser(
        description="Run Module 1 Pruning Experiment with Auto Threshold Search"
    )
    parser.add_argument(
        '--num-kernels',
        type=int,
        default=1,
        help='Number of von Mises kernels per frequency band (default: 1, optimized)'
    )
    parser.add_argument(
        '--mlp-hidden-dim',
        type=int,
        default=64,
        help='MLP hidden dimension (default: 64)'
    )
    parser.add_argument(
        '--use-avg-pool',
        action='store_true',
        help='Use average pooling instead of MLP (default: False)'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        help='Path to config YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--experiment-name',
        type=str,
        default=None,
        help='Custom experiment name (default: auto-generated)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to existing checkpoint (skip training if provided)'
    )
    parser.add_argument(
        '--kappa-init',
        type=float,
        default=2.5,
        help='Initial value for kappa (default: 2.5, important for num_kernels=1)'
    )
    parser.add_argument(
        '--fixed-keys',
        type=int,
        default=None,
        help='Fixed number of keys to retain per query (default: None = use auto threshold search)'
    )

    args = parser.parse_args()

    # Setup logging
    log_dir = EXP_DIR / 'output' / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / 'pruning_experiments.log'
    logger = setup_logging(log_file=log_file)

    try:
        results = run_pruning_experiment(
            num_kernels=args.num_kernels,
            mlp_hidden_dim=args.mlp_hidden_dim,
            use_mlp=not args.use_avg_pool,
            config_path=args.config,
            experiment_name=args.experiment_name,
            checkpoint_path=args.checkpoint,
            kappa_init=args.kappa_init,
            fixed_keys=args.fixed_keys,
            logger=logger
        )

        logger.info("Experiment completed successfully!")
        return 0

    except Exception as e:
        logger.error(f"Experiment failed: {e}", exc_info=True)
        return 1


if __name__ == '__main__':
    sys.exit(main())
