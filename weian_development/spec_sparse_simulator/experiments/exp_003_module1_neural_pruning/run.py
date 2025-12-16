"""
Main entry point for Experiment 003: Module 1 Neural Network Key Pruning.

Supports 3 modes:
- train: Train model and save checkpoint
- evaluate: Evaluate checkpoint and compute metrics
- inference: Run reference implementation inference for validation
"""

import argparse
import json
import logging
import sys
from pathlib import Path
import yaml
import torch
import torch.nn.functional as F

# Ensure experiment directory is in path for imports
EXP_DIR = Path(__file__).parent
sys.path.insert(0, str(EXP_DIR))

# Add parent directory to path for reference implementation
ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT))

from model import Module1KeyPruningNetwork
from train import train, setup_logging as train_setup_logging


def setup_logging(log_level='INFO', log_to_console=True, log_to_file=True, log_file=None):
    """Setup logging configuration."""
    handlers = []
    if log_to_console:
        handlers.append(logging.StreamHandler(sys.stdout))
    if log_to_file and log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


def load_config(config_path):
    """Load experiment configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def run_train_mode(config, logger):
    """
    Execute training mode.

    Calls train() from train.py and saves checkpoint to output/checkpoints/.

    Args:
        config: Configuration dict
        logger: Logger instance

    Returns:
        Path to final checkpoint
    """
    logger.info("=" * 60)
    logger.info("MODE: TRAIN")
    logger.info("=" * 60)

    # Setup training logging
    train_logger = train_setup_logging(config)

    # Run training
    checkpoint_path = train(config, train_logger)

    logger.info(f"Training completed successfully.")
    logger.info(f"Final checkpoint saved to: {checkpoint_path}")
    logger.info("\nNext steps:")
    logger.info(f"  1. Run evaluation: python run.py --mode evaluate --checkpoint {checkpoint_path}")
    logger.info(f"  2. Check training logs: tail -f output/logs/train.log")

    return checkpoint_path


def load_trace_data_for_eval(config, logger):
    """
    Load trace data for evaluation.

    Args:
        config: Configuration dict
        logger: Logger instance

    Returns:
        dict with Q, K tensors and metadata
    """
    trace_path = EXP_DIR / config['data']['trace_file']
    head_sample_path = EXP_DIR / config['data']['head_sample_file']

    if not trace_path.exists():
        raise FileNotFoundError(f"Trace file not found: {trace_path}")
    if not head_sample_path.exists():
        raise FileNotFoundError(f"Head sample file not found: {head_sample_path}")

    logger.info(f"Loading trace data from: {trace_path}")
    qk_data = torch.load(trace_path, map_location='cpu')

    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    # Use first head for POC
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    Q = qk_data['q'][layer, head]
    K = qk_data['k'][layer, head]
    seq_len, head_dim = Q.shape

    # Compute full attention matrix for ground truth
    scale = head_dim ** -0.5
    attention = F.softmax(Q @ K.T * scale, dim=-1)

    return {
        'Q': Q,
        'K': K,
        'attention': attention,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'layer': layer,
        'head': head
    }


def compute_argmax_hit_rate_from_prune_mask(prune_mask, attention):
    """
    Compute Argmax Hit Rate from prune mask and attention matrix.

    Args:
        prune_mask: (seq_len, seq_len) boolean tensor, True = key retained
        attention: (seq_len, seq_len) attention weights

    Returns:
        float: Hit rate (0.0 to 1.0)
    """
    seq_len = attention.shape[0]
    hits = 0
    total = 0

    for q_idx in range(seq_len):
        # Get valid keys (causal mask: keys <= q_idx)
        valid_keys = torch.arange(q_idx + 1)

        # Get attention weights for valid keys
        attn_weights = attention[q_idx, valid_keys]

        # Find argmax key
        argmax_key = attn_weights.argmax().item()

        # Check if argmax key is retained in prune mask
        if prune_mask[q_idx, argmax_key]:
            hits += 1
        total += 1

    return hits / total if total > 0 else 0.0


def run_evaluate_mode(config, checkpoint_path, threshold, logger):
    """
    Execute evaluation mode.

    Loads checkpoint and computes metrics:
    - Argmax Hit Rate
    - Keys per Query
    - Computation Reduction
    - Retention Rate

    Args:
        config: Configuration dict
        checkpoint_path: Path to checkpoint file
        threshold: Drop probability threshold
        logger: Logger instance

    Returns:
        dict: Evaluation results
    """
    logger.info("=" * 60)
    logger.info("MODE: EVALUATE")
    logger.info("=" * 60)

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load checkpoint
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    logger.info(f"Loading checkpoint: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Create model
    trace_data = load_trace_data_for_eval(config, logger)
    head_dim = trace_data['head_dim']

    model = Module1KeyPruningNetwork(
        num_bins=config['model']['num_bins'],
        num_freqs=head_dim // 2,
        num_kernels=config['model']['num_kernels'],
        mlp_hidden=config['model']['mlp_hidden_dim'],
        anchor_positions=config['model']['position_anchors']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    logger.info("Model loaded successfully")

    # Run inference on trace
    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']
    round_window = config['data']['round_window']

    # Create prune mask by running model on all rounds
    prune_mask = torch.zeros((seq_len, seq_len), dtype=torch.bool, device=device)
    total_keys_evaluated = 0
    total_keys_retained = 0

    logger.info(f"Running inference over {seq_len // round_window + 1} rounds...")

    with torch.no_grad():
        for round_start in range(0, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)

            if round_start == 0:
                # No historical keys yet
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
                # Causal mask: can only attend to keys <= q_idx
                valid_keys = key_positions <= q_idx

                # Combine retention mask with causal mask
                prune_mask[q_idx, :round_start] = retain_mask.squeeze() & valid_keys

                # Always retain current token
                prune_mask[q_idx, q_idx] = True

            total_keys_evaluated += round_start
            total_keys_retained += retain_mask.sum().item()

    # Compute metrics
    logger.info("Computing metrics...")

    # 1. Argmax Hit Rate
    hit_rate = compute_argmax_hit_rate_from_prune_mask(prune_mask, attention)

    # 2. Keys per Query
    keys_per_query = prune_mask.sum(dim=1).float().mean().item()

    # 3. Retention Rate
    retention_rate = (total_keys_retained / total_keys_evaluated * 100.0) if total_keys_evaluated > 0 else 0.0

    # 4. Computation Reduction
    total_possible_keys = sum(range(seq_len + 1))  # Triangular number
    total_retained_keys = prune_mask.sum().item()
    computation_reduction = ((total_possible_keys - total_retained_keys) / total_possible_keys * 100.0) if total_possible_keys > 0 else 0.0

    results = {
        'argmax_hit_rate': hit_rate,
        'keys_per_query': keys_per_query,
        'retention_rate': retention_rate,
        'computation_reduction': computation_reduction,
        'threshold': threshold,
        'seq_len': seq_len,
        'total_keys_evaluated': total_keys_evaluated,
        'total_keys_retained': total_keys_retained
    }

    # Log results
    logger.info("\n" + "=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Argmax Hit Rate:        {hit_rate * 100:.2f}%")
    logger.info(f"Keys per Query:         {keys_per_query:.2f}")
    logger.info(f"Retention Rate:         {retention_rate:.2f}%")
    logger.info(f"Computation Reduction:  {computation_reduction:.2f}%")
    logger.info(f"Threshold:              {threshold}")
    logger.info("=" * 60)

    # Save results
    results_dir = EXP_DIR / config['output']['results_dir']
    results_dir.mkdir(parents=True, exist_ok=True)
    results_path = results_dir / 'metrics.json'

    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)

    logger.info(f"\nResults saved to: {results_path}")

    return results


def run_inference_mode(config, checkpoint_path, threshold, logger):
    """
    Execute inference mode using reference implementation's compute_pooled_attention.

    Compares sparse attention (with pruning) vs full attention (baseline).

    Args:
        config: Configuration dict
        checkpoint_path: Path to checkpoint file
        threshold: Drop probability threshold
        logger: Logger instance

    Returns:
        dict: Inference comparison results
    """
    logger.info("=" * 60)
    logger.info("MODE: INFERENCE (Reference Implementation Validation)")
    logger.info("=" * 60)

    # Import reference implementation
    try:
        from weian_development.spec_sparse_simulator.attention_pruning_case_study_hybrid_rounds_xtrace import (
            compute_pooled_attention
        )
        logger.info("Successfully imported compute_pooled_attention from reference implementation")
    except ImportError as e:
        logger.error(f"Failed to import reference implementation: {e}")
        raise

    logger.info("Inference mode would use reference implementation's compute_pooled_attention")
    logger.info("This mode is for validation purposes and is not yet fully implemented.")
    logger.info("\nFor now, use evaluate mode for metrics computation.")

    return {'status': 'not_implemented'}


def update_readme(results):
    """
    Update README.md with experiment results.

    Updates the '结果摘要' section with actual metrics.

    Args:
        results: dict with evaluation results
    """
    readme_path = EXP_DIR / 'README.md'

    if not readme_path.exists():
        logging.warning(f"README.md not found: {readme_path}")
        return

    with open(readme_path, 'r', encoding='utf-8') as f:
        content = f.read()

    # Update 结果摘要 section
    hit_rate = results.get('argmax_hit_rate', 0.0) * 100
    keys_per_query = results.get('keys_per_query', 0.0)
    retention_rate = results.get('retention_rate', 0.0)
    computation_reduction = results.get('computation_reduction', 0.0)

    # Find and replace 结果摘要 section
    import re

    new_results_section = f"""## 结果摘要
- Argmax Hit Rate: {hit_rate:.2f}%
- Keys per Query: {keys_per_query:.2f}
- Retention Rate: {retention_rate:.2f}%
- Computation Reduction: {computation_reduction:.2f}%"""

    # Replace existing 结果摘要 section
    pattern = r'## 结果摘要.*?(?=\n## |\Z)'
    content = re.sub(pattern, new_results_section, content, flags=re.DOTALL)

    # Update 结论 section based on hit rate
    if hit_rate >= 99.0:
        conclusion_text = f"""## 结论
成功验证Module 1神经网络架构的有效性。Argmax Hit Rate达到{hit_rate:.2f}%，超过99%目标。模型能够准确预测Key重要性，实现{computation_reduction:.2f}%的计算量减少。"""
    else:
        conclusion_text = f"""## 结论
Module 1神经网络架构需要进一步调优。当前Argmax Hit Rate为{hit_rate:.2f}%，未达到99%目标。建议调整超参数或增加训练轮数。"""

    pattern = r'## 结论.*?(?=\n## |\Z)'
    content = re.sub(pattern, conclusion_text, content, flags=re.DOTALL)

    # Write updated README
    with open(readme_path, 'w', encoding='utf-8') as f:
        f.write(content)

    logging.info(f"README.md updated with results: {readme_path}")


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Experiment 003: Module 1 Neural Network Key Pruning"
    )
    parser.add_argument(
        '--mode',
        type=str,
        required=True,
        choices=['train', 'evaluate', 'inference'],
        help='Execution mode: train, evaluate, or inference'
    )
    parser.add_argument(
        '--config',
        type=str,
        default='config.yaml',
        help='Path to config YAML file (default: config.yaml)'
    )
    parser.add_argument(
        '--checkpoint',
        type=str,
        default=None,
        help='Path to checkpoint file (required for evaluate/inference modes)'
    )
    parser.add_argument(
        '--threshold',
        type=float,
        default=0.5,
        help='Drop probability threshold for evaluation (default: 0.5)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='output/',
        help='Output directory (default: output/)'
    )

    args = parser.parse_args()

    # Load configuration
    config_path = EXP_DIR / args.config
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)

    # Setup logging
    log_file = EXP_DIR / config['output']['logs_dir'] / 'run.log'
    logger = setup_logging(
        log_level=config['output']['log_level'],
        log_to_console=config['output']['log_to_console'],
        log_to_file=config['output']['log_to_file'],
        log_file=log_file
    )

    logger.info(f"Experiment: {config['experiment']['name']}")
    logger.info(f"Mode: {args.mode}")
    logger.info(f"Config: {config_path}")

    # Execute based on mode
    try:
        if args.mode == 'train':
            checkpoint_path = run_train_mode(config, logger)
            logger.info("\nTraining completed successfully!")
            sys.exit(0)

        elif args.mode == 'evaluate':
            # Validate checkpoint argument
            if args.checkpoint is None:
                # Try to find final checkpoint
                checkpoint_path = EXP_DIR / config['output']['checkpoints_dir'] / 'final_model.pt'
                if not checkpoint_path.exists():
                    logger.error("No checkpoint specified and default checkpoint not found.")
                    logger.error("Please specify checkpoint with --checkpoint argument.")
                    sys.exit(1)
            else:
                checkpoint_path = Path(args.checkpoint)
                if not checkpoint_path.is_absolute():
                    checkpoint_path = EXP_DIR / checkpoint_path

            results = run_evaluate_mode(config, checkpoint_path, args.threshold, logger)

            # Update README with results
            update_readme(results)

            logger.info("\nEvaluation completed successfully!")
            sys.exit(0)

        elif args.mode == 'inference':
            # Validate checkpoint argument
            if args.checkpoint is None:
                logger.error("Checkpoint required for inference mode. Use --checkpoint argument.")
                sys.exit(1)

            checkpoint_path = Path(args.checkpoint)
            if not checkpoint_path.is_absolute():
                checkpoint_path = EXP_DIR / checkpoint_path

            results = run_inference_mode(config, checkpoint_path, args.threshold, logger)
            logger.info("\nInference mode completed!")
            sys.exit(0)

    except Exception as e:
        logger.error(f"Error during {args.mode} mode: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
