"""
Training logic for Module 1 Neural Network Key Pruning.

Implements training loop with BCE loss and argmax-based labels.
"""

import json
import logging
import math
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml

from model import Module1KeyPruningNetwork


def setup_logging(config):
    """Setup logging configuration."""
    exp_dir = Path(__file__).parent
    log_dir = exp_dir / config['output']['logs_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'train.log'

    handlers = []
    if config['output']['log_to_file']:
        handlers.append(logging.FileHandler(log_file))
    if config['output']['log_to_console']:
        handlers.append(logging.StreamHandler(sys.stdout))

    logging.basicConfig(
        level=getattr(logging, config['output']['log_level']),
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
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
    trace_path = exp_dir / config['data']['trace_file']

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
    logger.info(f"Using head [layer={layer}, head={head}] for POC training")

    # Extract Q, K for selected head
    Q = qk_data['q'][layer, head]  # (seq_len, head_dim)
    K = qk_data['k'][layer, head]  # (seq_len, head_dim)

    seq_len, head_dim = Q.shape
    logger.info(f"Trace shape: Q={Q.shape}, K={K.shape}")

    # Compute attention matrix
    scale = head_dim ** -0.5
    attention = F.softmax(Q @ K.T * scale, dim=-1)  # (seq_len, seq_len)

    return {
        'Q': Q,
        'K': K,
        'attention': attention,
        'seq_len': seq_len,
        'head_dim': head_dim,
        'layer': layer,
        'head': head
    }


def extract_pruning_labels(attention_trace, round_start, round_end, seq_len):
    """
    Extract binary drop labels from attention trace.

    Label semantics:
        label=0: Key will be attended (should retain)
        label=1: Key will not be attended (should drop)

    Args:
        attention_trace: (seq_len, seq_len) attention weights
        round_start: Current round start position
        round_end: Current round end position (unused, but kept for API consistency)
        seq_len: Total sequence length

    Returns:
        labels: (round_start,) binary labels for each historical key
    """
    # Default: all historical keys should drop (label=1)
    labels = torch.ones(round_start, dtype=torch.float32)

    # Iterate over all queries from round_start to seq_len
    for q_idx in range(round_start, seq_len):
        # Get attention weights for historical keys only (< round_start)
        attn_weights = attention_trace[q_idx, :round_start]

        # Find argmax key
        argmax_key = attn_weights.argmax().item()

        # This key will be attended, so it should NOT be dropped
        labels[argmax_key] = 0

    return labels


def compute_module1_loss(drop_probs, labels, key_positions, seq_len, exclude_tail=1000):
    """
    Compute BCE loss with tail exclusion.

    Only keys with positions < (seq_len - exclude_tail) are included in loss.
    This avoids label noise from keys near sequence end.

    Args:
        drop_probs: (num_keys,) model predictions
        labels: (num_keys,) ground truth labels
        key_positions: (num_keys,) position indices
        seq_len: Total sequence length
        exclude_tail: Number of tail positions to exclude (default: 1000)

    Returns:
        Scalar loss tensor
    """
    # Create mask for valid keys (position < seq_len - exclude_tail)
    valid_mask = key_positions < (seq_len - exclude_tail)

    # Edge case: no valid keys
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=drop_probs.device)

    # Compute BCE loss only on valid keys
    loss = F.binary_cross_entropy(
        drop_probs[valid_mask],
        labels[valid_mask]
    )

    return loss


def train_epoch(model, trace_data, optimizer, config, device, logger):
    """
    Train for one epoch over all rounds in trace.

    Args:
        model: Module1KeyPruningNetwork instance
        trace_data: Dict with Q, K, attention, seq_len
        optimizer: Optimizer
        config: Configuration dict
        device: torch.device
        logger: Logger instance

    Returns:
        Average epoch loss
    """
    model.train()

    K = trace_data['K'].to(device)
    attention = trace_data['attention'].to(device)
    seq_len = trace_data['seq_len']
    head_dim = trace_data['head_dim']

    round_window = config['data']['round_window']
    exclude_tail = config['training']['exclude_tail_keys']

    epoch_loss = 0.0
    num_rounds = 0

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
        keys = K[:round_start]  # (round_start, head_dim)

        # Create position indices
        key_positions = torch.arange(round_start, device=device)

        # Compute reference angles
        reference_angles = model.kernel_layer._compute_reference_angles(
            round_start,
            round_window=round_window
        )

        # Forward pass
        drop_probs = model(keys, key_positions, reference_angles)

        # Compute loss
        loss = compute_module1_loss(drop_probs, labels, key_positions, seq_len, exclude_tail)

        # Skip if loss is zero (no valid keys)
        if loss.item() == 0.0:
            continue

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Accumulate loss
        epoch_loss += loss.item()
        num_rounds += 1

    # Return average loss
    avg_loss = epoch_loss / num_rounds if num_rounds > 0 else 0.0
    return avg_loss


def train(config, logger):
    """
    Main training loop.

    Args:
        config: Configuration dict
        logger: Logger instance

    Returns:
        Path to final checkpoint
    """
    logger.info("Initializing training...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(config['experiment']['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config['experiment']['seed'])

    # Load trace data
    trace_data = load_trace_data(config, logger)

    # Create model
    model_cfg = config['model']
    model = Module1KeyPruningNetwork(
        num_bins=model_cfg['num_bins'],
        num_freqs=trace_data['head_dim'] // 2,  # head_dim=128 -> 64 freqs
        num_kernels=model_cfg['num_kernels'],
        mlp_hidden=model_cfg['mlp_hidden_dim'],
        anchor_positions=model_cfg['position_anchors']
    )
    model = model.to(device)

    param_counts = model.get_param_count()
    logger.info(f"Model parameters: {param_counts}")
    logger.info(f"Total parameters: {param_counts['total']:,}")

    # Setup optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    # Setup directories
    exp_dir = Path(__file__).parent
    checkpoints_dir = exp_dir / config['output']['checkpoints_dir']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    # Training loop
    num_epochs = config['training']['epochs']
    logger.info(f"Starting training for {num_epochs} epochs...")

    for epoch in range(1, num_epochs + 1):
        # Train epoch
        avg_loss = train_epoch(model, trace_data, optimizer, config, device, logger)

        # Log progress
        logger.info(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}")

        # Save checkpoint periodically
        if epoch % config['training']['save_every'] == 0:
            checkpoint_path = checkpoints_dir / f'checkpoint_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    # Save final checkpoint
    final_checkpoint_path = checkpoints_dir / 'final_model.pt'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config,
    }, final_checkpoint_path)
    logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
    logger.info(f"Training completed. Final loss: {avg_loss:.6f}")

    return final_checkpoint_path


def main():
    """Main entry point."""
    # Load configuration
    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Setup logging
    logger = setup_logging(config)

    try:
        # Run training
        final_checkpoint = train(config, logger)
        logger.info(f"Training successful. Final checkpoint: {final_checkpoint}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
