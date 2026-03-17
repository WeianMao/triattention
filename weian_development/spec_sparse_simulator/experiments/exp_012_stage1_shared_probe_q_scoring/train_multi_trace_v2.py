"""
Multi-Trace Training for Module 2 (V2).

Based on train.py with correct RoPE layout and multi-trace support.

Key differences from train_multi_trace.py:
1. Uses correct RoPE front/back vector layout (not interleaved)
2. Uses model's YaRN-scaled inv_freq
3. Properly handles use_l2_norm and invert_to_origin flags
4. Supports weight decay toward initialization
"""

import json
import logging
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from model import create_model
from compute_kmeans_init_multi_trace_v2 import (
    compute_kmeans_init_multi_trace,
    get_training_trace_paths,
)
from compute_kmeans_init import compute_magnitude_init


def setup_logging(config):
    """Setup logging configuration."""
    exp_dir = Path(__file__).parent
    log_dir = exp_dir / config['output']['logs_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'train_multi_trace_v2.log'

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


def load_single_head_trace(trace_path, layer, head, logger):
    """
    Load trace data and immediately extract only the required head.

    This prevents OOM by discarding unused heads before they accumulate.
    """
    logger.info(f"Loading trace: {trace_path}")

    qk_data = torch.load(trace_path, map_location='cpu')

    Q = qk_data['q'][layer, head].clone()
    K = qk_data['k'][layer, head].clone()

    del qk_data

    seq_len, head_dim = Q.shape
    logger.info(f"  Extracted head [{layer}, {head}]: seq_len={seq_len}, head_dim={head_dim}")

    return {'Q': Q, 'K': K, 'seq_len': seq_len, 'head_dim': head_dim}


def compute_attention_matrix(Q, K):
    """Compute causal attention matrix."""
    seq_len, head_dim = Q.shape
    scale = head_dim ** -0.5

    attention_logits = Q @ K.T * scale
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention_logits.masked_fill_(causal_mask, float('-inf'))
    attention = F.softmax(attention_logits, dim=-1)

    return attention


def extract_query_to_key_labels(attention_trace, round_start, round_end, seq_len, exclude_tail=1000):
    """Extract argmax key indices for each query in the round."""
    valid_end = min(seq_len - exclude_tail, seq_len)

    query_indices = []
    argmax_keys = []
    argmax_in_recent = []

    for q_idx in range(round_start, min(round_end, valid_end)):
        attn_weights = attention_trace[q_idx, :q_idx + 1]
        argmax_key = attn_weights.argmax().item()

        query_indices.append(q_idx)
        argmax_keys.append(argmax_key)
        argmax_in_recent.append(argmax_key >= round_start)

    if len(query_indices) == 0:
        return None

    return {
        'query_indices': torch.tensor(query_indices, dtype=torch.long),
        'argmax_keys': torch.tensor(argmax_keys, dtype=torch.long),
        'argmax_in_recent': torch.tensor(argmax_in_recent, dtype=torch.bool)
    }


def compute_attraction_loss(key_probs, query_bin_probs, argmax_keys, argmax_in_recent, eps=1e-8):
    """
    Compute Attraction Loss for Module 2.

    Loss = -log(sum_bins(p_q[b] * P[argmax_key, b]) + eps).mean()
    """
    valid_mask = ~argmax_in_recent

    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=key_probs.device, requires_grad=True)

    valid_query_probs = query_bin_probs[valid_mask]
    valid_argmax_keys = argmax_keys[valid_mask]

    P_matched = key_probs[valid_argmax_keys]
    match_prob = (valid_query_probs * P_matched).sum(dim=1)

    loss = -torch.log(match_prob + eps).mean()
    return loss


def compute_init_regularization_loss(model, init_params):
    """
    Compute regularization loss that pulls weights toward initialization.

    L_reg = sum((w - w_init)^2)
    """
    reg_loss = 0.0
    for name, param in model.named_parameters():
        if name in init_params:
            reg_loss = reg_loss + ((param - init_params[name]) ** 2).sum()
    return reg_loss


def train_on_trace(model, trace_data, optimizer, config, device, logger,
                   init_params=None, weight_decay=0.0):
    """
    Train on a single trace for one pass.

    Args:
        model: Module2Network instance
        trace_data: Dict with Q, K for single head
        optimizer: Optimizer
        config: Configuration dict
        device: torch.device
        logger: Logger instance
        init_params: Dict of initial parameter values for regularization
        weight_decay: Regularization strength

    Returns:
        Average loss for this trace
    """
    model.train()

    Q = trace_data['Q']
    K = trace_data['K']
    seq_len = trace_data['seq_len']

    # Compute attention matrix on the fly
    attention = compute_attention_matrix(Q, K)

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']
    query_batch_size = config['training'].get('query_batch_size', 64)

    trace_loss = 0.0
    num_batches = 0

    for round_start in range(0, seq_len, round_window):
        round_end = min(round_start + round_window, seq_len)

        if round_start == 0:
            continue

        labels = extract_query_to_key_labels(
            attention, round_start, round_end, seq_len, exclude_tail
        )

        if labels is None:
            continue

        query_indices = labels['query_indices']
        argmax_keys = labels['argmax_keys']
        argmax_in_recent = labels['argmax_in_recent']

        historical_keys = K[:round_start].to(device)
        reference_angles = model.compute_reference_angles(round_start, round_window)

        key_probs = model.forward_keys(historical_keys, reference_angles)
        empty_bin_mask = (key_probs.sum(dim=0) == 0).detach()

        del historical_keys
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        num_queries = len(query_indices)
        round_loss = 0.0
        valid_batches = 0

        for batch_start in range(0, num_queries, query_batch_size):
            batch_end = min(batch_start + query_batch_size, num_queries)

            batch_query_indices = query_indices[batch_start:batch_end]
            batch_argmax_keys = argmax_keys[batch_start:batch_end].to(device)
            batch_argmax_in_recent = argmax_in_recent[batch_start:batch_end].to(device)

            batch_queries = Q[batch_query_indices].to(device)
            query_bin_probs = model.forward_queries(batch_queries, reference_angles, empty_bin_mask)

            batch_loss = compute_attraction_loss(key_probs, query_bin_probs, batch_argmax_keys, batch_argmax_in_recent)

            if batch_loss.item() == 0.0:
                del batch_queries, query_bin_probs
                continue

            round_loss = round_loss + batch_loss
            valid_batches += 1

            del batch_queries, query_bin_probs, batch_loss

        if valid_batches > 0:
            optimizer.zero_grad()

            avg_attraction_loss = round_loss / valid_batches

            if init_params is not None and weight_decay > 0:
                reg_loss = compute_init_regularization_loss(model, init_params)
                total_loss = avg_attraction_loss + weight_decay * reg_loss
            else:
                total_loss = avg_attraction_loss

            total_loss.backward()
            optimizer.step()

            trace_loss += total_loss.item()
            num_batches += 1

        del key_probs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    del attention

    avg_loss = trace_loss / num_batches if num_batches > 0 else 0.0
    return avg_loss


def train_epoch_multi_trace(model, all_traces, optimizer, config, device, logger,
                            init_params=None, weight_decay=0.0):
    """
    Train for one epoch over all preloaded traces.

    Returns:
        Average epoch loss across all traces
    """
    epoch_loss = 0.0
    num_traces = 0

    for trace_data in all_traces:
        trace_loss = train_on_trace(
            model, trace_data, optimizer, config, device, logger,
            init_params=init_params, weight_decay=weight_decay
        )

        if trace_loss > 0:
            epoch_loss += trace_loss
            num_traces += 1
            logger.info(f"  {trace_data['name']}: loss={trace_loss:.6f}")

        if device.type == 'cuda':
            torch.cuda.empty_cache()

    avg_loss = epoch_loss / num_traces if num_traces > 0 else 0.0
    return avg_loss


def preload_all_traces(trace_paths, layer, head, logger):
    """
    Preload all traces at startup, keeping only the required head.
    """
    logger.info(f"Preloading {len(trace_paths)} traces (keeping only head [{layer}, {head}])...")

    all_traces = []
    total_queries = 0

    for i, trace_path in enumerate(trace_paths):
        logger.info(f"  [{i+1}/{len(trace_paths)}] Loading {trace_path.parent.name}...")

        qk_data = torch.load(trace_path, map_location='cpu')

        Q = qk_data['q'][layer, head].clone()
        K = qk_data['k'][layer, head].clone()

        del qk_data

        seq_len = Q.shape[0]
        total_queries += seq_len

        trace_data = {
            'Q': Q,
            'K': K,
            'seq_len': seq_len,
            'head_dim': Q.shape[1],
            'name': trace_path.parent.name
        }
        all_traces.append(trace_data)

        logger.info(f"       seq_len={seq_len}")

    logger.info(f"Preloading complete. Total sequences: {total_queries}")
    return all_traces


def train(config, logger, use_l2_norm=False, invert_to_origin=False, weight_decay=0.0):
    """
    Main training loop with multi-trace support.

    Args:
        config: Configuration dict
        logger: Logger instance
        use_l2_norm: If True, use L2 normalization (default: False)
        invert_to_origin: If True, invert to origin (default: False)
        weight_decay: Regularization strength (default: 0.0)

    Returns:
        Path to final checkpoint
    """
    logger.info("Initializing Multi-Trace Module 2 training (V2)...")
    logger.info(f"Settings: use_l2_norm={use_l2_norm}, invert_to_origin={invert_to_origin}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    exp_dir = Path(__file__).parent

    # Get head info
    head_sample_path = exp_dir / config['data']['head_sample_file']
    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    # Get training trace paths
    test_trace_path = exp_dir / config['data']['test_trace_path']
    trace_paths = get_training_trace_paths(test_trace_path, logger)

    # Preload all traces
    all_traces = preload_all_traces(trace_paths, layer, head, logger)

    # Compute K-means initialization using ALL training traces
    use_kmeans_init = config.get('model', {}).get('use_kmeans_init', True)
    init_probes = None

    if use_kmeans_init:
        logger.info("Computing K-means initialization from ALL training traces...")
        init_probes = compute_kmeans_init_multi_trace(
            config, logger, n_clusters=config['model']['num_bins'],
            use_l2_norm=use_l2_norm, invert_to_origin=invert_to_origin
        )
        logger.info(f"K-means initialization completed. Probe shape: {init_probes.shape}")

    # Create model
    model = create_model(config, init_probes=init_probes, use_l2_norm=use_l2_norm)
    model = model.to(device)

    # Save initial parameters for regularization
    init_params = None
    if weight_decay > 0:
        init_params = {name: param.detach().clone() for name, param in model.named_parameters()}
        logger.info(f"Saved initial parameters for regularization (weight_decay={weight_decay})")

    param_counts = model.get_param_count()
    logger.info(f"Model parameters: {param_counts}")
    logger.info(f"Total parameters: {param_counts['total']:,}")

    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate']
    )

    checkpoints_dir = exp_dir / config['output']['checkpoints_dir']
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    num_epochs = config['training']['epochs']
    log_every = config['training'].get('log_every', 1)
    save_every = config['training'].get('save_every', 10)

    logger.info(f"Starting training for {num_epochs} epochs on {len(all_traces)} traces...")

    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        logger.info(f"=== Epoch {epoch}/{num_epochs} ===")

        avg_loss = train_epoch_multi_trace(
            model, all_traces, optimizer, config, device, logger,
            init_params=init_params, weight_decay=weight_decay
        )

        if epoch % log_every == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{num_epochs} - Average Loss: {avg_loss:.6f}")

        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            best_checkpoint_path = checkpoints_dir / 'best_model_multi_trace.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
                'num_traces': len(all_traces),
            }, best_checkpoint_path)

        if epoch % save_every == 0:
            checkpoint_path = checkpoints_dir / f'checkpoint_multi_trace_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
                'num_traces': len(all_traces),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

    final_checkpoint_path = checkpoints_dir / 'final_model_multi_trace.pt'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'config': config,
        'num_traces': len(all_traces),
    }, final_checkpoint_path)
    logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
    logger.info(f"Training completed. Final loss: {avg_loss:.6f}, Best loss: {best_loss:.6f}")

    return final_checkpoint_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Trace Module 2 Training (V2)')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=0.0,
        help='Regularization strength for pulling weights toward initialization. '
             '(default: 0.0, no regularization)'
    )
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    logger = setup_logging(config)

    # Fixed settings based on RoPE layout fix experiments
    use_l2_norm = False
    invert_to_origin = False
    logger.info(f"Using fixed settings: use_l2_norm={use_l2_norm}, invert_to_origin={invert_to_origin}")
    logger.info(f"Weight decay (toward init): {args.weight_decay}")

    try:
        final_checkpoint = train(
            config, logger,
            use_l2_norm=use_l2_norm,
            invert_to_origin=invert_to_origin,
            weight_decay=args.weight_decay
        )
        logger.info(f"Training successful. Final checkpoint: {final_checkpoint}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
