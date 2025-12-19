"""
Training logic for Module 2 Multi-Bin Sparse Attention with Expanded Training Set.

Implements training loop with Attraction Loss:
    loss = -log(sum_bins(p_q[b] * P[argmax_key, b])).mean()

This loss encourages the Query's selected bin to contain its argmax Key.

Expanded Training Set Version:
- Loads multiple trace files as training data
- Filters unused heads immediately after loading to save memory
- Keeps test trace separate from training traces
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

from model import Module2Network, create_model


def setup_logging(config):
    """Setup logging configuration."""
    exp_dir = Path(__file__).parent
    log_dir = exp_dir / config['output']['logs_dir']
    log_dir.mkdir(parents=True, exist_ok=True)

    log_file = log_dir / 'train.log'

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


def load_multiple_training_traces(config, logger):
    """
    Load multiple trace files for training, filtering to only required heads.

    Args:
        config: Configuration dict
        logger: Logger instance

    Returns:
        List of dicts, each containing Q, K, attention, seq_len, head_dim for one trace
    """
    exp_dir = Path(__file__).parent

    # Load head sample file
    head_sample_path = exp_dir / config['data']['head_sample_file']
    if not head_sample_path.exists():
        raise FileNotFoundError(f"Head sample file not found: {head_sample_path}")

    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)

    logger.info(f"Using {len(head_samples)} heads for training")

    # Get training trace paths
    trace_paths = config['data']['training_traces']

    all_trace_data = []

    for trace_path_str in trace_paths:
        trace_path = Path(trace_path_str)
        if not trace_path.exists():
            logger.warning(f"Trace file not found, skipping: {trace_path}")
            continue

        # Load and filter
        filtered_data = load_qk_filtered_by_heads(trace_path, head_samples, logger)

        # Process each head and compute attention
        for head_idx, (layer, head) in enumerate(head_samples):
            Q = filtered_data['q'][head_idx]  # (seq_len, head_dim)
            K = filtered_data['k'][head_idx]  # (seq_len, head_dim)
            seq_len = filtered_data['seq_len']
            head_dim = filtered_data['head_dim']

            # Compute attention matrix with causal mask
            scale = head_dim ** -0.5
            attention_logits = Q @ K.T * scale  # (seq_len, seq_len)

            # Apply causal mask: keys must be <= query position
            causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
            attention_logits.masked_fill_(causal_mask, float('-inf'))

            attention = F.softmax(attention_logits, dim=-1)  # (seq_len, seq_len)

            all_trace_data.append({
                'Q': Q,
                'K': K,
                'attention': attention,
                'seq_len': seq_len,
                'head_dim': head_dim,
                'layer': layer,
                'head': head,
                'source': str(trace_path)
            })

        # Delete filtered data to free memory
        del filtered_data

    logger.info(f"Loaded {len(all_trace_data)} training samples from {len(trace_paths)} traces")
    return all_trace_data


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


def extract_query_to_key_labels(attention_trace, round_start, round_end, seq_len, exclude_tail=1000):
    """
    Extract argmax key indices for each query in the round.

    For Module 2, we need to know which Key each Query should attend to (argmax).

    Args:
        attention_trace: (seq_len, seq_len) attention weights
        round_start: Current round start position
        round_end: Current round end position
        seq_len: Total sequence length
        exclude_tail: Number of tail queries to exclude (default: 1000)

    Returns:
        dict with:
            - query_indices: (num_valid_queries,) query position indices
            - argmax_keys: (num_valid_queries,) argmax key index for each query
            - argmax_in_recent: (num_valid_queries,) bool, True if argmax is in current round
    """
    # Determine valid query range (exclude tail)
    valid_end = min(seq_len - exclude_tail, seq_len)

    query_indices = []
    argmax_keys = []
    argmax_in_recent = []

    # Iterate over queries in current round [round_start, round_end)
    # but only include queries where we have historical keys (round_start > 0)
    for q_idx in range(round_start, min(round_end, valid_end)):
        # Get attention weights for all keys <= q_idx (causal)
        attn_weights = attention_trace[q_idx, :q_idx + 1]

        # Find argmax key
        argmax_key = attn_weights.argmax().item()

        query_indices.append(q_idx)
        argmax_keys.append(argmax_key)

        # Check if argmax is in current round (recent keys: >= round_start)
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

    This encourages the Query's bin selection to overlap with its argmax Key's bin assignment.

    Args:
        key_probs: (num_keys, num_bins) Key bin probabilities (softmax over keys, dim=0)
        query_bin_probs: (num_queries, num_bins) Query bin probabilities (softmax over bins, dim=-1)
        argmax_keys: (num_queries,) argmax key index for each query (into historical keys)
        argmax_in_recent: (num_queries,) bool, True if argmax is in current round (exclude from loss)
        eps: Small value for numerical stability

    Returns:
        Scalar loss tensor
    """
    # Exclude queries whose argmax is in recent keys (they don't need routing)
    valid_mask = ~argmax_in_recent

    if valid_mask.sum() == 0:
        # No valid queries (all argmax in recent keys)
        return torch.tensor(0.0, device=key_probs.device, requires_grad=True)

    # Filter to valid queries
    valid_query_probs = query_bin_probs[valid_mask]  # (num_valid, num_bins)
    valid_argmax_keys = argmax_keys[valid_mask]  # (num_valid,)

    # Get key probabilities for argmax keys
    # key_probs: (num_keys, num_bins)
    # valid_argmax_keys: (num_valid,) - indices into historical keys
    P_matched = key_probs[valid_argmax_keys]  # (num_valid, num_bins)

    # Compute matching probability: sum over bins of (query_prob * key_prob)
    # This is the probability that query and its argmax key are in the same bin
    match_prob = (valid_query_probs * P_matched).sum(dim=1)  # (num_valid,)

    # Attraction Loss: -log(match_prob)
    loss = -torch.log(match_prob + eps).mean()

    return loss


def train_epoch_on_trace(model, trace_data, optimizer, config, device, logger):
    """
    Train for one pass over all rounds in a single trace.

    Memory-optimized version: keeps data on CPU, only moves batches to device.

    Args:
        model: Module2Network instance
        trace_data: Dict with Q, K, attention, seq_len
        optimizer: Optimizer
        config: Configuration dict
        device: torch.device
        logger: Logger instance

    Returns:
        Tuple of (total_loss, num_batches)
    """
    model.train()

    # Keep data on CPU to save memory
    Q = trace_data['Q']
    K = trace_data['K']
    attention = trace_data['attention']
    seq_len = trace_data['seq_len']

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']
    query_batch_size = config['training'].get('query_batch_size', 64)

    total_loss = 0.0
    num_batches = 0

    # Iterate over rounds
    for round_start in range(0, seq_len, round_window):
        round_end = min(round_start + round_window, seq_len)

        # Skip first round (no historical keys)
        if round_start == 0:
            continue

        # Extract query-to-key labels for this round
        labels = extract_query_to_key_labels(
            attention, round_start, round_end, seq_len, exclude_tail
        )

        if labels is None:
            continue

        query_indices = labels['query_indices']
        argmax_keys = labels['argmax_keys']
        argmax_in_recent = labels['argmax_in_recent']

        # Get historical keys (< round_start) - move to device
        historical_keys = K[:round_start].to(device)

        # Compute reference angles for this round
        reference_angles = model.compute_reference_angles(round_start, round_window)

        # Forward pass: Key network on historical keys
        key_probs = model.forward_keys(historical_keys, reference_angles)  # (round_start, num_bins)
        empty_bin_mask = (key_probs.sum(dim=0) == 0).detach()  # (num_bins,) - detach mask only

        # Clear GPU memory from key forward pass
        del historical_keys
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Process queries in batches, accumulate loss for one backward per round
        num_queries = len(query_indices)
        round_loss = 0.0
        valid_batches = 0

        for batch_start in range(0, num_queries, query_batch_size):
            batch_end = min(batch_start + query_batch_size, num_queries)

            batch_query_indices = query_indices[batch_start:batch_end]
            batch_argmax_keys = argmax_keys[batch_start:batch_end].to(device)
            batch_argmax_in_recent = argmax_in_recent[batch_start:batch_end].to(device)

            # Get queries for this batch
            batch_queries = Q[batch_query_indices].to(device)

            # Forward pass: Query network
            query_bin_probs = model.forward_queries(batch_queries, reference_angles, empty_bin_mask)

            # Compute Attraction Loss
            batch_loss = compute_attraction_loss(key_probs, query_bin_probs, batch_argmax_keys, batch_argmax_in_recent)

            # Skip if loss is zero (no valid queries)
            if batch_loss.item() == 0.0:
                del batch_queries, query_bin_probs
                continue

            # Accumulate loss
            round_loss = round_loss + batch_loss
            valid_batches += 1

            # Clear batch tensors (except loss which we need for backward)
            del batch_queries, query_bin_probs, batch_loss

        # Backward pass once per round (if we have valid losses)
        if valid_batches > 0:
            optimizer.zero_grad()
            (round_loss / valid_batches).backward()
            optimizer.step()

            total_loss += (round_loss / valid_batches).item()
            num_batches += 1

        # Clear key probs after processing all query batches
        del key_probs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    return total_loss, num_batches


def train_epoch(model, all_trace_data, optimizer, config, device, logger):
    """
    Train for one epoch over all traces.

    Args:
        model: Module2Network instance
        all_trace_data: List of trace data dicts
        optimizer: Optimizer
        config: Configuration dict
        device: torch.device
        logger: Logger instance

    Returns:
        Average epoch loss
    """
    total_loss = 0.0
    total_batches = 0

    for trace_idx, trace_data in enumerate(all_trace_data):
        trace_loss, trace_batches = train_epoch_on_trace(
            model, trace_data, optimizer, config, device, logger
        )
        total_loss += trace_loss
        total_batches += trace_batches

    avg_loss = total_loss / total_batches if total_batches > 0 else 0.0
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
    logger.info("Initializing Module 2 training with expanded training set...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Load all training trace data
    all_trace_data = load_multiple_training_traces(config, logger)
    logger.info(f"Total training samples: {len(all_trace_data)}")

    # Create model
    model = create_model(config)
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
    log_every = config['training'].get('log_every', 10)
    save_every = config['training'].get('save_every', 10)

    logger.info(f"Starting training for {num_epochs} epochs...")

    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        # Train epoch
        avg_loss = train_epoch(model, all_trace_data, optimizer, config, device, logger)

        # Log progress
        if epoch % log_every == 0 or epoch == 1:
            logger.info(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}")

        # Track best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            best_checkpoint_path = checkpoints_dir / 'best_model.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'config': config,
            }, best_checkpoint_path)

        # Save checkpoint periodically
        if epoch % save_every == 0:
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
    logger.info(f"Training completed. Final loss: {avg_loss:.6f}, Best loss: {best_loss:.6f}")

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
