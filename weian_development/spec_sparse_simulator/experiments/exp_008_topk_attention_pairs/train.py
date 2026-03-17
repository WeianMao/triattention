"""
Training logic for Module 2 Multi-Bin Sparse Attention.

Implements training loop with Attraction Loss:
    loss = -log(sum_bins(p_q[b] * P[argmax_key, b])).mean()

This loss encourages the Query's selected bin to contain its argmax Key.
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
    trace_path = exp_dir / config['data']['trace_path']

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


def extract_query_to_key_labels(attention_trace, round_start, round_end, seq_len, exclude_tail=1000, topk_gt=1):
    """
    Extract Top-K key indices for each query in the round.

    For Module 2, we need to know which Keys each Query should attend to (Top-K).

    Args:
        attention_trace: (seq_len, seq_len) attention weights
        round_start: Current round start position
        round_end: Current round end position
        seq_len: Total sequence length
        exclude_tail: Number of tail queries to exclude (default: 1000)
        topk_gt: Number of top attention keys as Ground Truth (default: 1 = Top-1)

    Returns:
        dict with:
            - query_indices: (num_valid_queries,) query position indices
            - topk_keys: (num_valid_queries, topk_gt) top-K key indices for each query
            - topk_in_recent: (num_valid_queries, topk_gt) bool, True if key is in current round
    """
    # Determine valid query range (exclude tail)
    valid_end = min(seq_len - exclude_tail, seq_len)

    query_indices = []
    topk_keys_list = []
    topk_in_recent_list = []

    # Iterate over queries in current round [round_start, round_end)
    # but only include queries where we have historical keys (round_start > 0)
    for q_idx in range(round_start, min(round_end, valid_end)):
        # Get attention weights for all keys <= q_idx (causal)
        attn_weights = attention_trace[q_idx, :q_idx + 1]

        # Find Top-K keys (handle case where available keys < topk_gt)
        actual_k = min(topk_gt, len(attn_weights))
        topk_values, topk_indices = torch.topk(attn_weights, k=actual_k)

        # Pad if needed (when available keys < topk_gt)
        if actual_k < topk_gt:
            # Pad with the last valid index (or -1 as placeholder)
            padding = torch.full((topk_gt - actual_k,), topk_indices[-1].item(), dtype=torch.long)
            topk_indices = torch.cat([topk_indices, padding])

        query_indices.append(q_idx)
        topk_keys_list.append(topk_indices)

        # Check if each key is in current round (recent keys: >= round_start)
        in_recent = topk_indices >= round_start
        topk_in_recent_list.append(in_recent)

    if len(query_indices) == 0:
        return None

    return {
        'query_indices': torch.tensor(query_indices, dtype=torch.long),
        'topk_keys': torch.stack(topk_keys_list),  # (num_valid_queries, topk_gt)
        'topk_in_recent': torch.stack(topk_in_recent_list)  # (num_valid_queries, topk_gt)
    }


def compute_attraction_loss(key_probs, query_bin_probs, topk_keys, topk_in_recent, eps=1e-8):
    """
    Compute Attraction Loss for Module 2 with Top-K support.

    Loss = -(1/valid_pairs) * sum_i sum_k log(sum_b p_q[i,b] * P_k[k,b])

    This encourages the Query's bin selection to overlap with its Top-K Keys' bin assignments.

    Args:
        key_probs: (num_keys, num_bins) Key bin probabilities (softmax over keys, dim=0)
        query_bin_probs: (num_queries, num_bins) Query bin probabilities (softmax over bins, dim=-1)
        topk_keys: (num_queries, topk_gt) Top-K key indices for each query
        topk_in_recent: (num_queries, topk_gt) bool, True if key is in current round (exclude from loss)
        eps: Small value for numerical stability

    Returns:
        Scalar loss tensor
    """
    num_queries, topk_gt = topk_keys.shape
    num_historical_keys = key_probs.shape[0]

    total_loss = 0.0
    valid_pairs = 0

    for i in range(num_queries):
        query_prob = query_bin_probs[i]  # (num_bins,)

        for k in range(topk_gt):
            # Skip if key is in recent round (doesn't need routing)
            if topk_in_recent[i, k]:
                continue

            key_idx = topk_keys[i, k].item()

            # Skip if key index is out of bounds (shouldn't happen, but safety check)
            if key_idx >= num_historical_keys:
                continue

            key_prob = key_probs[key_idx]  # (num_bins,)

            # Compute matching probability: sum over bins of (query_prob * key_prob)
            match_prob = (query_prob * key_prob).sum()

            # Accumulate loss: -log(match_prob)
            total_loss = total_loss + (-torch.log(match_prob + eps))
            valid_pairs += 1

    if valid_pairs == 0:
        # No valid pairs (all keys in recent round)
        return torch.tensor(0.0, device=key_probs.device, requires_grad=True)

    # Average over all valid pairs
    loss = total_loss / valid_pairs

    return loss


def compute_masked_ranking_loss(key_probs, query_bin_probs, topk_keys, topk_in_recent, eps=1e-6):
    """
    Compute Masked Ranking Loss for Module 2.

    This loss constructs K independent training scenarios where each scenario's
    target is "hit top-1 among remaining keys". For the m-th key, we mask out
    the top-(m-1) keys and train to hit the "new top-1" (which is the original top-m).

    Formula:
        For each query i and its Top-K keys [k_1, k_2, ..., k_K]:
        - For k_m (m-th key): mask out {k_1, ..., k_{m-1}}
        - Normalized prob: P_tilde[k_m, b] = P[k_m, b] / R_m(b)
          where R_m(b) = TotalProb(b) - sum_{j<m} P[k_j, b]
        - Match prob: p_match = sum_b p_q[b] * P_tilde[k_m, b]
        - Loss = -log(p_match)

    Vectorized implementation using cumsum for efficiency.

    Args:
        key_probs: (num_keys, num_bins) Key bin probabilities (softmax over keys, dim=0)
        query_bin_probs: (num_queries, num_bins) Query bin probabilities (softmax over bins, dim=-1)
        topk_keys: (num_queries, topk_gt) Top-K key indices for each query
        topk_in_recent: (num_queries, topk_gt) bool, True if key is in current round (exclude from loss)
        eps: Small value for numerical stability

    Returns:
        Scalar loss tensor
    """
    num_queries, topk_gt = topk_keys.shape
    num_bins = key_probs.shape[1]
    num_historical_keys = key_probs.shape[0]
    device = key_probs.device

    # Step 1: Precompute total key probability per bin
    # total_key_prob[b] = sum_k P[k, b]
    total_key_prob = key_probs.sum(dim=0)  # (num_bins,)

    # Step 2: Gather Top-K keys' probabilities
    # Clamp indices to valid range to avoid index errors
    topk_keys_clamped = topk_keys.clamp(0, num_historical_keys - 1)
    # topk_key_probs[i, m, b] = key_probs[topk_keys[i, m], b]
    topk_key_probs = key_probs[topk_keys_clamped]  # (num_queries, topk_gt, num_bins)

    # Step 3: Compute cumulative sum for masked probabilities
    # cumsum[i, m, b] = sum_{j=0}^{m} topk_key_probs[i, j, b]
    cumsum = torch.cumsum(topk_key_probs, dim=1)  # (num_queries, topk_gt, num_bins)

    # Step 4: Construct masked_sum (probability sum of keys to be masked)
    # For k_m, we mask {k_1, ..., k_{m-1}}, so masked_sum[m] = cumsum[m-1]
    # Use shift: masked_sum[:, m, :] = cumsum[:, m-1, :] for m > 0, else 0
    masked_sum = torch.zeros_like(cumsum)
    masked_sum[:, 1:, :] = cumsum[:, :-1, :]  # (num_queries, topk_gt, num_bins)

    # Step 5: Compute remaining probability per bin after masking
    # R_m(b) = TotalProb(b) - masked_sum[m](b)
    remaining_prob = total_key_prob.unsqueeze(0).unsqueeze(0) - masked_sum
    # remaining_prob: (num_queries, topk_gt, num_bins)

    # Numerical stability: clamp remaining_prob to avoid division by very small numbers
    # This is critical when top-K keys dominate the probability mass
    remaining_prob = remaining_prob.clamp(min=eps)

    # Step 6: Normalize key probabilities
    # P_tilde[k_m, b] = P[k_m, b] / R_m(b)
    normalized_key_prob = topk_key_probs / remaining_prob
    # Clamp to prevent extremely large values
    normalized_key_prob = normalized_key_prob.clamp(max=1.0)
    # normalized_key_prob: (num_queries, topk_gt, num_bins)

    # Step 7: Compute match probability
    # match_prob[i, m] = sum_b query_bin_probs[i, b] * normalized_key_prob[i, m, b]
    # query_bin_probs: (num_queries, num_bins) -> (num_queries, 1, num_bins)
    match_prob = (query_bin_probs.unsqueeze(1) * normalized_key_prob).sum(dim=2)
    # match_prob: (num_queries, topk_gt)

    # Step 8: Apply mask for recent keys and out-of-bound keys
    # Create valid mask: not in recent AND within historical key range
    valid_mask = ~topk_in_recent & (topk_keys < num_historical_keys)
    # valid_mask: (num_queries, topk_gt)

    # Compute loss only for valid pairs
    if valid_mask.sum() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Apply mask and compute average loss
    log_match_prob = torch.log(match_prob + eps)
    masked_log_prob = log_match_prob * valid_mask.float()
    loss = -masked_log_prob.sum() / valid_mask.sum().float()

    return loss


def compute_probe_activation_loss(key_logits, key_probs, query_bin_probs, batch_argmax_keys, num_bins, alpha_dead_threshold=0.05, use_weighted=True):
    """
    Compute Weighted Probe Activation Loss to prevent dead probes.

    This loss encourages dead probes (probes with usage below threshold) to learn
    the argmax keys of the current batch, with priority given to "important" keys.

    Weighted Formula: L_activation = -(1/|D|) * sum_d sum_k w_k * log(sigma_d,k)
    where:
        - D = dead probes
        - K+ = unique argmax keys (in historical range)
        - w_k = normalized importance weight (max-pooled probability, detached)
        - sigma = softmax(key_logits, dim=0)

    Weight Computation:
        1. Max pooling over probes: m_k = max_p(sigma_k,p) for each key k
        2. Normalize on K+: w_k = m_k / sum_{k' in K+}(m_k')
        3. Detach weights to prevent gradient flow

    Args:
        key_logits: (num_keys, num_bins) Raw key logits BEFORE softmax
        key_probs: (num_keys, num_bins) Key probabilities (softmax over keys, dim=0)
        query_bin_probs: (batch_size, num_bins) Query bin probabilities (softmax over bins)
        batch_argmax_keys: (batch_size,) Argmax key indices for each query
        num_bins: Number of bins/probes
        alpha_dead_threshold: Minimum allowed usage rate (default 0.05 = 5% of uniform)
        use_weighted: If True, use max-pooled weights; if False, use uniform weights

    Returns:
        Scalar loss tensor with requires_grad=True
    """
    device = key_logits.device

    # Step 1: Dead probe detection
    # Threshold: tau = alpha / N (below this, probe is considered dead)
    tau = alpha_dead_threshold / num_bins

    # Batch average query probabilities over probes
    # query_bin_probs: (batch_size, num_bins) -> p_bar: (num_bins,)
    p_bar = query_bin_probs.mean(dim=0)

    # Dead probe mask: probes with average probability below threshold
    dead_mask = (p_bar < tau)  # (num_bins,) boolean

    # Early return if no dead probes
    num_dead = dead_mask.sum().item()
    if num_dead == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Step 2: Extract positive key set K+ and filter to historical keys
    # IMPORTANT: Only use keys that are within historical keys (indices < num_historical_keys)
    # batch_argmax_keys may contain indices for recent keys which are not in key_logits
    num_historical_keys = key_logits.size(0)
    historical_mask = batch_argmax_keys < num_historical_keys
    historical_argmax_keys = batch_argmax_keys[historical_mask]

    if historical_argmax_keys.numel() == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    positive_keys = torch.unique(historical_argmax_keys)  # (|K+|,)
    num_positive_keys = positive_keys.size(0)

    if num_positive_keys == 0:
        return torch.tensor(0.0, device=device, requires_grad=True)

    # Step 3: Compute weights for positive keys
    if use_weighted:
        # Max pooling over probes: importance score per key
        # key_probs: (num_keys, num_bins) -> max_probs: (num_keys,)
        max_probs, _ = key_probs.max(dim=1)

        # Extract max probs for positive keys only
        positive_max_probs = max_probs[positive_keys]  # (|K+|,)

        # Normalize to get weights (sum to 1)
        weights = positive_max_probs / (positive_max_probs.sum() + 1e-8)  # (|K+|,)

        # CRITICAL: Detach weights to prevent gradient flow through weights
        weights = weights.detach()
    else:
        # Uniform weights (original behavior)
        weights = torch.ones(num_positive_keys, device=device) / num_positive_keys

    # Step 4: Compute log softmax over keys (dim=0) for numerical stability
    # key_logits: (num_keys, num_bins) -> log_key_probs: (num_keys, num_bins)
    log_key_probs = F.log_softmax(key_logits, dim=0)

    # Select dead probe columns
    # log_key_probs: (num_keys, num_bins) -> log_key_probs_dead: (num_keys, |D|)
    log_key_probs_dead = log_key_probs[:, dead_mask]

    # Index positive keys
    # log_key_probs_dead: (num_keys, |D|) -> log_probs_selected: (|K+|, |D|)
    log_probs_selected = log_key_probs_dead[positive_keys, :]

    # Step 5: Compute weighted cross-entropy loss
    # weights: (|K+|,) -> (|K+|, 1) for broadcasting
    # weighted sum over K+: (|K+|, |D|) * (|K+|, 1) -> sum -> (|D|,)
    weighted_log_probs = (weights.unsqueeze(1) * log_probs_selected).sum(dim=0)  # (|D|,)

    # Average over dead probes
    mean_over_dead = weighted_log_probs.mean()  # scalar

    # Negate for loss (maximize log prob = minimize negative log prob)
    activation_loss = -mean_over_dead

    return activation_loss


def compute_load_balancing_loss(query_bin_probs, num_bins):
    """
    Compute Load Balancing Loss using Batch-Level Entropy Regularization.

    This loss encourages uniform probe utilization by maximizing the entropy
    of the batch-average probability distribution (equivalent to minimizing
    negative entropy).

    Formula: L_balance = -H(p_bar) = sum_i (p_bar_i * log(p_bar_i))
    where p_bar = (1/B) * sum_j(P[j,:]) is the batch-average probability

    Properties:
        - When p_bar is uniform: H(p_bar) = log(N), L_balance = -log(N) (minimum)
        - Minimizing L_balance encourages uniform probe usage

    Args:
        query_bin_probs: (batch_size, num_bins) Query bin probabilities
        num_bins: Number of bins/probes

    Returns:
        Scalar loss tensor representing negative entropy
    """
    # Compute batch-average probability distribution
    # p_bar: (num_bins,) - average probability mass on each probe
    p_bar = query_bin_probs.mean(dim=0)

    # Compute negative entropy: sum_i (p_bar_i * log(p_bar_i))
    # Use torch.xlogy for numerical stability (handles p=0 correctly, returns 0)
    negative_entropy = torch.xlogy(p_bar, p_bar).sum()

    return negative_entropy


def train_epoch(model, trace_data, optimizer, config, device, logger):
    """
    Train for one epoch over all rounds in trace.

    Memory-optimized version: keeps data on CPU, only moves batches to device.
    Includes Probe Activation Loss for anti-collapse training.

    Args:
        model: Module2Network instance
        trace_data: Dict with Q, K, attention, seq_len
        optimizer: Optimizer
        config: Configuration dict
        device: torch.device
        logger: Logger instance

    Returns:
        dict: {avg_loss, avg_attraction_loss, avg_activation_loss, avg_balance_loss}
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

    # Top-K Ground Truth selection
    topk_gt = config['training'].get('topk_gt', 1)

    # Loss type selection: 'attraction' (exp_008) or 'mrl' (Masked Ranking Loss, exp_010)
    loss_type = config['training'].get('loss_type', 'attraction')

    # Anti-collapse loss hyperparameters
    num_bins = config['model']['num_bins']
    alpha_dead_threshold = config['training'].get('alpha_dead_threshold', 0.05)
    lambda_activation = config['training'].get('lambda_activation', 0.0)
    lambda_balance = config['training'].get('lambda_balance', 0.0)

    epoch_loss = 0.0
    epoch_attraction_loss = 0.0
    epoch_activation_loss = 0.0
    epoch_balance_loss = 0.0
    num_batches = 0

    # Iterate over rounds
    for round_start in range(0, seq_len, round_window):
        round_end = min(round_start + round_window, seq_len)

        # Skip first round (no historical keys)
        if round_start == 0:
            continue

        # Extract query-to-key labels for this round (Top-K)
        labels = extract_query_to_key_labels(
            attention, round_start, round_end, seq_len, exclude_tail, topk_gt
        )

        if labels is None:
            continue

        query_indices = labels['query_indices']
        topk_keys = labels['topk_keys']
        topk_in_recent = labels['topk_in_recent']

        # Get historical keys (< round_start) - move to device
        historical_keys = K[:round_start].to(device)

        # Compute reference angles for this round
        reference_angles = model.compute_reference_angles(round_start, round_window)

        # Forward pass: Key network on historical keys (get both logits and probs)
        key_logits, key_probs = model.forward_keys_with_logits(historical_keys, reference_angles)
        empty_bin_mask = (key_probs.sum(dim=0) == 0).detach()  # (num_bins,) - detach mask only

        # Clear GPU memory from key forward pass
        del historical_keys
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        # Process queries in batches, accumulate loss for one backward per round
        num_queries = len(query_indices)
        round_attraction_loss = 0.0
        round_activation_loss = 0.0
        round_balance_loss = 0.0
        valid_batches = 0

        for batch_start in range(0, num_queries, query_batch_size):
            batch_end = min(batch_start + query_batch_size, num_queries)

            batch_query_indices = query_indices[batch_start:batch_end]
            batch_topk_keys = topk_keys[batch_start:batch_end].to(device)
            batch_topk_in_recent = topk_in_recent[batch_start:batch_end].to(device)

            # Get queries for this batch
            batch_queries = Q[batch_query_indices].to(device)

            # Forward pass: Query network
            query_bin_probs = model.forward_queries(batch_queries, reference_angles, empty_bin_mask)

            # Compute primary loss based on loss_type
            if loss_type == 'mrl':
                # Masked Ranking Loss (exp_010)
                attraction_loss = compute_masked_ranking_loss(key_probs, query_bin_probs, batch_topk_keys, batch_topk_in_recent)
            else:
                # Attraction Loss (exp_008 original, default)
                attraction_loss = compute_attraction_loss(key_probs, query_bin_probs, batch_topk_keys, batch_topk_in_recent)

            # Compute Probe Activation Loss (if enabled)
            # Note: Still uses Top-1 (argmax) for probe activation loss
            if lambda_activation > 0:
                batch_argmax_keys = batch_topk_keys[:, 0]  # Use Top-1 for probe activation
                activation_loss = compute_probe_activation_loss(
                    key_logits, key_probs, query_bin_probs, batch_argmax_keys,
                    num_bins, alpha_dead_threshold, use_weighted=True
                )
            else:
                activation_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Compute Load Balancing Loss (if enabled)
            if lambda_balance > 0:
                balance_loss = compute_load_balancing_loss(query_bin_probs, num_bins)
            else:
                balance_loss = torch.tensor(0.0, device=device, requires_grad=True)

            # Skip if attraction loss is zero (no valid queries)
            if attraction_loss.item() == 0.0:
                del batch_queries, query_bin_probs
                continue

            # Accumulate losses
            round_attraction_loss = round_attraction_loss + attraction_loss
            round_activation_loss = round_activation_loss + activation_loss
            round_balance_loss = round_balance_loss + balance_loss
            valid_batches += 1

            # Clear batch tensors
            del batch_queries, query_bin_probs

        # Backward pass once per round (if we have valid losses)
        if valid_batches > 0:
            # Compute total loss with lambda weighting
            avg_attraction = round_attraction_loss / valid_batches
            avg_activation = round_activation_loss / valid_batches
            avg_balance = round_balance_loss / valid_batches

            total_loss = avg_attraction + lambda_activation * avg_activation + lambda_balance * avg_balance

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            epoch_loss += total_loss.item()
            epoch_attraction_loss += avg_attraction.item()
            epoch_activation_loss += avg_activation.item()
            epoch_balance_loss += avg_balance.item()
            num_batches += 1

        # Clear key logits and probs after processing all query batches
        del key_logits, key_probs
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # Return average losses
    if num_batches > 0:
        return {
            'avg_loss': epoch_loss / num_batches,
            'avg_attraction_loss': epoch_attraction_loss / num_batches,
            'avg_activation_loss': epoch_activation_loss / num_batches,
            'avg_balance_loss': epoch_balance_loss / num_batches
        }
    else:
        return {
            'avg_loss': 0.0,
            'avg_attraction_loss': 0.0,
            'avg_activation_loss': 0.0,
            'avg_balance_loss': 0.0
        }


def train(config, logger):
    """
    Main training loop.

    Args:
        config: Configuration dict
        logger: Logger instance

    Returns:
        Path to final checkpoint
    """
    logger.info("Initializing Module 2 training...")

    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Set random seed
    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    # Load trace data
    trace_data = load_trace_data(config, logger)

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

    # Log loss type and anti-collapse settings
    loss_type = config['training'].get('loss_type', 'attraction')
    topk_gt = config['training'].get('topk_gt', 1)
    lambda_activation = config['training'].get('lambda_activation', 0.0)
    lambda_balance = config['training'].get('lambda_balance', 0.0)
    alpha_dead_threshold = config['training'].get('alpha_dead_threshold', 0.05)
    logger.info(f"Loss type: {loss_type}, topk_gt: {topk_gt}")
    logger.info(f"Anti-collapse settings: lambda_activation={lambda_activation}, lambda_balance={lambda_balance}, alpha={alpha_dead_threshold}")

    logger.info(f"Starting training for {num_epochs} epochs...")

    best_loss = float('inf')
    for epoch in range(1, num_epochs + 1):
        # Train epoch - returns dict with loss components
        loss_dict = train_epoch(model, trace_data, optimizer, config, device, logger)
        avg_loss = loss_dict['avg_loss']
        avg_attraction_loss = loss_dict['avg_attraction_loss']
        avg_activation_loss = loss_dict['avg_activation_loss']
        avg_balance_loss = loss_dict['avg_balance_loss']

        # Log progress
        if epoch % log_every == 0 or epoch == 1:
            if lambda_activation > 0 or lambda_balance > 0:
                logger.info(f"Epoch {epoch}/{num_epochs} - Total: {avg_loss:.6f}, Attraction: {avg_attraction_loss:.6f}, Activation: {avg_activation_loss:.6f}, Balance: {avg_balance_loss:.6f}")
            else:
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
                'attraction_loss': avg_attraction_loss,
                'activation_loss': avg_activation_loss,
                'balance_loss': avg_balance_loss,
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
                'attraction_loss': avg_attraction_loss,
                'activation_loss': avg_activation_loss,
                'balance_loss': avg_balance_loss,
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
        'attraction_loss': avg_attraction_loss,
        'activation_loss': avg_activation_loss,
        'balance_loss': avg_balance_loss,
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
