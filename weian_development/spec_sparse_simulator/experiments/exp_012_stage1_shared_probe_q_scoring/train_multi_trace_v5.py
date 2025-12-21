"""
Multi-Trace Training for Module 2 (V5) - Batched Forward Version.

Optimizations over V3:
1. Batched Forward: Use forward_keys_batched/forward_queries_batched
   - Process all rounds in a batch with a single forward pass
   - Uses einsum for parallel computation across different ref_positions
2. All V3 optimizations retained

Key changes from V3:
- Inner round loop replaced with batched forward operations
- Significant speedup for multi-round batches

Algorithm logic is UNCHANGED - only implementation optimized.
"""

import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml

from model_v5 import create_model
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

    log_file = log_dir / 'train_multi_trace_v5.log'

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


# Global cache for causal masks to avoid repeated creation
_causal_mask_cache = {}


def get_causal_mask(seq_len, device='cpu'):
    """Get or create cached causal mask."""
    key = (seq_len, device)
    if key not in _causal_mask_cache:
        _causal_mask_cache[key] = torch.triu(
            torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1
        )
    return _causal_mask_cache[key]


def compute_attention_matrix(Q, K, use_cache=True):
    """Compute causal attention matrix with optional mask caching."""
    seq_len, head_dim = Q.shape
    scale = head_dim ** -0.5
    device = Q.device

    attention_logits = Q @ K.T * scale

    if use_cache:
        causal_mask = get_causal_mask(seq_len, device)
    else:
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device), diagonal=1)

    attention_logits.masked_fill_(causal_mask, float('-inf'))
    attention = F.softmax(attention_logits, dim=-1)

    return attention


def extract_round_labels(attention_trace, round_start, round_end, seq_len, exclude_tail=1000):
    """
    Extract argmax key indices for each query in the round (vectorized).

    Returns:
        query_indices: Tensor of query indices
        argmax_keys: Tensor of argmax key indices
        argmax_in_recent: Tensor of bools indicating if argmax is in recent keys
    """
    valid_end = min(seq_len - exclude_tail, seq_len)
    actual_end = min(round_end, valid_end)

    if actual_end <= round_start:
        return None

    # Vectorized: extract all queries at once
    # query_indices: [round_start, round_start+1, ..., actual_end-1]
    query_indices = torch.arange(round_start, actual_end, dtype=torch.long)
    num_queries = len(query_indices)

    if num_queries == 0:
        return None

    # For causal attention, query q_idx can only attend to keys [0, q_idx]
    # We need argmax of attention_trace[q_idx, :q_idx+1] for each q_idx
    # This is tricky to vectorize perfectly due to variable-length rows

    # Approach: Use the full attention slice and mask out future positions
    # attention_trace[round_start:actual_end, :] has shape (num_queries, seq_len)
    attn_slice = attention_trace[round_start:actual_end, :]  # (num_queries, seq_len)

    # Create a mask for valid positions: position j is valid for query i if j <= round_start + i
    # query i (0-indexed in slice) corresponds to q_idx = round_start + i
    # valid positions: j <= round_start + i, i.e., j < round_start + i + 1
    row_indices = torch.arange(num_queries, device=attn_slice.device).unsqueeze(1)  # (num_queries, 1)
    col_indices = torch.arange(seq_len, device=attn_slice.device).unsqueeze(0)  # (1, seq_len)
    valid_mask = col_indices <= (round_start + row_indices)  # (num_queries, seq_len)

    # Mask out invalid positions with -inf before argmax
    masked_attn = attn_slice.clone()
    masked_attn[~valid_mask] = float('-inf')

    # Vectorized argmax
    argmax_keys = masked_attn.argmax(dim=1)  # (num_queries,)

    # Check if argmax is in recent keys (>= round_start)
    argmax_in_recent = argmax_keys >= round_start

    return {
        'query_indices': query_indices,
        'argmax_keys': argmax_keys.cpu() if argmax_keys.is_cuda else argmax_keys,
        'argmax_in_recent': argmax_in_recent.cpu() if argmax_in_recent.is_cuda else argmax_in_recent
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
    Vectorized: avoids Python loop over parameters.
    """
    # Collect all parameter differences in a list, then sum
    diffs = [(param - init_params[name]).pow(2).sum()
             for name, param in model.named_parameters() if name in init_params]
    if len(diffs) == 0:
        return torch.tensor(0.0, device=next(model.parameters()).device)
    return torch.stack(diffs).sum()


def compute_gradient_norm(model):
    """
    Compute the L2 norm of all gradients.
    Vectorized: avoids multiple .item() calls that cause CPU-GPU sync.
    """
    grads = [p.grad.detach().pow(2).sum() for p in model.parameters() if p.grad is not None]
    if len(grads) == 0:
        return 0.0
    total_norm_sq = torch.stack(grads).sum()
    return total_norm_sq.sqrt().item()


def compute_parameter_norm(model):
    """
    Compute the L2 norm of all parameters.
    Vectorized: avoids multiple .item() calls that cause CPU-GPU sync.
    """
    norms = [p.detach().pow(2).sum() for p in model.parameters()]
    if len(norms) == 0:
        return 0.0
    total_norm_sq = torch.stack(norms).sum()
    return total_norm_sq.sqrt().item()


def compute_parameter_stats(model):
    """Compute parameter statistics per layer."""
    stats = {}
    for name, param in model.named_parameters():
        stats[name] = {
            'mean': param.data.mean().item(),
            'std': param.data.std().item(),
            'norm': param.data.norm(2).item(),
            'max': param.data.abs().max().item(),
        }
        if param.grad is not None:
            stats[name]['grad_norm'] = param.grad.data.norm(2).item()
            stats[name]['grad_max'] = param.grad.data.abs().max().item()
    return stats


def compute_batch_accuracy_fast(key_probs, query_bin_probs, argmax_keys, argmax_in_recent, top_k_bins=1, k_threshold=50):
    """
    Compute bin accuracy for a batch of queries (fully vectorized).

    Args:
        key_probs: (num_keys, num_bins) - key probability distribution
        query_bin_probs: (num_queries, num_bins) - query bin probabilities
        argmax_keys: (num_queries,) - ground truth argmax key indices
        argmax_in_recent: (num_queries,) - whether argmax is in recent keys
        top_k_bins: Number of top bins to consider (1 or 8)
        k_threshold: Consider hit if argmax key is in top-k of selected bin

    Returns:
        hits: Number of correct predictions (including recent hits)
        total: Total number of queries
        recent_hits: Number of queries with argmax in recent keys
        bin_hits: Number of queries where argmax key is in selected bins' top keys
    """
    num_queries = query_bin_probs.shape[0]
    num_keys, num_bins = key_probs.shape
    device = key_probs.device

    # Count recent hits
    recent_hits = argmax_in_recent.sum().item()

    # For non-recent queries, check if argmax key is in top bins' top keys
    non_recent_mask = ~argmax_in_recent
    num_non_recent = non_recent_mask.sum().item()
    bin_hits = 0

    if num_non_recent > 0:
        # Step 1: Pre-compute top-k_threshold keys for ALL bins at once
        # This is the key optimization - only ONE topk call instead of O(queries * bins)
        actual_k = min(k_threshold, num_keys)
        _, topk_keys_per_bin = torch.topk(key_probs, actual_k, dim=0)
        # topk_keys_per_bin: (k_threshold, num_bins) - indices of top keys for each bin

        # Step 2: Get top bins for each non-recent query
        actual_top_bins = min(top_k_bins, num_bins)
        _, top_bin_indices = torch.topk(query_bin_probs[non_recent_mask], actual_top_bins, dim=1)
        # top_bin_indices: (num_non_recent, top_k_bins)

        non_recent_argmax = argmax_keys[non_recent_mask]
        # Clamp to valid range
        non_recent_argmax = non_recent_argmax.clamp(0, num_keys - 1)

        # Step 3: For each query, check if argmax_key is in topk of ANY selected bin
        # Gather the topk keys for the selected bins
        # topk_keys_per_bin: (k_threshold, num_bins)
        # top_bin_indices: (num_non_recent, top_k_bins)
        # We need: topk_keys_selected[q, b, k] = topk_keys_per_bin[k, top_bin_indices[q, b]]

        # Expand for gathering: (k_threshold, num_non_recent, top_k_bins)
        expanded_bin_indices = top_bin_indices.unsqueeze(0).expand(actual_k, -1, -1)
        # topk_keys_per_bin: (k_threshold, num_bins) -> expand for queries
        topk_keys_expanded = topk_keys_per_bin.unsqueeze(1).expand(-1, num_non_recent, -1)
        # Gather: select the bins for each query
        topk_keys_selected = topk_keys_expanded.gather(2, expanded_bin_indices)
        # topk_keys_selected: (k_threshold, num_non_recent, top_k_bins)

        # Step 4: Check if argmax_key matches any of the topk keys in any selected bin
        # non_recent_argmax: (num_non_recent,) -> (1, num_non_recent, 1) for broadcasting
        argmax_expanded = non_recent_argmax.view(1, num_non_recent, 1)
        # Compare: (k_threshold, num_non_recent, top_k_bins)
        matches = (topk_keys_selected == argmax_expanded)
        # For each query, check if ANY match exists across all (k, bin) combinations
        query_hits = matches.any(dim=0).any(dim=1)  # (num_non_recent,)

        bin_hits = query_hits.sum().item()

    total = num_queries
    hits = recent_hits + bin_hits

    return hits, total, recent_hits, bin_hits


def train_on_trace_batched(
    model, trace_data, optimizer, config, device, logger,
    init_params=None, weight_decay=0.0, round_batch_size=64,
    compute_accuracy=False
):
    """
    Train on a single trace with round batching (V5 - batched forward).

    Uses forward_keys_batched/forward_queries_batched to process multiple
    rounds in parallel with a single forward pass.

    Label caching: On first epoch, computes attention matrix to extract labels,
    then caches labels and discards attention to save memory. Subsequent epochs
    use cached labels directly without recomputing attention.

    Args:
        model: Module2Network instance
        trace_data: Dict with Q, K for single head. Labels are cached after first epoch.
        optimizer: Optimizer
        config: Configuration dict
        device: torch.device
        logger: Logger instance
        init_params: Dict of initial parameter values for regularization
        weight_decay: Regularization strength
        round_batch_size: Number of rounds to process together
        compute_accuracy: Whether to compute accuracy metrics (slow, default: False)

    Returns:
        trace_loss: Average loss for this trace
        trace_metrics: Dict with accuracy metrics
    """
    model.train()

    # Data is already on device (preloaded)
    Q = trace_data['Q']
    K = trace_data['K']
    seq_len = trace_data['seq_len']

    round_window = config['training']['round_window']
    exclude_tail = config['training']['exclude_tail']

    # Check if labels are cached (from previous epoch)
    if 'cached_rounds' in trace_data:
        # Use cached labels - no attention computation needed
        rounds = trace_data['cached_rounds']
    else:
        # First epoch: compute attention, extract labels, cache them
        # Compute attention matrix (will be discarded after extracting labels)
        attention = compute_attention_matrix(Q, K, use_cache=True)

        # Collect all valid rounds and their labels
        rounds = []
        for round_start in range(round_window, seq_len, round_window):  # Skip first round
            round_end = min(round_start + round_window, seq_len)
            labels = extract_round_labels(attention, round_start, round_end, seq_len, exclude_tail)
            if labels is not None and len(labels['query_indices']) > 0:
                rounds.append({
                    'round_start': round_start,
                    'round_end': round_end,
                    'labels': labels
                })

        # Cache labels for subsequent epochs (small memory footprint)
        trace_data['cached_rounds'] = rounds

        # Delete attention matrix to free memory (labels are already extracted)
        del attention

    if len(rounds) == 0:
        return 0.0, {'top1_hits': 0, 'top8_hits': 0, 'total': 0, 'recent_hits': 0}

    # Process rounds in batches
    trace_loss = 0.0
    trace_attraction_loss = 0.0
    trace_reg_loss = 0.0
    trace_grad_norm = 0.0
    num_batches = 0
    metrics = {'top1_hits': 0, 'top8_hits': 0, 'total': 0, 'recent_hits': 0}

    for batch_start in range(0, len(rounds), round_batch_size):
        batch_end = min(batch_start + round_batch_size, len(rounds))
        batch_rounds = rounds[batch_start:batch_end]
        actual_batch_size = len(batch_rounds)

        # Collect batch data for batched forward
        round_starts = [r['round_start'] for r in batch_rounds]
        max_round_start = max(round_starts)

        # Prepare ref_positions and key_lengths for batched key forward
        ref_positions = torch.tensor(
            [rs + round_window // 2 for rs in round_starts],
            dtype=torch.float32, device=device
        )
        key_lengths = torch.tensor(round_starts, dtype=torch.long, device=device)

        # Batched key forward: all rounds share K[:max_round_start]
        # Each round uses keys up to its own round_start (handled by key_lengths mask)
        K_shared = K[:max_round_start]
        key_probs_batch, key_mask = model.forward_keys_batched(K_shared, ref_positions, key_lengths)
        # key_probs_batch: (batch_size, max_round_start, num_bins)

        # Compute empty_bin_mask for each round
        # A bin is empty if no keys have non-zero probability (after masking)
        # For each round i, only keys[:round_starts[i]] are valid
        empty_bin_mask_batch = []
        for i, rs in enumerate(round_starts):
            # key_probs_batch[i, :rs, :] are valid
            valid_key_probs = key_probs_batch[i, :rs, :]  # (rs, num_bins)
            bin_sum = valid_key_probs.sum(dim=0)  # (num_bins,)
            empty_mask = (bin_sum == 0).detach()
            empty_bin_mask_batch.append(empty_mask)
        empty_bin_mask_batch = torch.stack(empty_bin_mask_batch, dim=0)  # (batch_size, num_bins)

        # Prepare query batch for batched query forward
        # Each round may have different number of queries, need to pad
        max_num_queries = max(len(r['labels']['query_indices']) for r in batch_rounds)
        Q_batch = torch.zeros(actual_batch_size, max_num_queries, Q.shape[1], device=device)
        query_lengths = []

        for i, round_info in enumerate(batch_rounds):
            query_indices = round_info['labels']['query_indices']
            num_queries = len(query_indices)
            Q_batch[i, :num_queries] = Q[query_indices]
            query_lengths.append(num_queries)

        # Batched query forward
        bin_probs_batch = model.forward_queries_batched(Q_batch, ref_positions, empty_bin_mask_batch)
        # bin_probs_batch: (batch_size, max_num_queries, num_bins)

        # Compute loss for each round and accumulate
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)
        valid_rounds = 0

        for i, round_info in enumerate(batch_rounds):
            rs = round_info['round_start']
            labels = round_info['labels']
            num_queries = query_lengths[i]

            argmax_keys = labels['argmax_keys']
            argmax_in_recent = labels['argmax_in_recent']

            # Extract this round's key_probs (only valid keys)
            key_probs_i = key_probs_batch[i, :rs, :]  # (rs, num_bins)

            # Extract this round's query bin probs (only valid queries)
            query_bin_probs_i = bin_probs_batch[i, :num_queries, :]  # (num_queries, num_bins)

            # Labels may be on CPU, move to device
            argmax_keys_dev = argmax_keys.to(device) if argmax_keys.device != device else argmax_keys
            argmax_in_recent_dev = argmax_in_recent.to(device) if argmax_in_recent.device != device else argmax_in_recent

            # Compute loss
            round_loss = compute_attraction_loss(
                key_probs_i, query_bin_probs_i,
                argmax_keys_dev, argmax_in_recent_dev
            )

            if round_loss.item() > 0:
                batch_loss = batch_loss + round_loss
                valid_rounds += 1

            # Compute accuracy metrics only if requested (expensive)
            if compute_accuracy:
                with torch.no_grad():
                    # Top-1 accuracy
                    hits1, total, recent, bin_hits1 = compute_batch_accuracy_fast(
                        key_probs_i, query_bin_probs_i,
                        argmax_keys_dev, argmax_in_recent_dev,
                        top_k_bins=1
                    )
                    metrics['top1_hits'] += hits1
                    metrics['recent_hits'] += recent

                    # Top-8 accuracy
                    hits8, _, _, bin_hits8 = compute_batch_accuracy_fast(
                        key_probs_i, query_bin_probs_i,
                        argmax_keys_dev, argmax_in_recent_dev,
                        top_k_bins=8
                    )
                    metrics['top8_hits'] += hits8
                    metrics['total'] += total

        # Backward pass for the batch
        if valid_rounds > 0:
            optimizer.zero_grad()

            avg_batch_loss = batch_loss / valid_rounds

            if init_params is not None and weight_decay > 0:
                reg_loss = compute_init_regularization_loss(model, init_params)
                total_loss = avg_batch_loss + weight_decay * reg_loss
                trace_reg_loss += reg_loss.item() if torch.is_tensor(reg_loss) else reg_loss
            else:
                total_loss = avg_batch_loss

            total_loss.backward()

            # Compute gradient norm before optimizer step
            grad_norm = compute_gradient_norm(model)
            trace_grad_norm += grad_norm

            optimizer.step()

            trace_loss += total_loss.item()
            trace_attraction_loss += avg_batch_loss.item()
            num_batches += 1

    avg_loss = trace_loss / num_batches if num_batches > 0 else 0.0
    avg_attraction_loss = trace_attraction_loss / num_batches if num_batches > 0 else 0.0
    avg_reg_loss = trace_reg_loss / num_batches if num_batches > 0 else 0.0
    avg_grad_norm = trace_grad_norm / num_batches if num_batches > 0 else 0.0

    metrics['attraction_loss'] = avg_attraction_loss
    metrics['reg_loss'] = avg_reg_loss
    metrics['grad_norm'] = avg_grad_norm

    return avg_loss, metrics


def train_epoch_multi_trace(
    model, all_traces, optimizer, config, device, logger,
    init_params=None, weight_decay=0.0, round_batch_size=64,
    compute_accuracy=False
):
    """
    Train for one epoch over all preloaded traces.

    Returns:
        epoch_loss: Average epoch loss
        epoch_metrics: Dict with aggregated accuracy metrics
    """
    epoch_loss = 0.0
    epoch_attraction_loss = 0.0
    epoch_reg_loss = 0.0
    epoch_grad_norm = 0.0
    num_traces = 0
    epoch_metrics = {'top1_hits': 0, 'top8_hits': 0, 'total': 0, 'recent_hits': 0}
    trace_metrics_list = []

    for trace_data in all_traces:
        trace_loss, trace_metrics = train_on_trace_batched(
            model, trace_data, optimizer, config, device, logger,
            init_params=init_params, weight_decay=weight_decay,
            round_batch_size=round_batch_size,
            compute_accuracy=compute_accuracy
        )

        if trace_loss > 0:
            epoch_loss += trace_loss
            epoch_attraction_loss += trace_metrics.get('attraction_loss', 0.0)
            epoch_reg_loss += trace_metrics.get('reg_loss', 0.0)
            epoch_grad_norm += trace_metrics.get('grad_norm', 0.0)
            num_traces += 1

            # Aggregate metrics if available
            if compute_accuracy:
                for k in epoch_metrics:
                    epoch_metrics[k] += trace_metrics[k]

                # Compute per-trace accuracy
                trace_total = trace_metrics['total']
                if trace_total > 0:
                    trace_top1_acc = trace_metrics['top1_hits'] / trace_total * 100
                    trace_top8_acc = trace_metrics['top8_hits'] / trace_total * 100
                    logger.info(f"  {trace_data['name']}: loss={trace_loss:.6f}, "
                               f"top1={trace_top1_acc:.2f}%, top8={trace_top8_acc:.2f}%")
                    trace_metrics_list.append({
                        'name': trace_data['name'],
                        'loss': trace_loss,
                        'top1_acc': trace_top1_acc,
                        'top8_acc': trace_top8_acc
                    })
            else:
                # Just log loss
                logger.info(f"  {trace_data['name']}: loss={trace_loss:.6f}")

    avg_loss = epoch_loss / num_traces if num_traces > 0 else 0.0

    # Compute overall accuracy
    total = epoch_metrics['total']
    if total > 0:
        epoch_metrics['top1_acc'] = epoch_metrics['top1_hits'] / total * 100
        epoch_metrics['top8_acc'] = epoch_metrics['top8_hits'] / total * 100
        epoch_metrics['recent_rate'] = epoch_metrics['recent_hits'] / total * 100
    else:
        epoch_metrics['top1_acc'] = 0.0
        epoch_metrics['top8_acc'] = 0.0
        epoch_metrics['recent_rate'] = 0.0

    # Add averaged training metrics
    epoch_metrics['attraction_loss'] = epoch_attraction_loss / num_traces if num_traces > 0 else 0.0
    epoch_metrics['reg_loss'] = epoch_reg_loss / num_traces if num_traces > 0 else 0.0
    epoch_metrics['grad_norm'] = epoch_grad_norm / num_traces if num_traces > 0 else 0.0
    epoch_metrics['trace_metrics'] = trace_metrics_list

    return avg_loss, epoch_metrics


def compute_topk_hit_rate_fast(model, test_trace_data, K_values, device, logger, top_bins=1):
    """
    Compute TopK Hit Rate using vectorized operations (fast version).

    Uses preloaded test data with cached labels to avoid recomputing attention.
    Processes all queries in a round together (batched forward).

    Args:
        model: Module2Network instance (in eval mode)
        test_trace_data: Preloaded test trace dict with Q, K, cached_rounds
        K_values: List of K values to evaluate
        device: torch.device
        logger: Logger instance
        top_bins: Number of top bins to use

    Returns:
        Dict with hit rates for each K value
    """
    Q = test_trace_data['Q']
    K = test_trace_data['K']
    cached_rounds = test_trace_data['cached_rounds']
    round_window = test_trace_data['round_window']

    # Initialize counters for each K value
    results = {k: {'hits': 0, 'total': 0, 'recent_hits': 0, 'bin_hits': 0} for k in K_values}

    with torch.no_grad():
        for round_info in cached_rounds:
            round_start = round_info['round_start']
            labels = round_info['labels']

            query_indices = labels['query_indices']
            argmax_keys = labels['argmax_keys'].to(device)
            argmax_in_recent = labels['argmax_in_recent'].to(device)

            num_queries = len(query_indices)
            if num_queries == 0:
                continue

            # Historical keys
            historical_keys = K[:round_start]
            num_historical = round_start

            # Compute reference angles
            reference_angles = model.compute_reference_angles(round_start, round_window)

            # Forward pass: Key network (once per round)
            key_probs = model.forward_keys(historical_keys, reference_angles)

            # Detect empty bins
            empty_bin_mask = (key_probs.sum(dim=0) == 0).detach()

            # Forward pass: Query network (batched for all queries in this round)
            queries = Q[query_indices]
            query_bin_probs = model.forward_queries(queries, reference_angles, empty_bin_mask)

            # Pre-compute topk keys for all bins at once (key optimization!)
            max_k = max(K_values)
            actual_k = min(max_k, num_historical)
            _, topk_keys_per_bin = torch.topk(key_probs, actual_k, dim=0)
            # topk_keys_per_bin: (actual_k, num_bins)

            # Get top bins for each query
            actual_top_bins = min(top_bins, model.num_bins)
            _, top_bin_indices = torch.topk(query_bin_probs, actual_top_bins, dim=1)
            # top_bin_indices: (num_queries, top_bins)

            # Count recent hits (vectorized)
            recent_mask = argmax_in_recent
            num_recent = recent_mask.sum().item()

            # For non-recent queries, check if argmax is in selected bins' topk
            non_recent_mask = ~argmax_in_recent
            num_non_recent = non_recent_mask.sum().item()

            # For each K value
            for k_val in K_values:
                results[k_val]['total'] += num_queries
                results[k_val]['recent_hits'] += num_recent
                results[k_val]['hits'] += num_recent

                if num_non_recent > 0:
                    # Get topk keys for this K value
                    k_actual = min(k_val, actual_k)
                    topk_keys_k = topk_keys_per_bin[:k_actual, :]  # (k_actual, num_bins)

                    # For each non-recent query, check if argmax is in any selected bin's topk
                    non_recent_indices = torch.where(non_recent_mask)[0]
                    non_recent_argmax = argmax_keys[non_recent_mask]
                    non_recent_top_bins = top_bin_indices[non_recent_mask]  # (num_non_recent, top_bins)

                    # Gather topk keys for selected bins
                    # topk_keys_k: (k_actual, num_bins)
                    # non_recent_top_bins: (num_non_recent, top_bins)
                    # We need: for each query, gather keys from its selected bins

                    # Expand for broadcasting
                    expanded_bins = non_recent_top_bins.unsqueeze(0).expand(k_actual, -1, -1)
                    # expanded_bins: (k_actual, num_non_recent, top_bins)

                    topk_expanded = topk_keys_k.unsqueeze(1).expand(-1, num_non_recent, -1)
                    # topk_expanded: (k_actual, num_non_recent, num_bins)

                    selected_keys = topk_expanded.gather(2, expanded_bins)
                    # selected_keys: (k_actual, num_non_recent, top_bins)

                    # Check if argmax matches any selected key
                    argmax_expanded = non_recent_argmax.view(1, num_non_recent, 1)
                    matches = (selected_keys == argmax_expanded)
                    # matches: (k_actual, num_non_recent, top_bins)

                    query_hits = matches.any(dim=0).any(dim=1)  # (num_non_recent,)
                    bin_hits = query_hits.sum().item()

                    results[k_val]['bin_hits'] += bin_hits
                    results[k_val]['hits'] += bin_hits

    # Compute hit rates
    hit_rates = {}
    for k_val in K_values:
        total = results[k_val]['total']
        if total > 0:
            hit_rate = results[k_val]['hits'] / total * 100
            recent_rate = results[k_val]['recent_hits'] / total * 100
            bin_rate = results[k_val]['bin_hits'] / total * 100
        else:
            hit_rate = recent_rate = bin_rate = 0.0

        keys_per_query = top_bins * k_val
        hit_rates[k_val] = {
            'hit_rate': hit_rate,
            'recent_hit_rate': recent_rate,
            'bin_hit_rate': bin_rate,
            'total_queries': total,
            'total_hits': results[k_val]['hits'],
            'recent_hits': results[k_val]['recent_hits'],
            'bin_hits': results[k_val]['bin_hits'],
            'top_bins': top_bins,
            'keys_per_query': keys_per_query
        }

        if top_bins == 1:
            logger.info(
                f"K={k_val}: Hit Rate = {hit_rate:.2f}% "
                f"(Recent: {recent_rate:.2f}%, Bin: {bin_rate:.2f}%, "
                f"Total: {results[k_val]['hits']}/{total})"
            )
        else:
            logger.info(
                f"K={k_val} (top-{top_bins} bins, {keys_per_query} keys/query): Hit Rate = {hit_rate:.2f}% "
                f"(Recent: {recent_rate:.2f}%, Bin: {bin_rate:.2f}%, "
                f"Total: {results[k_val]['hits']}/{total})"
            )

    return hit_rates


def evaluate_on_test(model, config, device, logger, test_trace_data=None):
    """
    Evaluate model on test set.

    Args:
        model: Model to evaluate
        config: Configuration dict
        device: torch.device
        logger: Logger instance
        test_trace_data: Preloaded test trace data (optional, will load if not provided)

    Returns:
        hit_rates: Dict with hit rates for different K values
    """
    eval_start = time.time()
    model.eval()

    # Use preloaded data or load from disk (fallback for standalone evaluation)
    if test_trace_data is None:
        from evaluate import load_trace_data, compute_topk_hit_rate
        trace_data = load_trace_data(config, logger, trace_type='test')
        K_values = config['evaluation']['topk_K']
        round_window = config['evaluation'].get('round_window', config['training']['round_window'])
        exclude_tail = config['training']['exclude_tail']

        # Use original slow path
        hit_rates_top1 = compute_topk_hit_rate(
            model, trace_data, K_values, round_window, exclude_tail, device, logger, top_bins=1
        )
        hit_rates_top8 = compute_topk_hit_rate(
            model, trace_data, K_values, round_window, exclude_tail, device, logger, top_bins=8
        )
    else:
        # Use fast vectorized evaluation
        K_values = config['evaluation']['topk_K']

        hit_rates_top1 = compute_topk_hit_rate_fast(
            model, test_trace_data, K_values, device, logger, top_bins=1
        )
        hit_rates_top8 = compute_topk_hit_rate_fast(
            model, test_trace_data, K_values, device, logger, top_bins=8
        )

    model.train()

    eval_time = time.time() - eval_start
    logger.info(f"Evaluation completed in {eval_time:.1f}s")

    return {'top1': hit_rates_top1, 'top8': hit_rates_top8}


def preload_all_traces(trace_paths, layer, head, logger, device='cpu', precompute_attention=True):
    """
    Preload all traces at startup, keeping only the required head.

    Args:
        trace_paths: List of paths to trace files
        layer: Layer index
        head: Head index
        logger: Logger instance
        device: Device to load data to ('cpu' or 'cuda')
        precompute_attention: If True, precompute attention matrix (saves time during training)

    Returns:
        List of trace_data dicts with Q, K, attention (optional), and metadata
    """
    logger.info(f"Preloading {len(trace_paths)} traces to {device} (head [{layer}, {head}])...")

    all_traces = []
    total_queries = 0

    for i, trace_path in enumerate(trace_paths):
        trace_name = trace_path.parent.name
        logger.info(f"  [{i+1}/{len(trace_paths)}] Loading {trace_name}...")

        # Check for cached single-head file
        cache_dir = trace_path.parent / 'head_cache'
        cache_file = cache_dir / f'layer{layer}_head{head}.pt'

        if cache_file.exists():
            # Load from cache (fast path)
            cached_data = torch.load(cache_file, map_location='cpu')
            Q = cached_data['q'].to(device)
            K = cached_data['k'].to(device)
            del cached_data
            logger.info(f"       loaded from cache")
        else:
            # Load from full file, then cache
            qk_data = torch.load(trace_path, map_location='cpu')
            Q = qk_data['q'][layer, head].clone().to(device)
            K = qk_data['k'][layer, head].clone().to(device)

            # Save to cache for future runs
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_data = {
                'q': qk_data['q'][layer, head].clone(),
                'k': qk_data['k'][layer, head].clone(),
                'layer': layer,
                'head': head,
            }
            torch.save(cache_data, cache_file)
            del qk_data
            logger.info(f"       cached to {cache_file.name}")

        seq_len = Q.shape[0]
        total_queries += seq_len

        trace_data = {
            'Q': Q,
            'K': K,
            'seq_len': seq_len,
            'head_dim': Q.shape[1],
            'name': trace_name
        }

        # Precompute attention matrix (deterministic, only needs to be done once)
        if precompute_attention:
            with torch.no_grad():
                attention = compute_attention_matrix(Q, K, use_cache=True)
                trace_data['attention'] = attention
            logger.info(f"       seq_len={seq_len}, attention precomputed")
        else:
            logger.info(f"       seq_len={seq_len}")

        all_traces.append(trace_data)

    logger.info(f"Preloading complete. Total sequences: {total_queries}")
    if device != 'cpu' and torch.cuda.is_available():
        gpu_mem = torch.cuda.memory_allocated() / 1024**3
        logger.info(f"GPU memory after preload: {gpu_mem:.2f} GB")

    return all_traces


def preload_test_trace(test_trace_path, layer, head, config, logger, device='cuda'):
    """
    Preload test trace and extract labels for evaluation.

    This avoids reloading the test trace and recomputing attention on every evaluation.

    Args:
        test_trace_path: Path to test trace qk.pt file
        layer: Layer index
        head: Head index
        config: Configuration dict
        logger: Logger instance
        device: Device to load data to

    Returns:
        dict with Q, K, cached_rounds (labels), and metadata
    """
    from pathlib import Path
    test_trace_path = Path(test_trace_path)

    # Check for cached single-head file
    cache_dir = test_trace_path.parent / 'head_cache'
    cache_file = cache_dir / f'layer{layer}_head{head}.pt'

    if cache_file.exists():
        cached_data = torch.load(cache_file, map_location='cpu')
        Q = cached_data['q'].to(device)
        K = cached_data['k'].to(device)
        del cached_data
        logger.info(f"       test trace loaded from cache")
    else:
        qk_data = torch.load(test_trace_path, map_location='cpu')
        Q = qk_data['q'][layer, head].clone().to(device)
        K = qk_data['k'][layer, head].clone().to(device)

        # Save to cache
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_data = {
            'q': qk_data['q'][layer, head].clone(),
            'k': qk_data['k'][layer, head].clone(),
            'layer': layer,
            'head': head,
        }
        torch.save(cache_data, cache_file)
        del qk_data
        logger.info(f"       test trace cached")

    seq_len = Q.shape[0]
    round_window = config['evaluation'].get('round_window', config['training']['round_window'])
    exclude_tail = config['training']['exclude_tail']

    # Compute attention and extract labels (then delete attention)
    logger.info(f"       extracting labels (seq_len={seq_len})...")
    with torch.no_grad():
        attention = compute_attention_matrix(Q, K, use_cache=True)

        # Extract labels for all rounds
        cached_rounds = []
        for round_start in range(round_window, seq_len, round_window):
            round_end = min(round_start + round_window, seq_len)
            labels = extract_round_labels(attention, round_start, round_end, seq_len, exclude_tail)
            if labels is not None and len(labels['query_indices']) > 0:
                cached_rounds.append({
                    'round_start': round_start,
                    'round_end': round_end,
                    'labels': labels
                })

        # Free attention memory
        del attention

    logger.info(f"       {len(cached_rounds)} rounds extracted")

    return {
        'Q': Q,
        'K': K,
        'seq_len': seq_len,
        'head_dim': Q.shape[1],
        'layer': layer,
        'head': head,
        'cached_rounds': cached_rounds,
        'round_window': round_window,
        'exclude_tail': exclude_tail,
    }


def train(config, logger, use_l2_norm=False, invert_to_origin=False, weight_decay=0.0,
          round_batch_size=64, eval_every=2, use_tensorboard=True):
    """
    Main training loop with optimizations.

    Args:
        config: Configuration dict
        logger: Logger instance
        use_l2_norm: If True, use L2 normalization (default: False)
        invert_to_origin: If True, invert to origin (default: False)
        weight_decay: Regularization strength (default: 0.0)
        round_batch_size: Number of rounds to process together (default: 64)
        eval_every: Evaluate on test set every N epochs (default: 2)
        use_tensorboard: Enable TensorBoard logging (default: True)

    Returns:
        Path to final checkpoint
    """
    logger.info("Initializing Multi-Trace Module 2 training (V5 - Batched Forward)...")
    logger.info(f"Settings: use_l2_norm={use_l2_norm}, invert_to_origin={invert_to_origin}")
    logger.info(f"Optimizations: round_batch_size={round_batch_size}, eval_every={eval_every}")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    torch.manual_seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    exp_dir = Path(__file__).parent

    # Setup TensorBoard
    writer = None
    if use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tb_dir = exp_dir / config['output']['logs_dir'] / 'tensorboard'
            tb_dir.mkdir(parents=True, exist_ok=True)
            writer = SummaryWriter(log_dir=str(tb_dir))
            logger.info(f"TensorBoard logging enabled: {tb_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
            writer = None

    # Get head info
    head_sample_path = exp_dir / config['data']['head_sample_file']
    with open(head_sample_path, 'r') as f:
        head_samples = json.load(f)
    layer, head = head_samples[0]
    logger.info(f"Using head [layer={layer}, head={head}]")

    # Get training trace paths
    test_trace_path = exp_dir / config['data']['test_trace_path']
    trace_paths = get_training_trace_paths(test_trace_path, logger)

    # Step 1: Preload all traces to GPU (training + test)
    step1_start = time.time()
    logger.info("=" * 60)
    logger.info("Step 1: Preloading traces to GPU...")

    # 1a. Preload training traces
    all_traces = preload_all_traces(
        trace_paths, layer, head, logger,
        device=device, precompute_attention=False
    )

    # 1b. Preload test trace (for evaluation)
    logger.info("  Preloading test trace...")
    test_trace_data = preload_test_trace(
        test_trace_path, layer, head, config, logger, device
    )

    step1_time = time.time() - step1_start
    logger.info(f"Step 1 completed in {step1_time:.1f}s")
    logger.info("=" * 60)

    # Step 2: Compute K-means initialization
    use_kmeans_init = config.get('model', {}).get('use_kmeans_init', True)
    init_probes = None

    if use_kmeans_init:
        step2_start = time.time()
        logger.info("Step 2: Computing K-means initialization...")
        init_probes = compute_kmeans_init_multi_trace(
            config, logger, n_clusters=config['model']['num_bins'],
            use_l2_norm=use_l2_norm, invert_to_origin=invert_to_origin,
            preloaded_traces=all_traces  # Reuse already loaded data
        )
        step2_time = time.time() - step2_start
        logger.info(f"Step 2 completed in {step2_time:.1f}s. Probe shape: {init_probes.shape}")
        logger.info("=" * 60)

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
    training_start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start_time = time.time()
        logger.info(f"=== Epoch {epoch}/{num_epochs} ===")

        # Compute accuracy only occasionally (e.g., before eval epochs) to save time
        compute_acc_this_epoch = (epoch % eval_every == 0) or (epoch == 1)

        avg_loss, epoch_metrics = train_epoch_multi_trace(
            model, all_traces, optimizer, config, device, logger,
            init_params=init_params, weight_decay=weight_decay,
            round_batch_size=round_batch_size,
            compute_accuracy=compute_acc_this_epoch
        )

        epoch_time = time.time() - epoch_start_time

        # Log to console (always log epoch summary with timing)
        if compute_acc_this_epoch and epoch_metrics['total'] > 0:
            logger.info(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}, "
                       f"Top1: {epoch_metrics['top1_acc']:.2f}%, "
                       f"Top8: {epoch_metrics['top8_acc']:.2f}%, "
                       f"Time: {epoch_time:.1f}s")
        else:
            logger.info(f"Epoch {epoch}/{num_epochs} - Loss: {avg_loss:.6f}, "
                       f"Time: {epoch_time:.1f}s")

        # Log to TensorBoard
        if writer is not None:
            # Basic training metrics
            writer.add_scalar('train/loss', avg_loss, epoch)
            writer.add_scalar('train/epoch_time', epoch_time, epoch)

            # Loss breakdown
            writer.add_scalar('train/attraction_loss', epoch_metrics.get('attraction_loss', avg_loss), epoch)
            if weight_decay > 0:
                writer.add_scalar('train/reg_loss', epoch_metrics.get('reg_loss', 0.0), epoch)
                writer.add_scalar('train/reg_loss_weighted', epoch_metrics.get('reg_loss', 0.0) * weight_decay, epoch)

            # Gradient and parameter norms
            writer.add_scalar('train/grad_norm', epoch_metrics.get('grad_norm', 0.0), epoch)
            param_norm = compute_parameter_norm(model)
            writer.add_scalar('train/param_norm', param_norm, epoch)

            # Learning rate
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/learning_rate', current_lr, epoch)

            # GPU memory (if available)
            if device.type == 'cuda':
                gpu_memory_allocated = torch.cuda.memory_allocated(device) / 1024**3  # GB
                gpu_memory_reserved = torch.cuda.memory_reserved(device) / 1024**3  # GB
                writer.add_scalar('system/gpu_memory_allocated_gb', gpu_memory_allocated, epoch)
                writer.add_scalar('system/gpu_memory_reserved_gb', gpu_memory_reserved, epoch)

            # Accuracy metrics
            if compute_acc_this_epoch and epoch_metrics['total'] > 0:
                writer.add_scalar('train/top1_accuracy', epoch_metrics['top1_acc'], epoch)
                writer.add_scalar('train/top8_accuracy', epoch_metrics['top8_acc'], epoch)
                writer.add_scalar('train/recent_rate', epoch_metrics['recent_rate'], epoch)

                # Per-trace metrics
                for trace_m in epoch_metrics.get('trace_metrics', []):
                    writer.add_scalar(f'train_trace/{trace_m["name"]}/loss', trace_m['loss'], epoch)
                    writer.add_scalar(f'train_trace/{trace_m["name"]}/top1_acc', trace_m['top1_acc'], epoch)
                    writer.add_scalar(f'train_trace/{trace_m["name"]}/top8_acc', trace_m['top8_acc'], epoch)

            # Per-parameter stats (every 5 epochs to avoid too much data)
            if epoch % 5 == 0 or epoch == 1:
                param_stats = compute_parameter_stats(model)
                for name, stats in param_stats.items():
                    writer.add_scalar(f'params/{name}/norm', stats['norm'], epoch)
                    writer.add_scalar(f'params/{name}/mean', stats['mean'], epoch)
                    writer.add_scalar(f'params/{name}/std', stats['std'], epoch)
                    if 'grad_norm' in stats:
                        writer.add_scalar(f'grads/{name}/norm', stats['grad_norm'], epoch)

        # Save best model
        if avg_loss < best_loss and avg_loss > 0:
            best_loss = avg_loss
            best_checkpoint_path = checkpoints_dir / 'best_model_multi_trace_v5.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': epoch_metrics,
                'config': config,
                'num_traces': len(all_traces),
            }, best_checkpoint_path)

        # Periodic checkpoint
        if epoch % save_every == 0:
            checkpoint_path = checkpoints_dir / f'checkpoint_multi_trace_v5_epoch_{epoch}.pt'
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'metrics': epoch_metrics,
                'config': config,
                'num_traces': len(all_traces),
            }, checkpoint_path)
            logger.info(f"Checkpoint saved: {checkpoint_path}")

        # Periodic evaluation on test set
        if epoch % eval_every == 0:
            logger.info(f"=== Evaluating on test set (epoch {epoch}) ===")
            eval_results = evaluate_on_test(model, config, device, logger, test_trace_data)

            # Log evaluation results
            if writer is not None:
                for k_val, metrics in eval_results['top1'].items():
                    writer.add_scalar(f'eval/top1_K{k_val}', metrics['hit_rate'], epoch)
                for k_val, metrics in eval_results['top8'].items():
                    writer.add_scalar(f'eval/top8_K{k_val}', metrics['hit_rate'], epoch)

            logger.info("Test set results (Top-1 bin):")
            for k_val, metrics in eval_results['top1'].items():
                logger.info(f"  K={k_val}: {metrics['hit_rate']:.2f}%")

    # Final checkpoint
    final_checkpoint_path = checkpoints_dir / 'final_model_multi_trace_v5.pt'
    torch.save({
        'epoch': num_epochs,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': avg_loss,
        'metrics': epoch_metrics,
        'config': config,
        'num_traces': len(all_traces),
    }, final_checkpoint_path)

    total_time = time.time() - training_start_time
    logger.info(f"Final checkpoint saved: {final_checkpoint_path}")
    logger.info(f"Training completed. Final loss: {avg_loss:.6f}, Best loss: {best_loss:.6f}")
    logger.info(f"Total training time: {total_time:.1f}s ({total_time/60:.1f} min)")

    if writer is not None:
        writer.close()

    return final_checkpoint_path


def main():
    """Main entry point."""
    import argparse

    parser = argparse.ArgumentParser(description='Multi-Trace Module 2 Training (V5 - Batched Forward)')
    parser.add_argument(
        '--weight-decay',
        type=float,
        default=7e-5,
        help='Regularization strength (default: 7e-5)'
    )
    parser.add_argument(
        '--round-batch-size',
        type=int,
        default=64,
        help='Number of rounds to process together (default: 64)'
    )
    parser.add_argument(
        '--eval-every',
        type=int,
        default=2,
        help='Evaluate on test set every N epochs (default: 2)'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=None,
        help='Override number of epochs from config'
    )
    parser.add_argument(
        '--no-tensorboard',
        action='store_true',
        help='Disable TensorBoard logging'
    )
    args = parser.parse_args()

    exp_dir = Path(__file__).parent
    config_path = exp_dir / 'config.yaml'

    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Override epochs if specified
    if args.epochs is not None:
        config['training']['epochs'] = args.epochs

    logger = setup_logging(config)

    # Fixed settings based on experiments
    use_l2_norm = False
    invert_to_origin = False
    logger.info(f"Using fixed settings: use_l2_norm={use_l2_norm}, invert_to_origin={invert_to_origin}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Round batch size: {args.round_batch_size}")
    logger.info(f"Eval every: {args.eval_every} epochs")

    try:
        final_checkpoint = train(
            config, logger,
            use_l2_norm=use_l2_norm,
            invert_to_origin=invert_to_origin,
            weight_decay=args.weight_decay,
            round_batch_size=args.round_batch_size,
            eval_every=args.eval_every,
            use_tensorboard=not args.no_tensorboard
        )
        logger.info(f"Training successful. Final checkpoint: {final_checkpoint}")
    except Exception as e:
        logger.error(f"Training failed: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
