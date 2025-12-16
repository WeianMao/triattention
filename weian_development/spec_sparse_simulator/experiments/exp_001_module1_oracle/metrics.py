"""Evaluation metrics for Module 1 Key Pruning.

Following specification in docs/01_module1_key_pruning.md.

Primary metrics:
- Argmax Hit Rate: Query can still attend to original argmax Key (target >99%)
- Keys per Query: Average retained Keys (lower is better)
- Computation Reduction: 1 - (Keys per Query / N)

Secondary metrics:
- Retention Rate: Retained Keys / Total Keys
- False Negative Rate: Incorrectly dropped "should retain" Keys
"""

from typing import Dict
import torch


def compute_module1_metrics(
    drop_probs: torch.Tensor,
    labels: torch.Tensor,
    query_argmax_indices: torch.Tensor,
    argmax_in_history: torch.Tensor,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute Module 1 evaluation metrics.

    Args:
        drop_probs: (num_keys,) - Model predicted drop probabilities.
        labels: (num_keys,) - Ground truth labels (0=retain, 1=drop).
        query_argmax_indices: (num_queries,) - Index of argmax Key for each Query.
                              For queries whose argmax is in current round (not history),
                              this can be any value as argmax_in_history will be False.
        argmax_in_history: (num_queries,) - Boolean tensor indicating if Query's argmax
                           is in historical Keys (True) or current round new Keys (False).
        threshold: Drop threshold. Keys with drop_prob >= threshold are dropped.

    Returns:
        Dict with metrics:
            - retention_rate: Proportion of Keys retained
            - argmax_hit_rate: Proportion of Queries that can still attend to argmax Key
            - false_negative_rate: Proportion of "should retain" Keys that were dropped
            - keys_per_query: Average number of retained Keys (for Keys per Query)
            - computation_reduction: 1 - (retained / total)

    Hit judgment rules (from docs/01_module1_key_pruning.md):
    - argmax in history Key -> check if that Key is retained
    - argmax in current round new Key -> direct hit (Full Attention for new Keys)
    """
    num_keys = len(drop_probs)
    num_queries = len(query_argmax_indices)

    # Determine which keys are retained (drop_prob < threshold -> retain)
    retain_mask = drop_probs < threshold

    # 1. Retention Rate
    retention_rate = retain_mask.float().mean().item()

    # 2. Keys per Query (number of retained keys)
    keys_per_query = retain_mask.sum().item()

    # 3. Computation Reduction
    computation_reduction = 1.0 - (keys_per_query / num_keys) if num_keys > 0 else 0.0

    # 4. Argmax Hit Rate (most important metric)
    # Case A: argmax in current round new Key -> direct hit (Full Attention)
    hits_from_recent = (~argmax_in_history).sum().item()

    # Case B: argmax in history Key -> check if that Key is retained
    if argmax_in_history.any():
        history_query_mask = argmax_in_history
        history_argmax_indices = query_argmax_indices[history_query_mask]

        # Check if these argmax keys are retained
        # Clamp indices to valid range
        valid_indices = history_argmax_indices.clamp(0, num_keys - 1)
        hits_from_history = retain_mask[valid_indices].sum().item()
    else:
        hits_from_history = 0

    total_hits = hits_from_recent + hits_from_history
    argmax_hit_rate = total_hits / num_queries if num_queries > 0 else 1.0

    # 5. False Negative Rate (among "should retain" keys, how many were dropped)
    # "Should retain" = label == 0
    should_retain_mask = labels == 0
    num_should_retain = should_retain_mask.sum().item()

    if num_should_retain > 0:
        # Keys that should be retained but were dropped
        false_negatives = (~retain_mask & should_retain_mask).sum().item()
        false_negative_rate = false_negatives / num_should_retain
    else:
        false_negative_rate = 0.0

    return {
        "retention_rate": retention_rate,
        "argmax_hit_rate": argmax_hit_rate,
        "false_negative_rate": false_negative_rate,
        "keys_per_query": keys_per_query,
        "computation_reduction": computation_reduction,
    }


def compute_oracle_metrics(
    labels: torch.Tensor,
    query_argmax_indices: torch.Tensor,
    argmax_in_history: torch.Tensor,
) -> Dict[str, float]:
    """
    Compute metrics for Oracle Upper Bound experiment.

    Oracle uses ground-truth labels as predictions:
    - drop_probs = labels.float() (perfect prediction)
    - threshold = 0.5

    Oracle should always achieve:
    - 100% Argmax Hit Rate (by definition, since labels are based on argmax)
    - 0% False Negative Rate (perfect prediction)

    Args:
        labels: (num_keys,) - Ground truth labels (0=retain, 1=drop).
        query_argmax_indices: (num_queries,) - Index of argmax Key for each Query.
        argmax_in_history: (num_queries,) - Boolean tensor indicating if Query's argmax
                           is in historical Keys.

    Returns:
        Dict with Oracle metrics.
    """
    # Oracle prediction: drop_probs = labels (perfect prediction)
    drop_probs = labels.float()

    return compute_module1_metrics(
        drop_probs=drop_probs,
        labels=labels,
        query_argmax_indices=query_argmax_indices,
        argmax_in_history=argmax_in_history,
        threshold=0.5,
    )


def compute_round_statistics(
    labels: torch.Tensor,
    round_start: int,
    seq_len: int,
    round_window: int = 128,
) -> Dict[str, float]:
    """
    Compute statistics for a single round.

    Args:
        labels: (round_start,) - Labels for historical keys.
        round_start: Current round start position.
        seq_len: Total sequence length.
        round_window: Round window size (default: 128).

    Returns:
        Dict with round statistics:
            - num_history_keys: Number of historical keys
            - num_retain_labels: Number of keys with label=0 (should retain)
            - num_drop_labels: Number of keys with label=1 (should drop)
            - retain_label_ratio: Ratio of retain labels
            - num_queries_in_round: Number of queries in this round
    """
    num_history_keys = len(labels)
    num_retain_labels = (labels == 0).sum().item()
    num_drop_labels = (labels == 1).sum().item()

    retain_label_ratio = num_retain_labels / num_history_keys if num_history_keys > 0 else 0.0

    round_end = min(round_start + round_window, seq_len)
    num_queries_in_round = round_end - round_start

    return {
        "num_history_keys": num_history_keys,
        "num_retain_labels": num_retain_labels,
        "num_drop_labels": num_drop_labels,
        "retain_label_ratio": retain_label_ratio,
        "num_queries_in_round": num_queries_in_round,
    }
