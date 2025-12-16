"""Label extraction functions for Module 1 Key Pruning.

Following specification in docs/04_training_and_labels.md and docs/01_module1_key_pruning.md.

Label semantics:
- label = 0: Key will be attended by some future Query (argmax) -> don't drop
- label = 1: Key won't be attended by any future Query -> drop
"""

import torch


def extract_pruning_labels(
    attention_trace: torch.Tensor,
    round_start: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Extract Key Pruning labels from attention trace.

    For each historical Key (position < round_start), determine if any future Query
    (position >= round_start) will have this Key as its argmax attention target.

    Args:
        attention_trace: (seq_len, seq_len) attention weights matrix.
                         attention_trace[q, k] = attention weight from query q to key k.
                         Should be causal (upper triangle is masked/zero).
        round_start: Current round start position. Keys at positions < round_start
                     are historical keys to be labeled.
        seq_len: Total sequence length.

    Returns:
        labels: (round_start,) tensor of binary labels.
                label[i] = 0 if Key at position i will be attended (retain)
                label[i] = 1 if Key at position i won't be attended (drop)

    Note:
        - We iterate through all Queries from round_start to seq_len (inclusive of
          current round and all future Queries).
        - For each Query, we find its argmax Key among historical Keys only.
        - A Key is labeled 0 (retain) if ANY Query has it as argmax.
    """
    if round_start <= 0:
        return torch.empty(0, dtype=torch.long, device=attention_trace.device)

    # Default: all historical keys should be dropped (label=1)
    labels = torch.ones(round_start, dtype=torch.long, device=attention_trace.device)

    # Iterate through all queries from round_start to end of sequence
    for q_idx in range(round_start, seq_len):
        # Get attention weights for this query to historical keys only
        # attention_trace[q_idx, :round_start] gives weights to positions 0..round_start-1
        attn_weights = attention_trace[q_idx, :round_start]

        # Find the argmax key among historical keys
        argmax_key = attn_weights.argmax().item()

        # This key will be attended by this query -> mark as retain (label=0)
        labels[argmax_key] = 0

    return labels


def extract_pruning_labels_vectorized(
    attention_trace: torch.Tensor,
    round_start: int,
    seq_len: int,
) -> torch.Tensor:
    """
    Vectorized version of extract_pruning_labels.

    More efficient for large sequences.

    Args:
        attention_trace: (seq_len, seq_len) attention weights matrix.
        round_start: Current round start position.
        seq_len: Total sequence length.

    Returns:
        labels: (round_start,) tensor of binary labels.
    """
    if round_start <= 0:
        return torch.empty(0, dtype=torch.long, device=attention_trace.device)

    # Get attention submatrix: queries from round_start onward, keys from 0 to round_start
    # Shape: (num_future_queries, round_start)
    future_attention = attention_trace[round_start:seq_len, :round_start]

    # Find argmax for each query
    # Shape: (num_future_queries,)
    argmax_keys = future_attention.argmax(dim=1)

    # Create labels tensor (default: drop = 1)
    labels = torch.ones(round_start, dtype=torch.long, device=attention_trace.device)

    # Mark all argmax positions as retain (label = 0)
    # Using unique to avoid redundant operations
    unique_argmax = argmax_keys.unique()
    labels[unique_argmax] = 0

    return labels


def compute_loss_mask(
    key_positions: torch.Tensor,
    seq_len: int,
    exclude_tail: int = 1000,
) -> torch.Tensor:
    """
    Compute mask for loss calculation, excluding keys in the tail of the sequence.

    Per docs/04_training_and_labels.md:
    "Training时，位置在序列末尾 1k 范围内的 Key 不计算 loss"

    Args:
        key_positions: (num_keys,) tensor of key positions.
        seq_len: Total sequence length.
        exclude_tail: Number of positions at the end to exclude (default: 1000).

    Returns:
        valid_mask: (num_keys,) boolean tensor.
                    True for keys that should be included in loss calculation.
    """
    return key_positions < (seq_len - exclude_tail)
