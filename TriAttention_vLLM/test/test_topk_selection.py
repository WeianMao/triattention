"""
Test TopK selection logic.

Validates token selection, prefill protection, and budget management.
"""

import pytest
import torch


def select_tokens_to_keep(scores, budget_slots, prefill_len=0, protect_prefill=False):
    """
    Reference implementation of TopK selection.

    Args:
        scores: [total_tokens] - token scores
        budget_slots: int - number of tokens to keep
        prefill_len: int - length of prefill sequence
        protect_prefill: bool - whether to protect prefill tokens

    Returns:
        torch.Tensor: [total_tokens] - boolean mask, True = keep
    """
    total_tokens = scores.shape[0]

    if protect_prefill and prefill_len > 0:
        # Protect prefill tokens
        if prefill_len >= budget_slots:
            # Prefill alone exceeds budget, keep first budget_slots
            keep_mask = torch.zeros(total_tokens, dtype=torch.bool)
            keep_mask[:budget_slots] = True
            return keep_mask

        # Keep all prefill + top-k from decode
        decode_scores = scores[prefill_len:]
        decode_keep_count = budget_slots - prefill_len

        if decode_keep_count <= 0:
            keep_mask = torch.zeros(total_tokens, dtype=torch.bool)
            keep_mask[:prefill_len] = True
            return keep_mask

        if decode_scores.numel() == 0:
            keep_mask = torch.zeros(total_tokens, dtype=torch.bool)
            keep_mask[:prefill_len] = True
            return keep_mask

        _, top_indices = decode_scores.topk(min(decode_keep_count, decode_scores.numel()))

        keep_mask = torch.zeros(total_tokens, dtype=torch.bool)
        keep_mask[:prefill_len] = True
        keep_mask[prefill_len + top_indices] = True
    else:
        # All tokens compete
        k = min(budget_slots, total_tokens)
        _, top_indices = scores.topk(k)
        keep_mask = torch.zeros(total_tokens, dtype=torch.bool)
        keep_mask[top_indices] = True

    return keep_mask


# ==================== Test Cases ====================


def test_topk_basic_selection(deterministic_seed):
    """
    Test basic TopK selection without prefill protection.
    """
    num_tokens = 100
    budget = 50

    scores = torch.randn(num_tokens)
    keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

    # Should keep exactly budget tokens
    assert keep_mask.sum().item() == budget

    # Kept tokens should have highest scores
    kept_scores = scores[keep_mask]
    dropped_scores = scores[~keep_mask]

    assert kept_scores.min() >= dropped_scores.max()


def test_topk_with_prefill_protection(deterministic_seed):
    """
    Test TopK selection with prefill protection enabled.
    """
    total_tokens = 150
    prefill_len = 60
    budget = 100

    scores = torch.randn(total_tokens)
    keep_mask = select_tokens_to_keep(scores, budget, prefill_len, protect_prefill=True)

    # Should keep exactly budget tokens
    assert keep_mask.sum().item() == budget

    # All prefill tokens should be kept
    assert keep_mask[:prefill_len].all()

    # Decode tokens: should keep top (budget - prefill_len)
    decode_keep_count = budget - prefill_len
    decode_kept = keep_mask[prefill_len:].sum().item()
    assert decode_kept == decode_keep_count


def test_topk_prefill_exceeds_budget(deterministic_seed):
    """
    Test when prefill length exceeds budget.
    """
    total_tokens = 200
    prefill_len = 120
    budget = 100

    scores = torch.randn(total_tokens)
    scores[:prefill_len] = torch.arange(prefill_len, dtype=torch.float32)  # Increasing scores

    keep_mask = select_tokens_to_keep(scores, budget, prefill_len, protect_prefill=True)

    # Should keep exactly budget tokens
    assert keep_mask.sum().item() == budget

    # Should keep first budget tokens from prefill
    assert keep_mask[:budget].all()
    assert not keep_mask[budget:].any()


def test_topk_no_decode_tokens(deterministic_seed):
    """
    Test when there are no decode tokens (only prefill).
    """
    prefill_len = 50
    total_tokens = 50
    budget = 100

    scores = torch.randn(total_tokens)
    keep_mask = select_tokens_to_keep(scores, budget, prefill_len, protect_prefill=True)

    # Should keep all tokens
    assert keep_mask.sum().item() == total_tokens
    assert keep_mask.all()


def test_topk_score_ranking(deterministic_seed):
    """
    Test that TopK correctly ranks scores.
    """
    num_tokens = 100
    budget = 30

    # Create scores with known ordering
    scores = torch.arange(num_tokens, dtype=torch.float32)

    keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

    # Should keep top 30 (indices 70-99)
    expected_kept_indices = set(range(70, 100))
    actual_kept_indices = set(torch.where(keep_mask)[0].tolist())

    assert expected_kept_indices == actual_kept_indices


def test_topk_deterministic(deterministic_seed):
    """
    Test that TopK selection is deterministic.
    """
    num_tokens = 80
    budget = 40

    scores = torch.randn(num_tokens)

    keep_mask_1 = select_tokens_to_keep(scores, budget, protect_prefill=False)
    keep_mask_2 = select_tokens_to_keep(scores, budget, protect_prefill=False)

    assert torch.equal(keep_mask_1, keep_mask_2)


def test_topk_empty_decode(deterministic_seed):
    """
    Test when decode region is empty but budget allows more.
    """
    total_tokens = 30
    prefill_len = 30
    budget = 50

    scores = torch.randn(total_tokens)
    keep_mask = select_tokens_to_keep(scores, budget, prefill_len, protect_prefill=True)

    # Should keep all prefill tokens
    assert keep_mask.sum().item() == prefill_len
    assert keep_mask[:prefill_len].all()


def test_topk_budget_equals_total(deterministic_seed):
    """
    Test when budget equals total tokens.
    """
    num_tokens = 100
    budget = 100

    scores = torch.randn(num_tokens)
    keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

    # Should keep all tokens
    assert keep_mask.all()


def test_topk_budget_exceeds_total(deterministic_seed):
    """
    Test when budget exceeds total tokens.
    """
    num_tokens = 50
    budget = 100

    scores = torch.randn(num_tokens)
    keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

    # Should keep all tokens
    assert keep_mask.all()


def test_topk_decode_scores_ordering(deterministic_seed):
    """
    Test that decode tokens are selected by score, not position.
    """
    total_tokens = 100
    prefill_len = 40
    budget = 60

    scores = torch.zeros(total_tokens)

    # Give highest scores to middle decode tokens
    scores[60:70] = 10.0  # High scores in middle
    scores[prefill_len:60] = 1.0  # Low scores
    scores[70:] = 1.0  # Low scores

    keep_mask = select_tokens_to_keep(scores, budget, prefill_len, protect_prefill=True)

    # Prefill should be kept
    assert keep_mask[:prefill_len].all()

    # High-score decode tokens should be kept
    assert keep_mask[60:70].all()

    # Should keep exactly budget tokens
    assert keep_mask.sum().item() == budget


def test_topk_negative_scores(deterministic_seed):
    """
    Test TopK works with negative scores.
    """
    num_tokens = 80
    budget = 40

    scores = torch.randn(num_tokens) - 10.0  # All negative

    keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

    # Should still keep budget tokens (highest negatives)
    assert keep_mask.sum().item() == budget

    kept_scores = scores[keep_mask]
    dropped_scores = scores[~keep_mask]

    assert kept_scores.min() >= dropped_scores.max()


def test_topk_tied_scores(deterministic_seed):
    """
    Test TopK behavior with tied scores.
    """
    num_tokens = 100
    budget = 50

    scores = torch.ones(num_tokens)  # All same score
    scores[0] = 2.0  # One higher

    keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

    # Should keep exactly budget tokens
    assert keep_mask.sum().item() == budget

    # Token 0 should always be kept
    assert keep_mask[0].item()


def test_topk_prefill_boundary_cases():
    """
    Test edge cases at prefill/decode boundary.
    """
    # Case 1: prefill_len = 0
    scores = torch.randn(100)
    keep_mask = select_tokens_to_keep(scores, 50, prefill_len=0, protect_prefill=True)
    assert keep_mask.sum().item() == 50

    # Case 2: prefill_len = total_tokens
    scores = torch.randn(100)
    keep_mask = select_tokens_to_keep(scores, 50, prefill_len=100, protect_prefill=True)
    assert keep_mask.sum().item() == 50

    # Case 3: budget = prefill_len
    scores = torch.randn(100)
    keep_mask = select_tokens_to_keep(scores, 50, prefill_len=50, protect_prefill=True)
    assert keep_mask.sum().item() == 50
    assert keep_mask[:50].all()


def test_topk_indices_validity(deterministic_seed):
    """
    Test that kept indices are valid.
    """
    num_tokens = 100
    budget = 60

    scores = torch.randn(num_tokens)
    keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

    kept_indices = torch.where(keep_mask)[0]

    # All indices should be in valid range
    assert kept_indices.min() >= 0
    assert kept_indices.max() < num_tokens

    # No duplicate indices
    assert len(kept_indices.unique()) == len(kept_indices)


def test_topk_with_inf_scores():
    """
    Test TopK handles infinite scores correctly.
    """
    num_tokens = 100
    budget = 50

    scores = torch.randn(num_tokens)
    scores[10:15] = float("inf")  # Some infinite scores
    scores[20:22] = float("-inf")  # Some negative infinite

    keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

    # Should keep exactly budget tokens
    assert keep_mask.sum().item() == budget

    # Positive infinity should be kept
    assert keep_mask[10:15].all()

    # Negative infinity should not be kept
    assert not keep_mask[20:22].any()


def test_topk_shape_preservation():
    """
    Test that output mask has same shape as input scores.
    """
    for num_tokens in [10, 50, 100, 500]:
        scores = torch.randn(num_tokens)
        budget = num_tokens // 2

        keep_mask = select_tokens_to_keep(scores, budget, protect_prefill=False)

        assert keep_mask.shape == scores.shape
        assert keep_mask.dtype == torch.bool


def test_topk_protected_prefill_decode_ranking(deterministic_seed):
    """
    Test that with prefill protection, decode tokens are still ranked correctly.
    """
    total_tokens = 120
    prefill_len = 50
    budget = 80

    scores = torch.randn(total_tokens)

    # Set specific scores for decode region
    decode_start = prefill_len
    scores[decode_start : decode_start + 10] = 100.0  # Top scores
    scores[decode_start + 10 : decode_start + 20] = 50.0  # Medium scores
    scores[decode_start + 20 :] = 1.0  # Low scores

    keep_mask = select_tokens_to_keep(scores, budget, prefill_len, protect_prefill=True)

    # All prefill kept
    assert keep_mask[:prefill_len].all()

    # Top decode tokens should be kept
    decode_keep_count = budget - prefill_len
    assert keep_mask[decode_start : decode_start + decode_keep_count].sum() >= 10

    # Specifically, top 10 should all be kept
    assert keep_mask[decode_start : decode_start + 10].all()
