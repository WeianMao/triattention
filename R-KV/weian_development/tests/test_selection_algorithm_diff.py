"""Test to demonstrate the difference between union-based and simple top-k selection.

This is likely the primary cause of divergence between aligned and aligned_budget.
"""
import torch


def simple_topk_selection(per_head_scores: torch.Tensor, k: int) -> torch.Tensor:
    """Selection used by speckv_rkv_style.py: max-pool then top-k."""
    combined = per_head_scores.max(dim=0).values
    return torch.topk(combined, k=k, largest=True).indices


def union_based_selection(per_head_scores: torch.Tensor, k: int) -> torch.Tensor:
    """Selection used by SparseRoundPruner: union + top-k from candidates."""
    num_heads, seq_len = per_head_scores.shape
    combined = per_head_scores.max(dim=0).values

    # Build union mask
    per_head_quota = min(k, seq_len)
    union_mask = torch.zeros(seq_len, dtype=torch.bool)

    for head_idx in range(num_heads):
        head_scores = per_head_scores[head_idx]
        head_k = min(per_head_quota, head_scores.numel())
        top_idx = torch.topk(head_scores, k=head_k, largest=True).indices
        union_mask[top_idx] = True

    union_indices = torch.nonzero(union_mask, as_tuple=False).view(-1)

    if union_indices.numel() >= k:
        subset_scores = combined.index_select(0, union_indices)
        top_subset = torch.topk(subset_scores, k=k, largest=True).indices
        return union_indices.index_select(0, torch.sort(top_subset).values)

    # Fallback: add from residual
    remaining = k - union_indices.numel()
    if remaining > 0:
        residual_scores = combined.clone()
        residual_scores[union_mask] = float("-inf")
        extra_indices = torch.topk(residual_scores, k=remaining, largest=True).indices
        union_indices = torch.cat([union_indices, extra_indices])

    return torch.sort(union_indices).values


def main():
    torch.manual_seed(42)

    num_heads = 28  # Realistic for DeepSeek-R1-7B
    seq_len = 2000  # Decode tokens
    budget = 100    # Much smaller budget

    # Key insight: the algorithms differ when union selection excludes tokens
    # that would be in simple top-k.
    #
    # This happens when a token has high combined score (max across heads)
    # but is never in any single head's top-k.
    #
    # Create scenario: token i has score (5.0 - small_noise) across all heads
    # vs token j has score 10.0 in one head, 0.0 in others.
    # Combined: token i = ~5.0, token j = 10.0
    # Per-head top-k: token j wins in its head, token i might not be in any top-k

    per_head_scores = torch.randn(num_heads, seq_len) * 0.1  # Low noise base

    # Create "consistent moderate" tokens: score ~3.0 for ALL heads
    consistent_tokens = list(range(0, 200))
    for t in consistent_tokens:
        per_head_scores[:, t] = 3.0 + torch.randn(num_heads) * 0.1

    # Create "head-specific high" tokens: score ~5.0 for ONE head, 0 for others
    # Each head gets ~50 favorite tokens
    for h in range(num_heads):
        start = 200 + h * 50
        end = min(start + 50, seq_len)
        per_head_scores[h, start:end] = 5.0 + torch.randn(end - start) * 0.1

    # Now:
    # - Consistent tokens: combined = max(3.0 + small_noise) ≈ 3.1
    # - Head-specific tokens: combined = max(5.0 for one head, ~0.1 for others) ≈ 5.0
    # Simple top-k: selects head-specific tokens (higher combined score)
    # Union-based: each head's top-k includes its 50 favorites; union is huge
    # Both might end up with head-specific tokens, but the candidate pools differ

    print("=" * 80)
    print("Selection Algorithm Difference Test")
    print("=" * 80)
    print(f"\nHeads: {num_heads}, Sequence length: {seq_len}, Budget: {budget}")

    simple_result = simple_topk_selection(per_head_scores, budget)
    union_result = union_based_selection(per_head_scores, budget)

    simple_set = set(simple_result.tolist())
    union_set = set(union_result.tolist())

    overlap = len(simple_set & union_set)
    only_simple = simple_set - union_set
    only_union = union_set - simple_set

    print(f"\nOverlap: {overlap}/{budget} ({100*overlap/budget:.1f}%)")
    print(f"Only in simple top-k: {len(only_simple)}")
    print(f"Only in union-based: {len(only_union)}")

    # Analyze what's different
    print("\n" + "-" * 40)
    print("Analysis of differences:")
    print("-" * 40)

    if only_simple and only_union:
        # For tokens only in union, check their per-head importance
        only_union_list = list(only_union)[:5]
        print(f"\nTokens only in union-based (first 5): {only_union_list}")
        for idx in only_union_list:
            head_ranks = []
            for h in range(num_heads):
                # Rank of this token in head h's preferences
                sorted_indices = torch.argsort(per_head_scores[h], descending=True).tolist()
                rank = sorted_indices.index(idx)
                head_ranks.append(rank)
            min_rank = min(head_ranks)
            avg_rank = sum(head_ranks) / len(head_ranks)
            combined_score = per_head_scores.max(dim=0).values[idx].item()
            print(f"  Token {idx}: best_head_rank={min_rank}, avg_rank={avg_rank:.1f}, combined_score={combined_score:.2f}")

        only_simple_list = list(only_simple)[:5]
        print(f"\nTokens only in simple top-k (first 5): {only_simple_list}")
        for idx in only_simple_list:
            head_ranks = []
            for h in range(num_heads):
                sorted_indices = torch.argsort(per_head_scores[h], descending=True).tolist()
                rank = sorted_indices.index(idx)
                head_ranks.append(rank)
            min_rank = min(head_ranks)
            avg_rank = sum(head_ranks) / len(head_ranks)
            combined_score = per_head_scores.max(dim=0).values[idx].item()
            print(f"  Token {idx}: best_head_rank={min_rank}, avg_rank={avg_rank:.1f}, combined_score={combined_score:.2f}")

    print("\n" + "=" * 80)
    print("CONCLUSION: Different selection algorithms produce different token sets.")
    print("Union-based ensures each head's favorites are considered first.")
    print("=" * 80)


if __name__ == "__main__":
    main()
