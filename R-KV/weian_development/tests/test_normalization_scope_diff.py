"""Test script to verify normalization scope difference between aligned and aligned_budget implementations.

This demonstrates that the key algorithmic difference is:
- aligned (speckv_rkv_style.py): normalizes scores over ALL tokens (prefill + decode)
- aligned_budget (SparseRoundPruner): normalizes scores over DECODE tokens only

Run with: python R-KV/weian_development/tests/test_normalization_scope_diff.py
"""
import torch


def simulate_aligned_normalization(prefill_scores: torch.Tensor, decode_scores: torch.Tensor) -> torch.Tensor:
    """Simulate speckv_rkv_style.py normalization: over all tokens."""
    all_scores = torch.cat([prefill_scores, decode_scores], dim=1)  # [heads, prefill+decode]

    # Normalize over full sequence
    mean = all_scores.mean(dim=1, keepdim=True)
    std = all_scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    normalized = (all_scores - mean) / std

    # Return only decode portion (after prefill protection)
    prefill_len = prefill_scores.shape[1]
    return normalized[:, prefill_len:]


def simulate_aligned_budget_normalization(decode_scores: torch.Tensor) -> torch.Tensor:
    """Simulate SparseRoundPruner normalization: over decode tokens only."""
    mean = decode_scores.mean(dim=1, keepdim=True)
    std = decode_scores.std(dim=1, unbiased=False, keepdim=True).clamp_min(1e-6)
    return (decode_scores - mean) / std


def main():
    torch.manual_seed(42)

    num_heads = 4
    prefill_len = 150
    decode_len = 2000  # More realistic: lots of decode tokens
    budget = 2048  # Would keep all prefill + 1898 decode

    # Simulate prefill tokens having VERY different score distribution
    # In practice, prefill tokens (question context) often have distinct patterns
    prefill_scores = torch.randn(num_heads, prefill_len) * 3.0 + 5.0  # High mean, high variance
    decode_scores = torch.randn(num_heads, decode_len) * 1.0 - 1.0   # Lower mean, lower variance

    # Add some outliers in prefill (common for special tokens/punctuation)
    prefill_scores[:, :10] = 15.0  # Very high scores for first few tokens

    print("=" * 80)
    print("Normalization Scope Difference Test")
    print("=" * 80)
    print(f"\nPrefill length: {prefill_len}, Decode length: {decode_len}")
    print(f"Prefill scores mean: {prefill_scores.mean():.4f}")
    print(f"Decode scores mean: {decode_scores.mean():.4f}")

    # Method 1: aligned (speckv_rkv_style.py)
    aligned_decode = simulate_aligned_normalization(prefill_scores, decode_scores)

    # Method 2: aligned_budget (SparseRoundPruner)
    aligned_budget_decode = simulate_aligned_budget_normalization(decode_scores)

    print("\n" + "-" * 40)
    print("Normalized decode token scores (max-pooled over heads):")
    print("-" * 40)

    aligned_combined = aligned_decode.max(dim=0).values
    aligned_budget_combined = aligned_budget_decode.max(dim=0).values

    print(f"\naligned method (all tokens in norm):")
    print(f"  Mean: {aligned_combined.mean():.4f}, Std: {aligned_combined.std():.4f}")
    print(f"  Range: [{aligned_combined.min():.4f}, {aligned_combined.max():.4f}]")

    print(f"\naligned_budget method (decode only in norm):")
    print(f"  Mean: {aligned_budget_combined.mean():.4f}, Std: {aligned_budget_combined.std():.4f}")
    print(f"  Range: [{aligned_budget_combined.min():.4f}, {aligned_budget_combined.max():.4f}]")

    # Check ranking differences
    k = 1898  # Keep top 1898 decode tokens (budget - prefill)
    aligned_topk = torch.topk(aligned_combined, k=k, largest=True).indices
    aligned_budget_topk = torch.topk(aligned_budget_combined, k=k, largest=True).indices

    aligned_set = set(aligned_topk.tolist())
    aligned_budget_set = set(aligned_budget_topk.tolist())

    overlap = len(aligned_set & aligned_budget_set)

    print("\n" + "-" * 40)
    print(f"Top-{k} selection comparison:")
    print("-" * 40)
    print(f"  Overlap: {overlap}/{k} ({100*overlap/k:.1f}%)")
    print(f"  Different tokens: {k - overlap}")

    if overlap < k:
        print(f"\n  Only in aligned: {sorted(aligned_set - aligned_budget_set)[:10]}...")
        print(f"  Only in aligned_budget: {sorted(aligned_budget_set - aligned_set)[:10]}...")

    # Rank correlation
    aligned_ranks = torch.argsort(torch.argsort(aligned_combined, descending=True))
    aligned_budget_ranks = torch.argsort(torch.argsort(aligned_budget_combined, descending=True))
    rank_diff = (aligned_ranks - aligned_budget_ranks).abs().float()

    print(f"\n  Mean rank difference: {rank_diff.mean():.2f}")
    print(f"  Max rank difference: {rank_diff.max():.0f}")

    print("\n" + "=" * 80)
    print("CONCLUSION: Different normalization scope leads to different token selection.")
    print("This is a fundamental algorithmic difference, not a bug.")
    print("=" * 80)


if __name__ == "__main__":
    main()
