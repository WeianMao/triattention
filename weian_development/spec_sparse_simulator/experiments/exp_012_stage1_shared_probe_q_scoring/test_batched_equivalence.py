"""
Equivalence tests for batched vs original forward methods in model_v5.py

This test verifies that:
1. forward_keys_batched produces identical results to sequential forward_keys calls
2. forward_queries_batched produces identical results to sequential forward_queries calls

Uses torch.isclose for numerical comparison.
"""

import torch
import torch.nn.functional as F
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from model_v5 import (
    Module2Network,
    apply_rope_rotation,
    apply_rope_rotation_batched,
)


def test_apply_rope_rotation_batched():
    """Test that batched RoPE rotation matches sequential rotation."""
    print("=" * 60)
    print("Testing apply_rope_rotation_batched equivalence...")

    torch.manual_seed(42)

    num_vectors = 128
    head_dim = 128
    batch_size = 8

    # Random vectors
    vectors = torch.randn(num_vectors, head_dim)

    # Random positions for batch
    positions = torch.tensor([100, 200, 300, 400, 500, 600, 700, 800], dtype=torch.float32)

    # Batched rotation
    rotated_batch = apply_rope_rotation_batched(vectors, positions)
    assert rotated_batch.shape == (batch_size, num_vectors, head_dim), \
        f"Expected shape ({batch_size}, {num_vectors}, {head_dim}), got {rotated_batch.shape}"

    # Sequential rotation
    rotated_seq = []
    for pos in positions:
        rotated = apply_rope_rotation(vectors, pos.item())
        rotated_seq.append(rotated)
    rotated_seq = torch.stack(rotated_seq, dim=0)

    # Check equivalence
    is_close = torch.isclose(rotated_batch, rotated_seq, rtol=1e-5, atol=1e-6)
    all_close = is_close.all()

    if not all_close:
        diff = (rotated_batch - rotated_seq).abs()
        max_diff = diff.max().item()
        mean_diff = diff.mean().item()
        num_different = (~is_close).sum().item()
        print(f"  FAILED: {num_different} elements differ")
        print(f"  Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
        return False

    print("  PASSED: Batched RoPE rotation matches sequential rotation")
    return True


def test_forward_keys_batched():
    """Test that forward_keys_batched matches sequential forward_keys calls."""
    print("=" * 60)
    print("Testing forward_keys_batched equivalence...")

    torch.manual_seed(42)

    num_bins = 128
    head_dim = 128
    max_keys = 2000
    batch_size = 8
    round_window = 128

    # Create model
    model = Module2Network(num_bins=num_bins, head_dim=head_dim, use_l2_norm=False)
    model.eval()

    # Random K vectors
    K = torch.randn(max_keys, head_dim)

    # Round starts for different rounds
    round_starts = [256, 384, 512, 640, 768, 896, 1024, 1152]
    key_lengths = torch.tensor(round_starts, dtype=torch.long)
    ref_positions = torch.tensor([rs + round_window // 2 for rs in round_starts], dtype=torch.float32)

    # Batched forward
    key_probs_batch, key_mask = model.forward_keys_batched(K, ref_positions, key_lengths)
    assert key_probs_batch.shape == (batch_size, max_keys, num_bins), \
        f"Expected shape ({batch_size}, {max_keys}, {num_bins}), got {key_probs_batch.shape}"

    # Sequential forward
    all_passed = True
    for i, round_start in enumerate(round_starts):
        ref_angles = model.compute_reference_angles(round_start, round_window)

        # Only use keys up to round_start (historical keys)
        K_slice = K[:round_start]
        key_probs_seq = model.forward_keys(K_slice, ref_angles)

        # Compare with batched result (only the valid portion)
        key_probs_batch_slice = key_probs_batch[i, :round_start, :]

        is_close = torch.isclose(key_probs_batch_slice, key_probs_seq, rtol=1e-5, atol=1e-6)

        if not is_close.all():
            diff = (key_probs_batch_slice - key_probs_seq).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            num_different = (~is_close).sum().item()
            print(f"  Round {i} (round_start={round_start}): FAILED")
            print(f"    {num_different} elements differ, Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
            all_passed = False
        else:
            print(f"  Round {i} (round_start={round_start}): PASSED")

    if all_passed:
        print("  PASSED: forward_keys_batched matches sequential forward_keys")
    else:
        print("  FAILED: Some rounds do not match")

    return all_passed


def test_forward_queries_batched():
    """Test that forward_queries_batched matches sequential forward_queries calls."""
    print("=" * 60)
    print("Testing forward_queries_batched equivalence...")

    torch.manual_seed(42)

    num_bins = 128
    head_dim = 128
    num_queries = 64
    batch_size = 8
    round_window = 128

    # Create model
    model = Module2Network(num_bins=num_bins, head_dim=head_dim, use_l2_norm=False)
    model.eval()

    # Round starts for different rounds
    round_starts = [256, 384, 512, 640, 768, 896, 1024, 1152]
    ref_positions = torch.tensor([rs + round_window // 2 for rs in round_starts], dtype=torch.float32)

    # Random Q vectors for each round (same number of queries per round for simplicity)
    Q_batch = torch.randn(batch_size, num_queries, head_dim)

    # Batched forward
    bin_probs_batch = model.forward_queries_batched(Q_batch, ref_positions)
    assert bin_probs_batch.shape == (batch_size, num_queries, num_bins), \
        f"Expected shape ({batch_size}, {num_queries}, {num_bins}), got {bin_probs_batch.shape}"

    # Sequential forward
    all_passed = True
    for i, round_start in enumerate(round_starts):
        ref_angles = model.compute_reference_angles(round_start, round_window)

        Q = Q_batch[i]  # (num_queries, head_dim)
        bin_probs_seq = model.forward_queries(Q, ref_angles)

        # Compare with batched result
        bin_probs_batch_i = bin_probs_batch[i]

        is_close = torch.isclose(bin_probs_batch_i, bin_probs_seq, rtol=1e-5, atol=1e-6)

        if not is_close.all():
            diff = (bin_probs_batch_i - bin_probs_seq).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            num_different = (~is_close).sum().item()
            print(f"  Round {i} (round_start={round_start}): FAILED")
            print(f"    {num_different} elements differ, Max diff: {max_diff:.2e}, Mean diff: {mean_diff:.2e}")
            all_passed = False
        else:
            print(f"  Round {i} (round_start={round_start}): PASSED")

    if all_passed:
        print("  PASSED: forward_queries_batched matches sequential forward_queries")
    else:
        print("  FAILED: Some rounds do not match")

    return all_passed


def test_forward_keys_batched_with_l2_norm():
    """Test forward_keys_batched with L2 normalization enabled."""
    print("=" * 60)
    print("Testing forward_keys_batched equivalence (with L2 norm)...")

    torch.manual_seed(42)

    num_bins = 128
    head_dim = 128
    max_keys = 2000
    batch_size = 8
    round_window = 128

    # Create model with L2 norm enabled
    model = Module2Network(num_bins=num_bins, head_dim=head_dim, use_l2_norm=True)
    model.eval()

    # Random K vectors
    K = torch.randn(max_keys, head_dim)

    # Round starts for different rounds
    round_starts = [256, 384, 512, 640, 768, 896, 1024, 1152]
    key_lengths = torch.tensor(round_starts, dtype=torch.long)
    ref_positions = torch.tensor([rs + round_window // 2 for rs in round_starts], dtype=torch.float32)

    # Batched forward
    key_probs_batch, key_mask = model.forward_keys_batched(K, ref_positions, key_lengths)

    # Sequential forward
    all_passed = True
    for i, round_start in enumerate(round_starts):
        ref_angles = model.compute_reference_angles(round_start, round_window)
        K_slice = K[:round_start]
        key_probs_seq = model.forward_keys(K_slice, ref_angles)

        key_probs_batch_slice = key_probs_batch[i, :round_start, :]

        is_close = torch.isclose(key_probs_batch_slice, key_probs_seq, rtol=1e-5, atol=1e-6)

        if not is_close.all():
            diff = (key_probs_batch_slice - key_probs_seq).abs()
            max_diff = diff.max().item()
            mean_diff = diff.mean().item()
            num_different = (~is_close).sum().item()
            print(f"  Round {i}: FAILED, {num_different} differ, Max: {max_diff:.2e}")
            all_passed = False
        else:
            print(f"  Round {i}: PASSED")

    if all_passed:
        print("  PASSED: forward_keys_batched matches (L2 norm enabled)")
    return all_passed


def test_forward_queries_batched_with_l2_norm():
    """Test forward_queries_batched with L2 normalization enabled."""
    print("=" * 60)
    print("Testing forward_queries_batched equivalence (with L2 norm)...")

    torch.manual_seed(42)

    num_bins = 128
    head_dim = 128
    num_queries = 64
    batch_size = 8
    round_window = 128

    # Create model with L2 norm enabled
    model = Module2Network(num_bins=num_bins, head_dim=head_dim, use_l2_norm=True)
    model.eval()

    round_starts = [256, 384, 512, 640, 768, 896, 1024, 1152]
    ref_positions = torch.tensor([rs + round_window // 2 for rs in round_starts], dtype=torch.float32)

    Q_batch = torch.randn(batch_size, num_queries, head_dim)

    # Batched forward
    bin_probs_batch = model.forward_queries_batched(Q_batch, ref_positions)

    # Sequential forward
    all_passed = True
    for i, round_start in enumerate(round_starts):
        ref_angles = model.compute_reference_angles(round_start, round_window)
        Q = Q_batch[i]
        bin_probs_seq = model.forward_queries(Q, ref_angles)

        bin_probs_batch_i = bin_probs_batch[i]

        is_close = torch.isclose(bin_probs_batch_i, bin_probs_seq, rtol=1e-5, atol=1e-6)

        if not is_close.all():
            diff = (bin_probs_batch_i - bin_probs_seq).abs()
            max_diff = diff.max().item()
            num_different = (~is_close).sum().item()
            print(f"  Round {i}: FAILED, {num_different} differ, Max: {max_diff:.2e}")
            all_passed = False
        else:
            print(f"  Round {i}: PASSED")

    if all_passed:
        print("  PASSED: forward_queries_batched matches (L2 norm enabled)")
    return all_passed


def test_forward_queries_batched_with_empty_bin_mask():
    """Test forward_queries_batched with empty bin masking."""
    print("=" * 60)
    print("Testing forward_queries_batched with empty bin mask...")

    torch.manual_seed(42)

    num_bins = 128
    head_dim = 128
    num_queries = 64
    batch_size = 4
    round_window = 128

    model = Module2Network(num_bins=num_bins, head_dim=head_dim, use_l2_norm=False)
    model.eval()

    round_starts = [256, 384, 512, 640]
    ref_positions = torch.tensor([rs + round_window // 2 for rs in round_starts], dtype=torch.float32)

    Q_batch = torch.randn(batch_size, num_queries, head_dim)

    # Create different empty bin masks for each round
    empty_bin_mask_batch = torch.zeros(batch_size, num_bins, dtype=torch.bool)
    for i in range(batch_size):
        # Mark some random bins as empty
        empty_indices = torch.randperm(num_bins)[:20]
        empty_bin_mask_batch[i, empty_indices] = True

    # Batched forward with mask
    bin_probs_batch = model.forward_queries_batched(Q_batch, ref_positions, empty_bin_mask_batch)

    # Sequential forward with mask
    all_passed = True
    for i, round_start in enumerate(round_starts):
        ref_angles = model.compute_reference_angles(round_start, round_window)
        Q = Q_batch[i]
        empty_mask = empty_bin_mask_batch[i]
        bin_probs_seq = model.forward_queries(Q, ref_angles, empty_mask)

        bin_probs_batch_i = bin_probs_batch[i]

        is_close = torch.isclose(bin_probs_batch_i, bin_probs_seq, rtol=1e-5, atol=1e-6)

        if not is_close.all():
            diff = (bin_probs_batch_i - bin_probs_seq).abs()
            max_diff = diff.max().item()
            num_different = (~is_close).sum().item()
            print(f"  Round {i}: FAILED, {num_different} differ, Max: {max_diff:.2e}")
            all_passed = False
        else:
            print(f"  Round {i}: PASSED")

    if all_passed:
        print("  PASSED: forward_queries_batched with empty bin mask matches")
    return all_passed


def run_all_tests():
    """Run all equivalence tests."""
    print("\n" + "=" * 60)
    print("Running Batched vs Sequential Equivalence Tests")
    print("=" * 60 + "\n")

    results = []

    results.append(("apply_rope_rotation_batched", test_apply_rope_rotation_batched()))
    results.append(("forward_keys_batched", test_forward_keys_batched()))
    results.append(("forward_queries_batched", test_forward_queries_batched()))
    results.append(("forward_keys_batched (L2 norm)", test_forward_keys_batched_with_l2_norm()))
    results.append(("forward_queries_batched (L2 norm)", test_forward_queries_batched_with_l2_norm()))
    results.append(("forward_queries_batched (empty bin mask)", test_forward_queries_batched_with_empty_bin_mask()))

    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)

    all_passed = True
    for name, passed in results:
        status = "PASSED" if passed else "FAILED"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print("=" * 60)
    if all_passed:
        print("All tests PASSED!")
    else:
        print("Some tests FAILED!")
    print("=" * 60)

    return all_passed


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
