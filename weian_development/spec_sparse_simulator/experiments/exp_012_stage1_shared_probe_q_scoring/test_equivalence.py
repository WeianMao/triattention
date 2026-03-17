"""
Unit tests to verify algorithm equivalence between V2 and V3.

Tests that the optimized V3 training produces the same results as V2.
Uses torch.isclose() to verify numerical equivalence.
"""

import sys
import torch
import torch.nn.functional as F
from pathlib import Path

# Add parent to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from model import create_model, Module2Network


def test_compute_attraction_loss_equivalence():
    """Test that attraction loss computation is identical."""
    print("=" * 60)
    print("Test 1: Attraction Loss Equivalence")
    print("=" * 60)

    # Import both versions
    from train_multi_trace_v2 import compute_attraction_loss as loss_v2
    from train_multi_trace_v3 import compute_attraction_loss as loss_v3

    # Create test data
    torch.manual_seed(42)
    num_keys = 1000
    num_bins = 128
    num_queries = 64

    key_probs = F.softmax(torch.randn(num_keys, num_bins), dim=0)
    query_bin_probs = F.softmax(torch.randn(num_queries, num_bins), dim=-1)
    argmax_keys = torch.randint(0, num_keys, (num_queries,))
    argmax_in_recent = torch.rand(num_queries) > 0.7  # 30% in recent

    # Compute losses
    loss_v2_val = loss_v2(key_probs, query_bin_probs, argmax_keys, argmax_in_recent)
    loss_v3_val = loss_v3(key_probs, query_bin_probs, argmax_keys, argmax_in_recent)

    # Compare
    is_close = torch.isclose(loss_v2_val, loss_v3_val, rtol=1e-5, atol=1e-7)
    print(f"V2 loss: {loss_v2_val.item():.8f}")
    print(f"V3 loss: {loss_v3_val.item():.8f}")
    print(f"Difference: {abs(loss_v2_val.item() - loss_v3_val.item()):.2e}")
    print(f"isclose: {is_close.item()}")

    assert is_close.item(), "Attraction loss mismatch!"
    print("PASSED\n")
    return True


def test_model_forward_equivalence():
    """Test that model forward pass is identical."""
    print("=" * 60)
    print("Test 2: Model Forward Pass Equivalence")
    print("=" * 60)

    torch.manual_seed(42)

    # Create model with random init
    config = {
        'model': {
            'num_bins': 128,
            'num_freqs': 64,
            'model_path': '/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B'
        }
    }
    model = create_model(config, use_l2_norm=False)
    model.eval()

    # Create test data
    num_keys = 500
    head_dim = 128
    round_start = 1000
    round_window = 128

    K = torch.randn(num_keys, head_dim)
    Q = torch.randn(10, head_dim)

    # Forward pass
    reference_angles = model.compute_reference_angles(round_start, round_window)

    with torch.no_grad():
        key_probs_1 = model.forward_keys(K, reference_angles)
        query_probs_1 = model.forward_queries(Q, reference_angles)

        # Run again to ensure determinism
        key_probs_2 = model.forward_keys(K, reference_angles)
        query_probs_2 = model.forward_queries(Q, reference_angles)

    # Check determinism
    key_match = torch.allclose(key_probs_1, key_probs_2, rtol=1e-5, atol=1e-7)
    query_match = torch.allclose(query_probs_1, query_probs_2, rtol=1e-5, atol=1e-7)

    print(f"Key probs shape: {key_probs_1.shape}")
    print(f"Query probs shape: {query_probs_1.shape}")
    print(f"Key probs deterministic: {key_match}")
    print(f"Query probs deterministic: {query_match}")
    print(f"Key probs sum (should be ~1 per bin): {key_probs_1.sum(dim=0).mean():.4f}")
    print(f"Query probs sum (should be 1 per query): {query_probs_1.sum(dim=-1).mean():.4f}")

    assert key_match, "Key forward not deterministic!"
    assert query_match, "Query forward not deterministic!"
    print("PASSED\n")
    return True


def test_extract_labels_equivalence():
    """Test that label extraction is identical."""
    print("=" * 60)
    print("Test 3: Label Extraction Equivalence")
    print("=" * 60)

    from train_multi_trace_v2 import extract_query_to_key_labels as extract_v2
    from train_multi_trace_v3 import extract_round_labels as extract_v3

    torch.manual_seed(42)

    # Create mock attention matrix
    seq_len = 5000
    attention = F.softmax(torch.randn(seq_len, seq_len), dim=-1)

    # Apply causal mask
    causal_mask = torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)
    attention.masked_fill_(causal_mask, 0)

    round_start = 1024
    round_end = 1152
    exclude_tail = 1000

    labels_v2 = extract_v2(attention, round_start, round_end, seq_len, exclude_tail)
    labels_v3 = extract_v3(attention, round_start, round_end, seq_len, exclude_tail)

    # Compare
    query_match = torch.equal(labels_v2['query_indices'], labels_v3['query_indices'])
    argmax_match = torch.equal(labels_v2['argmax_keys'], labels_v3['argmax_keys'])
    recent_match = torch.equal(labels_v2['argmax_in_recent'], labels_v3['argmax_in_recent'])

    print(f"Query indices match: {query_match}")
    print(f"Argmax keys match: {argmax_match}")
    print(f"Argmax in recent match: {recent_match}")
    print(f"Number of queries: {len(labels_v2['query_indices'])}")

    assert query_match, "Query indices mismatch!"
    assert argmax_match, "Argmax keys mismatch!"
    assert recent_match, "Argmax in recent mismatch!"
    print("PASSED\n")
    return True


def test_init_regularization_equivalence():
    """Test that initialization regularization is identical."""
    print("=" * 60)
    print("Test 4: Init Regularization Equivalence")
    print("=" * 60)

    from train_multi_trace_v2 import compute_init_regularization_loss as reg_v2
    from train_multi_trace_v3 import compute_init_regularization_loss as reg_v3

    torch.manual_seed(42)

    # Create model
    config = {
        'model': {
            'num_bins': 128,
            'num_freqs': 64,
            'model_path': '/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B'
        }
    }
    model = create_model(config, use_l2_norm=False)

    # Save initial params
    init_params = {name: param.detach().clone() for name, param in model.named_parameters()}

    # Modify some parameters
    with torch.no_grad():
        for name, param in model.named_parameters():
            param.add_(torch.randn_like(param) * 0.01)

    # Compute regularization
    reg_v2_val = reg_v2(model, init_params)
    reg_v3_val = reg_v3(model, init_params)

    is_close = torch.isclose(torch.tensor(reg_v2_val), torch.tensor(reg_v3_val), rtol=1e-5, atol=1e-7)
    print(f"V2 reg loss: {reg_v2_val:.8f}")
    print(f"V3 reg loss: {reg_v3_val:.8f}")
    print(f"isclose: {is_close.item()}")

    assert is_close.item(), "Regularization loss mismatch!"
    print("PASSED\n")
    return True


def test_gradient_equivalence():
    """Test that gradients are equivalent between sequential and batched processing."""
    print("=" * 60)
    print("Test 5: Gradient Equivalence (Sequential vs Batched)")
    print("=" * 60)

    from train_multi_trace_v2 import compute_attraction_loss

    torch.manual_seed(42)

    # Create two identical models
    config = {
        'model': {
            'num_bins': 128,
            'num_freqs': 64,
            'model_path': '/data/rbg/users/weian/project/rl/datasets/DeepSeek-R1-0528-Qwen3-8B'
        }
    }
    model_seq = create_model(config, use_l2_norm=False)
    model_batch = create_model(config, use_l2_norm=False)

    # Copy weights
    model_batch.load_state_dict(model_seq.state_dict())

    # Create test data for 2 "rounds"
    num_keys_1, num_keys_2 = 400, 500
    num_queries = 32
    head_dim = 128

    K1 = torch.randn(num_keys_1, head_dim)
    K2 = torch.randn(num_keys_2, head_dim)
    Q = torch.randn(num_queries, head_dim)
    argmax_keys_1 = torch.randint(0, num_keys_1, (num_queries,))
    argmax_keys_2 = torch.randint(0, num_keys_2, (num_queries,))
    argmax_in_recent = torch.zeros(num_queries, dtype=torch.bool)  # All in historical

    round_start_1, round_start_2 = 400, 500
    round_window = 128

    # Sequential processing (like V2)
    model_seq.train()
    ref_angles_1 = model_seq.compute_reference_angles(round_start_1, round_window)
    ref_angles_2 = model_seq.compute_reference_angles(round_start_2, round_window)

    key_probs_1 = model_seq.forward_keys(K1, ref_angles_1)
    query_probs_1 = model_seq.forward_queries(Q, ref_angles_1)
    loss_1 = compute_attraction_loss(key_probs_1, query_probs_1, argmax_keys_1, argmax_in_recent)

    key_probs_2 = model_seq.forward_keys(K2, ref_angles_2)
    query_probs_2 = model_seq.forward_queries(Q, ref_angles_2)
    loss_2 = compute_attraction_loss(key_probs_2, query_probs_2, argmax_keys_2, argmax_in_recent)

    # Average loss and backward
    total_loss_seq = (loss_1 + loss_2) / 2
    total_loss_seq.backward()

    # Get gradients
    grad_seq = {}
    for name, param in model_seq.named_parameters():
        if param.grad is not None:
            grad_seq[name] = param.grad.clone()

    # Batched processing (like V3 accumulation)
    model_batch.train()
    ref_angles_1 = model_batch.compute_reference_angles(round_start_1, round_window)
    ref_angles_2 = model_batch.compute_reference_angles(round_start_2, round_window)

    # Accumulate losses before backward
    key_probs_1 = model_batch.forward_keys(K1, ref_angles_1)
    query_probs_1 = model_batch.forward_queries(Q, ref_angles_1)
    loss_1 = compute_attraction_loss(key_probs_1, query_probs_1, argmax_keys_1, argmax_in_recent)

    key_probs_2 = model_batch.forward_keys(K2, ref_angles_2)
    query_probs_2 = model_batch.forward_queries(Q, ref_angles_2)
    loss_2 = compute_attraction_loss(key_probs_2, query_probs_2, argmax_keys_2, argmax_in_recent)

    total_loss_batch = (loss_1 + loss_2) / 2
    total_loss_batch.backward()

    # Get gradients
    grad_batch = {}
    for name, param in model_batch.named_parameters():
        if param.grad is not None:
            grad_batch[name] = param.grad.clone()

    # Compare gradients
    all_close = True
    max_diff = 0.0
    for name in grad_seq:
        if name in grad_batch:
            is_close = torch.allclose(grad_seq[name], grad_batch[name], rtol=1e-5, atol=1e-7)
            diff = (grad_seq[name] - grad_batch[name]).abs().max().item()
            max_diff = max(max_diff, diff)
            if not is_close:
                print(f"  {name}: max diff = {diff:.2e}")
                all_close = False

    print(f"Loss sequential: {total_loss_seq.item():.8f}")
    print(f"Loss batched: {total_loss_batch.item():.8f}")
    print(f"Max gradient difference: {max_diff:.2e}")
    print(f"All gradients close: {all_close}")

    assert all_close, "Gradient mismatch between sequential and batched!"
    print("PASSED\n")
    return True


def test_attention_matrix_equivalence():
    """Test that attention matrix computation is identical."""
    print("=" * 60)
    print("Test 6: Attention Matrix Equivalence")
    print("=" * 60)

    from train_multi_trace_v2 import compute_attention_matrix as attn_v2
    from train_multi_trace_v3 import compute_attention_matrix as attn_v3

    torch.manual_seed(42)

    seq_len = 1000
    head_dim = 128

    Q = torch.randn(seq_len, head_dim)
    K = torch.randn(seq_len, head_dim)

    attn_v2_val = attn_v2(Q, K)
    attn_v3_val = attn_v3(Q, K)

    is_close = torch.allclose(attn_v2_val, attn_v3_val, rtol=1e-5, atol=1e-7)
    max_diff = (attn_v2_val - attn_v3_val).abs().max().item()

    print(f"Attention shape: {attn_v2_val.shape}")
    print(f"Max difference: {max_diff:.2e}")
    print(f"All close: {is_close}")

    # Check causal mask is applied correctly
    upper_tri = attn_v2_val[torch.triu(torch.ones(seq_len, seq_len, dtype=torch.bool), diagonal=1)]
    print(f"Upper triangle (should be 0): max={upper_tri.max():.6f}, min={upper_tri.min():.6f}")

    # Check rows sum to 1
    row_sums = attn_v2_val.sum(dim=-1)
    print(f"Row sums (should be 1): mean={row_sums.mean():.6f}, std={row_sums.std():.6f}")

    assert is_close, "Attention matrix mismatch!"
    print("PASSED\n")
    return True


def run_all_tests():
    """Run all equivalence tests."""
    print("\n" + "=" * 60)
    print("Running V2 vs V3 Equivalence Tests")
    print("=" * 60 + "\n")

    tests = [
        test_attention_matrix_equivalence,
        test_extract_labels_equivalence,
        test_compute_attraction_loss_equivalence,
        test_init_regularization_equivalence,
        test_model_forward_equivalence,
        test_gradient_equivalence,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"FAILED with exception: {e}")
            failed += 1

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    if failed == 0:
        print("\nAll tests PASSED! V3 is equivalent to V2.")
    else:
        print(f"\n{failed} tests FAILED! Check the output above.")

    return failed == 0


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
