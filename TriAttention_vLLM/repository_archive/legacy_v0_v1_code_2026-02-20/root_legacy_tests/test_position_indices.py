"""Quick test to verify position_indices support in Triton kernel."""
import torch

def test_position_indices_interface():
    """Test that the Triton kernel accepts position_indices parameter."""
    from triattention.kernels.triton_scoring import speckv_scoring

    # Mock inputs
    batch_size = 2
    num_heads = 4
    seq_len = 8
    head_dim = 64
    freq_count = head_dim // 2
    num_offsets = 4

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Create dummy tensors
    K_rot = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.bfloat16)
    position_indices = torch.arange(seq_len, device=device, dtype=torch.int32)  # Sequential positions
    q_mean_real = torch.randn(num_heads, freq_count, device=device, dtype=torch.bfloat16)
    q_mean_imag = torch.randn(num_heads, freq_count, device=device, dtype=torch.bfloat16)
    q_abs_mean = torch.randn(num_heads, freq_count, device=device, dtype=torch.bfloat16).abs()
    freq_scale_sq = torch.ones(num_heads, freq_count, device=device, dtype=torch.bfloat16)
    omega = torch.linspace(0.1, 1.0, freq_count, device=device, dtype=torch.bfloat16)
    offsets = torch.tensor([1, 2, 4, 8], device=device, dtype=torch.bfloat16)
    round_start = 10

    # Call kernel
    scores = speckv_scoring(
        K_rot=K_rot,
        position_indices=position_indices,
        q_mean_real=q_mean_real,
        q_mean_imag=q_mean_imag,
        q_abs_mean=q_abs_mean,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets,
        round_start=round_start,
        aggregation="max",
    )

    # Verify output shape
    assert scores.shape == (batch_size, num_heads, seq_len), \
        f"Expected shape {(batch_size, num_heads, seq_len)}, got {scores.shape}"

    print(f"✓ Interface test passed: position_indices parameter accepted")
    print(f"  Output shape: {scores.shape}")
    print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")

    # Test with out-of-order positions
    position_indices_shuffled = torch.randperm(seq_len, device=device, dtype=torch.int32)
    scores_shuffled = speckv_scoring(
        K_rot=K_rot,
        position_indices=position_indices_shuffled,
        q_mean_real=q_mean_real,
        q_mean_imag=q_mean_imag,
        q_abs_mean=q_abs_mean,
        freq_scale_sq=freq_scale_sq,
        omega=omega,
        offsets=offsets,
        round_start=round_start,
        aggregation="max",
    )

    print(f"✓ Out-of-order positions test passed")
    print(f"  Shuffled positions: {position_indices_shuffled.tolist()}")
    print(f"  Score range: [{scores_shuffled.min().item():.4f}, {scores_shuffled.max().item():.4f}]")

    return True

if __name__ == "__main__":
    try:
        test_position_indices_interface()
        print("\n✓✓✓ All tests passed ✓✓✓")
    except Exception as e:
        print(f"\n✗✗✗ Test failed: {e} ✗✗✗")
        import traceback
        traceback.print_exc()
