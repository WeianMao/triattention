"""
Test Triton scoring kernel against PyTorch reference implementation.

Verifies:
1. Numerical correctness vs PyTorch
2. Batch/head/sequence dimension handling
3. Aggregation modes (max, mean)
"""

import torch
import pytest
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from triattention.kernels.triton_scoring import speckv_scoring


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to complex pairs (interleaved format: r0,i0,r1,i1,...).

    HuggingFace Transformers RoPE outputs use interleaved format.
    """
    freq_count = tensor.shape[-1] // 2
    # Interleaved format: even indices are real, odd indices are imaginary
    real = tensor[..., 0::2].contiguous()
    imag = tensor[..., 1::2].contiguous()
    return torch.complex(real, imag)


def pytorch_reference_scoring(
    K_rot: torch.Tensor,
    position_indices: torch.Tensor,
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    round_start: int,
    aggregation: str = "max",
) -> torch.Tensor:
    """
    Reference implementation matching the corrected Triton kernel.

    Key insight: When using K_rot (already rotated keys), the RoPE position info
    is "baked in". The phase only depends on query position t, not key position p.

    Formula:
        score = sum_f(A_f * cos(t*omega_f) - B_f * sin(t*omega_f)) + extra

    Where:
        A_f = freq_scale^2 * Re(Q_mean * conj(K_rot))
        B_f = freq_scale^2 * Im(Q_mean * conj(K_rot))
        t = round_start + offset

    See docs/RKV_EQUIVALENCE_FIX.md for derivation.

    Args:
        K_rot: [batch, num_heads, seq_len, head_dim]
        position_indices: [batch, seq_len] or [seq_len] (not used in corrected formula)
        q_mean_complex: [num_heads, freq_count]
        q_abs_mean: [num_heads, freq_count]
        freq_scale_sq: [num_heads, freq_count]
        omega: [freq_count]
        offsets: [num_offsets]
        round_start: int
        aggregation: "max" or "mean"

    Returns:
        scores: [batch, num_heads, seq_len]
    """
    batch_size, num_heads, seq_len, head_dim = K_rot.shape
    device = K_rot.device

    # Convert K_rot to complex (interleaved format)
    k_complex = to_complex_pairs(K_rot)  # [batch, num_heads, seq_len, freq_count]

    # Compute Q_mean * conj(K_rot)
    # Broadcast: [num_heads, freq_count] * [batch, num_heads, seq_len, freq_count]
    Z = q_mean_complex[None, :, None, :] * torch.conj(k_complex)

    # Position-independent coefficients (no atan2 needed!)
    # A = freq_scale^2 * Re(Z)
    # B = freq_scale^2 * Im(Z)
    A_coef = freq_scale_sq[None, :, None, :] * Z.real  # [batch, num_heads, seq_len, freq_count]
    B_coef = freq_scale_sq[None, :, None, :] * Z.imag  # [batch, num_heads, seq_len, freq_count]

    # Extra term: (|q_abs_mean| - |q_mean|) * |k_rot| * freq_scale^2
    q_mean_abs = torch.abs(q_mean_complex)  # [num_heads, freq_count]
    k_abs = torch.abs(k_complex)  # [batch, num_heads, seq_len, freq_count]
    extra = (q_abs_mean[None, :, None, :] - q_mean_abs[None, :, None, :]) * k_abs * freq_scale_sq[None, :, None, :]
    extra_sum = extra.sum(dim=-1)  # [batch, num_heads, seq_len]

    # Compute scores for each offset
    all_scores = []
    for offset in offsets:
        t = round_start + offset.item()

        # Phase only depends on t (query position), NOT key position p
        # This is the key correction!
        phase = t * omega  # [freq_count]

        cos_phase = torch.cos(phase)  # [freq_count]
        sin_phase = torch.sin(phase)  # [freq_count]

        # Base score: A * cos(t*omega) - B * sin(t*omega), summed over frequencies
        base_scores = (
            A_coef * cos_phase - B_coef * sin_phase
        ).sum(dim=-1)  # [batch, num_heads, seq_len]

        combined = base_scores + extra_sum
        all_scores.append(combined)

    # Stack and aggregate
    all_scores = torch.stack(all_scores, dim=-1)  # [batch, num_heads, seq_len, num_offsets]

    if aggregation == "max":
        return all_scores.max(dim=-1).values
    else:
        return all_scores.mean(dim=-1)


def build_omega(head_dim: int, device: torch.device, base: float = 10000.0) -> torch.Tensor:
    """Build RoPE angular frequencies."""
    freq_count = head_dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, freq_count, device=device).float() / freq_count))
    return inv_freq


@pytest.mark.parametrize("batch_size", [1, 2])
@pytest.mark.parametrize("num_heads", [1, 4])
@pytest.mark.parametrize("seq_len", [32, 128])
@pytest.mark.parametrize("head_dim", [64, 128])
@pytest.mark.parametrize("aggregation", ["max", "mean"])
def test_scoring_kernel_correctness(batch_size, num_heads, seq_len, head_dim, aggregation):
    """Test Triton kernel matches PyTorch reference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA not available")

    freq_count = head_dim // 2
    round_start = 100
    num_offsets = 8

    # Generate test data
    torch.manual_seed(42)
    K_rot = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    q_mean_real = torch.randn(num_heads, freq_count, device=device, dtype=torch.float32)
    q_mean_imag = torch.randn(num_heads, freq_count, device=device, dtype=torch.float32)
    q_mean_complex = torch.complex(q_mean_real, q_mean_imag)

    q_abs_mean = torch.abs(torch.randn(num_heads, freq_count, device=device, dtype=torch.float32))
    freq_scale_sq = torch.ones(num_heads, freq_count, device=device, dtype=torch.float32)

    offsets = torch.tensor([1.0, 2.0, 4.0, 8.0, 16.0, 32.0, 64.0, 128.0], device=device)

    # Build position_indices (sequential positions)
    position_indices = torch.arange(seq_len, device=device, dtype=torch.long)

    # Build omega (RoPE frequencies)
    omega = build_omega(head_dim, device)

    # Compute reference scores
    ref_scores = pytorch_reference_scoring(
        K_rot, position_indices, q_mean_complex, q_abs_mean, freq_scale_sq, omega, offsets, round_start, aggregation
    )

    # Compute Triton scores
    triton_scores = speckv_scoring(
        K_rot, position_indices, q_mean_real, q_mean_imag, q_abs_mean, freq_scale_sq, omega, offsets, round_start, aggregation
    )

    # Compare
    max_diff = (ref_scores - triton_scores).abs().max().item()
    mean_diff = (ref_scores - triton_scores).abs().mean().item()

    print(f"\nTest: batch={batch_size}, heads={num_heads}, seq={seq_len}, dim={head_dim}, agg={aggregation}")
    print(f"  Max diff: {max_diff:.6e}")
    print(f"  Mean diff: {mean_diff:.6e}")

    # Allow small numerical differences
    assert max_diff < 1e-3, f"Max difference {max_diff} exceeds threshold"
    assert mean_diff < 1e-4, f"Mean difference {mean_diff} exceeds threshold"


def test_basic_functionality():
    """Quick smoke test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        pytest.skip("CUDA not available")

    batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 128
    freq_count = head_dim // 2
    num_offsets = 16

    torch.manual_seed(42)
    K_rot = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    q_mean_real = torch.randn(num_heads, freq_count, device=device)
    q_mean_imag = torch.randn(num_heads, freq_count, device=device)
    q_abs_mean = torch.abs(torch.randn(num_heads, freq_count, device=device))
    freq_scale_sq = torch.ones(num_heads, freq_count, device=device)

    # Build position_indices and omega
    round_start = 1000
    position_indices = torch.arange(seq_len, device=device, dtype=torch.long)
    omega = build_omega(head_dim, device)
    offsets = torch.tensor([float(2**i) for i in range(num_offsets)], device=device)

    # Run kernel
    scores = speckv_scoring(
        K_rot, position_indices, q_mean_real, q_mean_imag, q_abs_mean, freq_scale_sq, omega, offsets, round_start, "max"
    )

    # Basic checks
    assert scores.shape == (batch_size, num_heads, seq_len)
    assert not torch.isnan(scores).any()
    assert not torch.isinf(scores).any()

    print("\nBasic functionality test passed")
    print(f"  Output shape: {scores.shape}")
    print(f"  Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")


if __name__ == "__main__":
    # Run quick test
    test_basic_functionality()

    # Run parametrized tests
    pytest.main([__file__, "-v", "-s"])
