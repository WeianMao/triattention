"""
Standalone test for Triton scoring kernel (no pytest dependency).
"""

import torch
from pathlib import Path
import sys

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from triattention.kernels.triton_scoring import speckv_scoring


def to_complex_pairs(tensor: torch.Tensor) -> torch.Tensor:
    """Convert tensor to complex pairs (front/back half pairing)."""
    freq_count = tensor.shape[-1] // 2
    real = tensor[..., :freq_count].contiguous()
    imag = tensor[..., freq_count:].contiguous()
    return torch.complex(real, imag)


def pytorch_reference_scoring(
    K_rot: torch.Tensor,
    q_mean_complex: torch.Tensor,
    q_abs_mean: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    round_start: int,
    offsets: torch.Tensor,
    aggregation: str = "max",
) -> torch.Tensor:
    """Reference implementation from R-KV."""
    batch_size, num_heads, seq_len, head_dim = K_rot.shape
    device = K_rot.device

    # Convert K_rot to complex
    k_complex = to_complex_pairs(K_rot)  # [batch, num_heads, seq_len, freq_count]

    # Compute amplitude and phase
    q_mean_abs = torch.abs(q_mean_complex)
    k_abs = torch.abs(k_complex)

    # Relative phase
    relative = q_mean_complex[None, :, None, :] * torch.conj(k_complex)
    phi_rot = torch.atan2(relative.imag, relative.real)

    # Amplitude
    amp = q_mean_abs[None, :, None, :] * k_abs

    # Extra term
    extra = (q_abs_mean[None, :, None, :] - q_mean_abs[None, :, None, :]) * k_abs

    # Omega
    freq_count = head_dim // 2
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, freq_count, device=device).float() / freq_count))
    omega = inv_freq

    # Compute scores for each offset
    all_scores = []
    for offset in offsets:
        t = round_start + offset.item()
        phase = t * omega[None, None, None, :] + phi_rot

        base_scores = (amp * freq_scale_sq[None, :, None, :] * torch.cos(phase)).sum(dim=-1)
        additive = (extra * freq_scale_sq[None, :, None, :]).sum(dim=-1)

        combined = base_scores + additive
        all_scores.append(combined)

    all_scores = torch.stack(all_scores, dim=-1)

    if aggregation == "max":
        return all_scores.max(dim=-1).values
    else:
        return all_scores.mean(dim=-1)


def build_cos_sin_tables(
    round_start: int,
    offsets: torch.Tensor,
    head_dim: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build precomputed trigonometric tables."""
    freq_count = head_dim // 2
    base = 10000.0
    inv_freq = 1.0 / (base ** (torch.arange(0, freq_count, device=device).float() / freq_count))
    omega = inv_freq

    num_offsets = len(offsets)
    cos_table = torch.zeros(num_offsets, freq_count, device=device, dtype=torch.float32)
    sin_table = torch.zeros(num_offsets, freq_count, device=device, dtype=torch.float32)

    for i, offset in enumerate(offsets):
        t = round_start + offset.item()
        angle = t * omega
        cos_table[i] = torch.cos(angle)
        sin_table[i] = torch.sin(angle)

    return cos_table, sin_table


def test_basic_functionality():
    """Quick smoke test."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping test")
        return

    print("Running basic functionality test...")

    batch_size, num_heads, seq_len, head_dim = 2, 4, 64, 128
    freq_count = head_dim // 2
    num_offsets = 8

    torch.manual_seed(42)
    K_rot = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device)
    q_mean_real = torch.randn(num_heads, freq_count, device=device)
    q_mean_imag = torch.randn(num_heads, freq_count, device=device)
    q_abs_mean = torch.abs(torch.randn(num_heads, freq_count, device=device))
    freq_scale_sq = torch.ones(num_heads, freq_count, device=device)

    # Build tables
    round_start = 1000
    offsets = torch.tensor([float(2**i) for i in range(num_offsets)], device=device)
    cos_table, sin_table = build_cos_sin_tables(round_start, offsets, head_dim, device)

    print(f"  Input shapes:")
    print(f"    K_rot: {K_rot.shape}")
    print(f"    q_mean_real/imag: {q_mean_real.shape}")
    print(f"    cos_table: {cos_table.shape}")

    # Run kernel
    try:
        scores = speckv_scoring(
            K_rot, q_mean_real, q_mean_imag, q_abs_mean, freq_scale_sq, cos_table, sin_table, "max"
        )
        print(f"  ✓ Kernel executed successfully")
    except Exception as e:
        print(f"  ✗ Kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Basic checks
    assert scores.shape == (batch_size, num_heads, seq_len), f"Wrong shape: {scores.shape}"
    assert not torch.isnan(scores).any(), "NaN values in output"
    assert not torch.isinf(scores).any(), "Inf values in output"

    print(f"  ✓ Output shape: {scores.shape}")
    print(f"  ✓ Score range: [{scores.min().item():.4f}, {scores.max().item():.4f}]")
    print()


def test_correctness():
    """Test against PyTorch reference."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        print("CUDA not available, skipping test")
        return

    print("Running correctness test...")

    batch_size, num_heads, seq_len, head_dim = 1, 2, 32, 64
    freq_count = head_dim // 2
    round_start = 100
    num_offsets = 4

    torch.manual_seed(42)
    K_rot = torch.randn(batch_size, num_heads, seq_len, head_dim, device=device, dtype=torch.float32)

    q_mean_real = torch.randn(num_heads, freq_count, device=device, dtype=torch.float32)
    q_mean_imag = torch.randn(num_heads, freq_count, device=device, dtype=torch.float32)
    q_mean_complex = torch.complex(q_mean_real, q_mean_imag)

    q_abs_mean = torch.abs(torch.randn(num_heads, freq_count, device=device, dtype=torch.float32))
    freq_scale_sq = torch.ones(num_heads, freq_count, device=device, dtype=torch.float32)

    offsets = torch.tensor([1.0, 2.0, 4.0, 8.0], device=device)

    # Build tables
    cos_table, sin_table = build_cos_sin_tables(round_start, offsets, head_dim, device)

    # Reference
    print("  Computing reference scores...")
    ref_scores = pytorch_reference_scoring(
        K_rot, q_mean_complex, q_abs_mean, freq_scale_sq, round_start, offsets, "max"
    )

    # Triton
    print("  Computing Triton scores...")
    try:
        triton_scores = speckv_scoring(
            K_rot, q_mean_real, q_mean_imag, q_abs_mean, freq_scale_sq, cos_table, sin_table, "max"
        )
    except Exception as e:
        print(f"  ✗ Kernel failed: {e}")
        import traceback
        traceback.print_exc()
        return

    # Compare
    max_diff = (ref_scores - triton_scores).abs().max().item()
    mean_diff = (ref_scores - triton_scores).abs().mean().item()

    print(f"  Max difference: {max_diff:.6e}")
    print(f"  Mean difference: {mean_diff:.6e}")

    tolerance_max = 1e-3
    tolerance_mean = 1e-4

    if max_diff < tolerance_max and mean_diff < tolerance_mean:
        print(f"  ✓ Correctness test PASSED")
    else:
        print(f"  ✗ Correctness test FAILED")
        print(f"    Expected max_diff < {tolerance_max}, mean_diff < {tolerance_mean}")

        # Debug: print some sample values
        print("\n  Sample values (first 5 tokens, first head):")
        print(f"    Reference: {ref_scores[0, 0, :5]}")
        print(f"    Triton:    {triton_scores[0, 0, :5]}")
        print(f"    Diff:      {(ref_scores - triton_scores)[0, 0, :5]}")

    print()


if __name__ == "__main__":
    print("=" * 60)
    print("Triton Scoring Kernel Tests")
    print("=" * 60)
    print()

    test_basic_functionality()
    test_correctness()

    print("=" * 60)
    print("All tests completed")
    print("=" * 60)
