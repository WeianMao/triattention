"""
Test Triton kernel vs PyTorch reference implementation equivalence.

Validates that the optimized Triton kernel produces numerically equivalent
results to the PyTorch reference implementation across various configurations.
"""

import pytest
import torch


@pytest.fixture
def test_configs():
    """Generate test configurations with different dimensions."""
    return [
        {"batch": 1, "num_heads": 4, "seq_len": 64, "freq_count": 32},
        {"batch": 2, "num_heads": 8, "seq_len": 128, "freq_count": 64},
        {"batch": 1, "num_heads": 16, "seq_len": 256, "freq_count": 32},
        {"batch": 4, "num_heads": 4, "seq_len": 32, "freq_count": 16},
    ]


@pytest.fixture
def num_offsets_configs():
    """Different offset configurations."""
    return [1, 4, 8, 16]


def generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets, dtype=torch.float32):
    """
    Generate random test inputs for scoring.

    Args:
        batch: Batch size
        num_heads: Number of attention heads
        seq_len: Sequence length
        freq_count: Number of frequency bins (head_dim // 2)
        num_offsets: Number of scoring offsets
        dtype: Data type for inputs

    Returns:
        dict: Test inputs with all required tensors
    """
    head_dim = freq_count * 2

    # K_rot: Rotated keys [batch, num_heads, seq_len, head_dim]
    K_rot = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)

    # Q statistics [num_heads, freq_count]
    q_mean_real = torch.randn(num_heads, freq_count, dtype=dtype)
    q_mean_imag = torch.randn(num_heads, freq_count, dtype=dtype)
    q_abs_mean = torch.rand(num_heads, freq_count, dtype=dtype) * 2.0 + 0.5
    freq_scale_sq = torch.rand(num_heads, freq_count, dtype=dtype) * 2.0 + 0.5

    # Position indices [seq_len]
    position_indices = torch.arange(seq_len, dtype=torch.long)

    # RoPE frequencies [freq_count]
    base = 10000.0
    omega = 1.0 / (base ** (torch.arange(0, freq_count, dtype=torch.float32) / freq_count))

    # Offsets [num_offsets]
    offsets = torch.tensor([float(2**i) for i in range(num_offsets)], dtype=torch.float32)

    # Round start
    round_start = 100

    return {
        "K_rot": K_rot,
        "position_indices": position_indices,
        "q_mean_real": q_mean_real,
        "q_mean_imag": q_mean_imag,
        "q_abs_mean": q_abs_mean,
        "freq_scale_sq": freq_scale_sq,
        "omega": omega,
        "offsets": offsets,
        "round_start": round_start,
    }


def compute_pytorch_reference(inputs, aggregation="max"):
    """
    Compute scores using PyTorch reference implementation.

    This implements the exact same formula as the Triton kernel but in pure PyTorch.

    Args:
        inputs: Dictionary of input tensors
        aggregation: "max" or "mean"

    Returns:
        torch.Tensor: Scores [batch, num_heads, seq_len]
    """
    K_rot = inputs["K_rot"]
    position_indices = inputs["position_indices"]
    q_mean_real = inputs["q_mean_real"]
    q_mean_imag = inputs["q_mean_imag"]
    q_abs_mean = inputs["q_abs_mean"]
    freq_scale_sq = inputs["freq_scale_sq"]
    omega = inputs["omega"]
    offsets = inputs["offsets"]
    round_start = inputs["round_start"]

    batch, num_heads, seq_len, head_dim = K_rot.shape
    freq_count = head_dim // 2
    num_offsets = offsets.shape[0]

    # Split K_rot into complex pairs
    k_pairs = K_rot.view(batch, num_heads, seq_len, freq_count, 2)
    k_r = k_pairs[..., 0]  # [batch, num_heads, seq_len, freq_count]
    k_i = k_pairs[..., 1]

    # Compute |K_rot|
    k_abs = torch.sqrt(k_r ** 2 + k_i ** 2 + 1e-8)

    # Compute |Q_mean_complex|
    q_mean_abs = torch.sqrt(q_mean_real ** 2 + q_mean_imag ** 2 + 1e-8)

    # Expand Q statistics to match K dimensions
    # [num_heads, freq_count] -> [1, num_heads, 1, freq_count]
    q_r_exp = q_mean_real.unsqueeze(0).unsqueeze(2)
    q_i_exp = q_mean_imag.unsqueeze(0).unsqueeze(2)
    q_abs_exp = q_abs_mean.unsqueeze(0).unsqueeze(2)
    q_mean_abs_exp = q_mean_abs.unsqueeze(0).unsqueeze(2)
    freq_scale_exp = freq_scale_sq.unsqueeze(0).unsqueeze(2)

    # Complex product: Q_mean * conj(K_rot)
    prod_real = q_r_exp * k_r + q_i_exp * k_i  # [batch, num_heads, seq_len, freq_count]
    prod_imag = q_i_exp * k_r - q_r_exp * k_i

    # Coefficients for trigonometric expansion
    A_coef = freq_scale_exp * prod_real
    B_coef = freq_scale_exp * prod_imag

    # Additive term (position-independent)
    extra_coef = (q_abs_exp - q_mean_abs_exp) * k_abs * freq_scale_exp
    extra_sum = extra_coef.sum(dim=-1)  # [batch, num_heads, seq_len]

    # Handle position_indices shape
    if position_indices.ndim == 1:
        positions = position_indices[None, None, :, None]  # [1, 1, seq_len, 1]
    else:
        positions = position_indices[:, None, :, None]  # [batch, 1, seq_len, 1]

    # Compute scores for each offset
    scores_per_offset = []

    for offset in offsets:
        t = round_start + offset.item()

        # Phase only depends on query position t, NOT key position
        # (Key position is "baked in" to K_rot via RoPE)
        # See docs/RKV_EQUIVALENCE_FIX.md for derivation
        phase = t * omega  # [freq_count]

        # Compute cos and sin (broadcast to all tokens)
        cos_vals = torch.cos(phase)  # [freq_count]
        sin_vals = torch.sin(phase)  # [freq_count]

        # Base scores: A * cos - B * sin
        base_scores = (A_coef * cos_vals - B_coef * sin_vals).sum(dim=-1)

        # Combined score
        combined = base_scores + extra_sum  # [batch, num_heads, seq_len]
        scores_per_offset.append(combined)

    # Aggregate across offsets
    scores_stacked = torch.stack(scores_per_offset, dim=0)  # [num_offsets, batch, num_heads, seq_len]

    if aggregation == "max":
        scores = scores_stacked.max(dim=0).values
    else:  # mean
        scores = scores_stacked.mean(dim=0)

    return scores


# ==================== Test Cases ====================


@pytest.mark.parametrize("aggregation", ["mean", "max"])
def test_basic_equivalence_fp32(cuda_only, aggregation, deterministic_seed):
    """
    Test basic Triton-PyTorch equivalence with FP32 precision.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count = 2, 4, 64, 32
    num_offsets = 8

    # Generate test inputs
    inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets, dtype=torch.float32)
    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Compute PyTorch reference
    scores_pytorch = compute_pytorch_reference(inputs, aggregation=aggregation)

    # Compute Triton
    scores_triton = speckv_scoring(**inputs_cuda, aggregation=aggregation)

    # Compare
    scores_triton_cpu = scores_triton.cpu()

    rtol, atol = 1e-4, 1e-4
    assert torch.allclose(scores_pytorch, scores_triton_cpu, rtol=rtol, atol=atol), (
        f"Triton-PyTorch mismatch for aggregation={aggregation}\n"
        f"Max abs error: {(scores_pytorch - scores_triton_cpu).abs().max().item():.2e}\n"
        f"Mean abs error: {(scores_pytorch - scores_triton_cpu).abs().mean().item():.2e}"
    )


@pytest.mark.parametrize("aggregation", ["mean", "max"])
def test_basic_equivalence_bf16(requires_sm80, aggregation, deterministic_seed):
    """
    Test Triton-PyTorch equivalence with BF16 precision.

    Requires SM80+ GPU for native bfloat16 support.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count = 2, 4, 64, 32
    num_offsets = 8

    # Generate test inputs in FP32 then convert
    inputs_fp32 = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets, dtype=torch.float32)
    # Convert tensors to BF16, keep scalars as-is
    inputs_bf16 = {k: v.bfloat16() if isinstance(v, torch.Tensor) else v for k, v in inputs_fp32.items()}
    inputs_bf16_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs_bf16.items()}

    # Compute PyTorch reference in BF16
    scores_pytorch = compute_pytorch_reference(inputs_bf16, aggregation=aggregation)

    # Compute Triton in BF16
    scores_triton = speckv_scoring(**inputs_bf16_cuda, aggregation=aggregation)
    scores_triton_cpu = scores_triton.cpu()

    # BF16 has lower precision
    rtol, atol = 5e-2, 5e-2

    # Convert to FP32 for comparison
    scores_pytorch_fp32 = scores_pytorch.float()
    scores_triton_fp32 = scores_triton_cpu.float()

    assert torch.allclose(scores_pytorch_fp32, scores_triton_fp32, rtol=rtol, atol=atol), (
        f"Triton-PyTorch mismatch for aggregation={aggregation} (BF16)\n"
        f"Max abs error: {(scores_pytorch_fp32 - scores_triton_fp32).abs().max().item():.2e}\n"
        f"Mean abs error: {(scores_pytorch_fp32 - scores_triton_fp32).abs().mean().item():.2e}"
    )


def test_different_batch_sizes(cuda_only, deterministic_seed):
    """
    Test equivalence across different batch sizes.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    num_heads, seq_len, freq_count, num_offsets = 4, 64, 32, 8

    for batch in [1, 2, 4, 8]:
        inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)
        inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        scores_pytorch = compute_pytorch_reference(inputs, aggregation="mean")
        scores_triton = speckv_scoring(**inputs_cuda, aggregation="mean").cpu()

        assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-4, atol=1e-4), (
            f"Batch size {batch} failed"
        )


def test_different_seq_lengths(cuda_only, deterministic_seed):
    """
    Test equivalence across different sequence lengths.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, freq_count, num_offsets = 2, 4, 32, 8

    for seq_len in [16, 32, 64, 128, 256]:
        inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)
        inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        scores_pytorch = compute_pytorch_reference(inputs, aggregation="max")
        scores_triton = speckv_scoring(**inputs_cuda, aggregation="max").cpu()

        assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-4, atol=1e-4), (
            f"Seq length {seq_len} failed"
        )


def test_different_num_heads(cuda_only, deterministic_seed):
    """
    Test equivalence across different number of heads.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, seq_len, freq_count, num_offsets = 2, 64, 32, 8

    for num_heads in [1, 4, 8, 16, 32]:
        inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)
        inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        scores_pytorch = compute_pytorch_reference(inputs, aggregation="mean")
        scores_triton = speckv_scoring(**inputs_cuda, aggregation="mean").cpu()

        assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-4, atol=1e-4), (
            f"Num heads {num_heads} failed"
        )


def test_different_num_offsets(cuda_only, deterministic_seed):
    """
    Test equivalence with different numbers of scoring offsets.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count = 2, 4, 64, 32

    for num_offsets in [1, 2, 4, 8, 16, 32]:
        inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)
        inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

        for aggregation in ["mean", "max"]:
            scores_pytorch = compute_pytorch_reference(inputs, aggregation=aggregation)
            scores_triton = speckv_scoring(**inputs_cuda, aggregation=aggregation).cpu()

            assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-4, atol=1e-4), (
                f"Num offsets {num_offsets}, aggregation {aggregation} failed"
            )


def test_edge_case_single_token(cuda_only, deterministic_seed):
    """
    Test edge case: single token sequence.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count, num_offsets = 2, 4, 1, 32, 8

    inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)
    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    scores_pytorch = compute_pytorch_reference(inputs, aggregation="mean")
    scores_triton = speckv_scoring(**inputs_cuda, aggregation="mean").cpu()

    assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-4, atol=1e-4)


def test_edge_case_very_long_sequence(cuda_only, deterministic_seed):
    """
    Test edge case: very long sequence (>512 tokens).
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count, num_offsets = 1, 4, 1024, 32, 8

    inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)
    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    scores_pytorch = compute_pytorch_reference(inputs, aggregation="max")
    scores_triton = speckv_scoring(**inputs_cuda, aggregation="max").cpu()

    assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-4, atol=1e-4)


def test_numerical_stability_zero_inputs(cuda_only):
    """
    Test numerical stability with zero inputs.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count, num_offsets = 2, 4, 64, 32, 8
    head_dim = freq_count * 2

    # All zeros
    inputs = {
        "K_rot": torch.zeros(batch, num_heads, seq_len, head_dim),
        "position_indices": torch.arange(seq_len, dtype=torch.long),
        "q_mean_real": torch.zeros(num_heads, freq_count),
        "q_mean_imag": torch.zeros(num_heads, freq_count),
        "q_abs_mean": torch.zeros(num_heads, freq_count),
        "freq_scale_sq": torch.zeros(num_heads, freq_count),
        "omega": torch.zeros(freq_count),
        "offsets": torch.zeros(num_offsets),
        "round_start": 100,
    }
    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    scores_pytorch = compute_pytorch_reference(inputs, aggregation="mean")
    scores_triton = speckv_scoring(**inputs_cuda, aggregation="mean").cpu()

    # Both should produce zeros
    assert torch.allclose(scores_pytorch, torch.zeros_like(scores_pytorch), atol=1e-5)
    assert torch.allclose(scores_triton, torch.zeros_like(scores_triton), atol=1e-5)


def test_numerical_stability_large_values(cuda_only, deterministic_seed):
    """
    Test numerical stability with large input values.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count, num_offsets = 2, 4, 64, 32, 8

    # Generate inputs with large magnitudes
    inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)

    # Scale up by 100x (only scale tensors, not scalars)
    for key in ["K_rot", "q_mean_real", "q_mean_imag", "q_abs_mean"]:
        inputs[key] = inputs[key] * 100.0

    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    scores_pytorch = compute_pytorch_reference(inputs, aggregation="mean")
    scores_triton = speckv_scoring(**inputs_cuda, aggregation="mean").cpu()

    # Should still be finite
    assert torch.all(torch.isfinite(scores_pytorch))
    assert torch.all(torch.isfinite(scores_triton))

    # Relative tolerance should still hold
    assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-3, atol=1e-2)


def test_reproducibility(cuda_only):
    """
    Test that Triton kernel produces reproducible results.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count, num_offsets = 2, 4, 64, 32, 8

    # Set deterministic seed
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)
    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    # Run multiple times
    scores_1 = speckv_scoring(**inputs_cuda, aggregation="mean")
    scores_2 = speckv_scoring(**inputs_cuda, aggregation="mean")
    scores_3 = speckv_scoring(**inputs_cuda, aggregation="mean")

    # Should be identical
    assert torch.equal(scores_1, scores_2)
    assert torch.equal(scores_2, scores_3)


def test_aggregation_mean_vs_max(cuda_only, deterministic_seed):
    """
    Test that mean and max aggregation produce different but valid results.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count, num_offsets = 2, 4, 64, 32, 8

    inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets)
    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    scores_mean = speckv_scoring(**inputs_cuda, aggregation="mean")
    scores_max = speckv_scoring(**inputs_cuda, aggregation="max")

    # Both should be valid (finite)
    assert torch.all(torch.isfinite(scores_mean))
    assert torch.all(torch.isfinite(scores_max))

    # Max should be >= mean (in general, with random inputs)
    # This is a statistical property, not guaranteed for all cases
    # So we just check they are different
    assert not torch.equal(scores_mean, scores_max)


def test_comprehensive_configurations(cuda_only, test_configs, num_offsets_configs):
    """
    Comprehensive test across multiple configuration combinations.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    torch.manual_seed(42)

    for config in test_configs:
        for num_offsets in num_offsets_configs:
            inputs = generate_test_inputs(
                config["batch"],
                config["num_heads"],
                config["seq_len"],
                config["freq_count"],
                num_offsets,
            )
            inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

            for aggregation in ["mean", "max"]:
                scores_pytorch = compute_pytorch_reference(inputs, aggregation=aggregation)
                scores_triton = speckv_scoring(**inputs_cuda, aggregation=aggregation).cpu()

                assert torch.allclose(scores_pytorch, scores_triton, rtol=1e-4, atol=1e-4), (
                    f"Failed for config {config}, num_offsets={num_offsets}, aggregation={aggregation}\n"
                    f"Max error: {(scores_pytorch - scores_triton).abs().max().item():.2e}"
                )


def test_dtype_promotion_consistency(requires_sm80, deterministic_seed):
    """
    Test that mixed precision inputs are handled consistently.

    Requires SM80+ GPU for bfloat16 support.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    batch, num_heads, seq_len, freq_count, num_offsets = 2, 4, 64, 32, 8

    # Generate FP32 inputs
    inputs_fp32 = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets, dtype=torch.float32)

    # Convert K_rot to BF16, keep stats in FP32
    inputs_mixed = inputs_fp32.copy()
    inputs_mixed["K_rot"] = inputs_fp32["K_rot"].bfloat16()
    inputs_mixed_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs_mixed.items()}

    # Should not crash and produce finite results
    scores_triton = speckv_scoring(**inputs_mixed_cuda, aggregation="mean")

    assert torch.all(torch.isfinite(scores_triton))
    assert scores_triton.shape == (batch, num_heads, seq_len)
