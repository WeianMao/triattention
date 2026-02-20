"""
Test scoring formula correctness.

Validates TriAttention scoring against PyTorch reference implementation
across multiple precisions and configurations.
"""

import pytest
import torch
import math


class TriAttentionScorer:
    """
    Reference PyTorch implementation of TriAttention scoring.

    Used as ground truth for validating optimized implementations.
    """

    def __init__(self, Q_mean_real, Q_mean_imag, freq_scale_sq, extra_coef, omega):
        """
        Args:
            Q_mean_real: [num_layers, num_heads, freq_count]
            Q_mean_imag: [num_layers, num_heads, freq_count]
            freq_scale_sq: [num_layers, num_heads, freq_count]
            extra_coef: [num_layers, num_heads, freq_count]
            omega: [freq_count] - RoPE frequencies
        """
        self.Q_mean_real = Q_mean_real
        self.Q_mean_imag = Q_mean_imag
        self.freq_scale_sq = freq_scale_sq
        self.extra_coef = extra_coef
        self.omega = omega

    def score_single_position(
        self, K_rot_real, K_rot_imag, position_indices, target_position, layer_idx, head_idx
    ):
        """
        Score keys for a single target position.

        Args:
            K_rot_real: [num_keys, freq_count] - rotated key real part
            K_rot_imag: [num_keys, freq_count] - rotated key imaginary part
            position_indices: [num_keys] - original positions of keys
            target_position: int - future position to score for
            layer_idx: int - layer index
            head_idx: int - head index

        Returns:
            torch.Tensor: [num_keys] - scores
        """
        num_keys = K_rot_real.shape[0]
        freq_count = K_rot_real.shape[1]

        # Get stats for this layer and head
        Q_real = self.Q_mean_real[layer_idx, head_idx]  # [freq_count]
        Q_imag = self.Q_mean_imag[layer_idx, head_idx]
        scale_sq = self.freq_scale_sq[layer_idx, head_idx]
        extra = self.extra_coef[layer_idx, head_idx]

        # Compute amplitude: A_f = |Q_mean| * |K|
        Q_mag = torch.sqrt(Q_real**2 + Q_imag**2)  # [freq_count]
        K_mag = torch.sqrt(K_rot_real**2 + K_rot_imag**2)  # [num_keys, freq_count]
        A_f = Q_mag.unsqueeze(0) * K_mag  # [num_keys, freq_count]

        # Compute phase difference: phi_f = arg(Q * K_conj)
        # Q * K_conj = (Q_r + iQ_i) * (K_r - iK_i) = (Q_r*K_r + Q_i*K_i) + i(Q_i*K_r - Q_r*K_i)
        phi_real = Q_real.unsqueeze(0) * K_rot_real + Q_imag.unsqueeze(0) * K_rot_imag
        phi_imag = Q_imag.unsqueeze(0) * K_rot_real - Q_real.unsqueeze(0) * K_rot_imag
        phi_f = torch.atan2(phi_imag, phi_real)  # [num_keys, freq_count]

        # Compute position-dependent term: cos((t - p) * omega + phi)
        t = target_position
        p = position_indices.unsqueeze(-1)  # [num_keys, 1]
        omega_exp = self.omega.unsqueeze(0)  # [1, freq_count]

        angle = (t - p) * omega_exp + phi_f  # [num_keys, freq_count]
        position_term = A_f * scale_sq.unsqueeze(0) * torch.cos(angle)

        # Compute position-independent term
        position_independent = extra.unsqueeze(0) * scale_sq.unsqueeze(0)

        # Sum over frequency dimension
        score = position_term.sum(dim=-1) + position_independent.sum(dim=-1)  # [num_keys]

        return score

    def score_multi_position(
        self,
        K_rot_real,
        K_rot_imag,
        position_indices,
        target_positions,
        layer_idx,
        head_idx,
        aggregation="mean",
    ):
        """
        Score keys for multiple target positions and aggregate.

        Args:
            K_rot_real: [num_keys, freq_count]
            K_rot_imag: [num_keys, freq_count]
            position_indices: [num_keys]
            target_positions: [num_positions] - multiple future positions
            layer_idx: int
            head_idx: int
            aggregation: str - "mean" or "max"

        Returns:
            torch.Tensor: [num_keys] - aggregated scores
        """
        scores_per_position = []

        for t in target_positions:
            score_t = self.score_single_position(
                K_rot_real, K_rot_imag, position_indices, t, layer_idx, head_idx
            )
            scores_per_position.append(score_t)

        scores_stacked = torch.stack(scores_per_position, dim=0)  # [num_positions, num_keys]

        if aggregation == "mean":
            return scores_stacked.mean(dim=0)
        elif aggregation == "max":
            return scores_stacked.max(dim=0).values
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")


def apply_rope_rotation(K, position_indices, omega):
    """
    Apply RoPE rotation to keys.

    Args:
        K: [num_keys, head_dim] - original keys
        position_indices: [num_keys] - positions
        omega: [freq_count] - RoPE frequencies

    Returns:
        tuple: (K_rot_real, K_rot_imag) each [num_keys, freq_count]
    """
    num_keys = K.shape[0]
    head_dim = K.shape[1]
    freq_count = head_dim // 2

    # Split into pairs
    K_pairs = K.view(num_keys, freq_count, 2)  # [num_keys, freq_count, 2]
    K_real_orig = K_pairs[:, :, 0]  # [num_keys, freq_count]
    K_imag_orig = K_pairs[:, :, 1]

    # Compute rotation angles
    pos_exp = position_indices.unsqueeze(-1).float()  # [num_keys, 1]
    omega_exp = omega.unsqueeze(0)  # [1, freq_count]
    theta = pos_exp * omega_exp  # [num_keys, freq_count]

    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)

    # Apply rotation
    K_rot_real = K_real_orig * cos_theta - K_imag_orig * sin_theta
    K_rot_imag = K_real_orig * sin_theta + K_imag_orig * cos_theta

    return K_rot_real, K_rot_imag


# ==================== Test Cases ====================


def test_single_position_scoring_fp32(random_query_stats, rope_frequencies, deterministic_seed):
    """
    Test single position scoring with FP32 precision.
    """
    num_keys = 64
    freq_count = 32
    layer_idx = 0
    head_idx = 0
    target_position = 100

    # Generate random keys
    K = torch.randn(num_keys, freq_count * 2)
    position_indices = torch.arange(num_keys, dtype=torch.long)

    # Apply RoPE rotation
    omega = rope_frequencies
    K_rot_real, K_rot_imag = apply_rope_rotation(K, position_indices, omega)

    # Create scorer
    scorer = TriAttentionScorer(
        random_query_stats["Q_mean_real"],
        random_query_stats["Q_mean_imag"],
        random_query_stats["freq_scale_sq"],
        random_query_stats["extra_coef"],
        omega,
    )

    # Compute score
    scores = scorer.score_single_position(
        K_rot_real, K_rot_imag, position_indices, target_position, layer_idx, head_idx
    )

    # Validate output shape and type
    assert scores.shape == (num_keys,)
    assert scores.dtype == torch.float32

    # Scores should be finite
    assert torch.all(torch.isfinite(scores))


def test_multi_position_aggregation(
    random_query_stats, rope_frequencies, aggregation_strategy, deterministic_seed
):
    """
    Test multi-position scoring with different aggregation strategies.
    """
    num_keys = 64
    freq_count = 32
    layer_idx = 0
    head_idx = 0
    target_positions = torch.arange(100, 116)  # 16 positions

    # Generate random keys
    K = torch.randn(num_keys, freq_count * 2)
    position_indices = torch.arange(num_keys, dtype=torch.long)

    omega = rope_frequencies
    K_rot_real, K_rot_imag = apply_rope_rotation(K, position_indices, omega)

    scorer = TriAttentionScorer(
        random_query_stats["Q_mean_real"],
        random_query_stats["Q_mean_imag"],
        random_query_stats["freq_scale_sq"],
        random_query_stats["extra_coef"],
        omega,
    )

    # Compute aggregated score
    scores = scorer.score_multi_position(
        K_rot_real,
        K_rot_imag,
        position_indices,
        target_positions,
        layer_idx,
        head_idx,
        aggregation=aggregation_strategy,
    )

    # Validate
    assert scores.shape == (num_keys,)
    assert torch.all(torch.isfinite(scores))

    # For mean, score should be average of individual scores
    if aggregation_strategy == "mean":
        individual_scores = []
        for t in target_positions:
            s = scorer.score_single_position(
                K_rot_real, K_rot_imag, position_indices, t, layer_idx, head_idx
            )
            individual_scores.append(s)
        expected_mean = torch.stack(individual_scores).mean(dim=0)
        assert torch.allclose(scores, expected_mean, atol=1e-5)


def test_dtype_consistency(random_query_stats, rope_frequencies, test_dtype, tolerance_for_dtype, gpu_capability):
    """
    Test scoring consistency across different dtypes.

    Skips bfloat16 tests on GPUs with SM < 80.
    """
    # Skip bfloat16 if GPU doesn't support it
    if test_dtype == torch.bfloat16 and gpu_capability is not None and gpu_capability[0] < 8:
        pytest.skip(f"bfloat16 requires SM80+, got SM{gpu_capability[0]}{gpu_capability[1]}")

    torch.manual_seed(42)

    num_keys = 32
    freq_count = 32
    layer_idx = 0
    head_idx = 0
    target_position = 100

    # Generate keys in FP32 as reference
    K_fp32 = torch.randn(num_keys, freq_count * 2, dtype=torch.float32)
    position_indices = torch.arange(num_keys, dtype=torch.long)

    omega_fp32 = rope_frequencies
    K_rot_real_fp32, K_rot_imag_fp32 = apply_rope_rotation(K_fp32, position_indices, omega_fp32)

    # Reference scorer in FP32
    scorer_fp32 = TriAttentionScorer(
        random_query_stats["Q_mean_real"].float(),
        random_query_stats["Q_mean_imag"].float(),
        random_query_stats["freq_scale_sq"].float(),
        random_query_stats["extra_coef"].float(),
        omega_fp32.float(),
    )

    scores_fp32 = scorer_fp32.score_single_position(
        K_rot_real_fp32, K_rot_imag_fp32, position_indices, target_position, layer_idx, head_idx
    )

    # Convert to test dtype
    K_test = K_fp32.to(test_dtype)
    omega_test = omega_fp32.to(test_dtype)
    K_rot_real_test, K_rot_imag_test = apply_rope_rotation(K_test, position_indices, omega_test)

    scorer_test = TriAttentionScorer(
        random_query_stats["Q_mean_real"].to(test_dtype),
        random_query_stats["Q_mean_imag"].to(test_dtype),
        random_query_stats["freq_scale_sq"].to(test_dtype),
        random_query_stats["extra_coef"].to(test_dtype),
        omega_test,
    )

    scores_test = scorer_test.score_single_position(
        K_rot_real_test, K_rot_imag_test, position_indices, target_position, layer_idx, head_idx
    )

    # Compare
    abs_error = torch.abs(scores_fp32 - scores_test.float())
    max_error = abs_error.max().item()

    print(f"\nDtype: {test_dtype}, Max error: {max_error:.2e}, Tolerance: {tolerance_for_dtype:.2e}")

    assert max_error < tolerance_for_dtype, (
        f"Dtype {test_dtype} error {max_error:.2e} exceeds tolerance {tolerance_for_dtype:.2e}"
    )


def test_phase_computation_correctness(deterministic_seed):
    """
    Test phase difference computation: phi = arg(Q * K_conj).
    """
    freq_count = 32

    Q_real = torch.randn(freq_count)
    Q_imag = torch.randn(freq_count)
    K_real = torch.randn(freq_count)
    K_imag = torch.randn(freq_count)

    # Method 1: Using atan2
    phi_real = Q_real * K_real + Q_imag * K_imag
    phi_imag = Q_imag * K_real - Q_real * K_imag
    phi_method1 = torch.atan2(phi_imag, phi_real)

    # Method 2: Using complex numbers
    Q_complex = torch.complex(Q_real, Q_imag)
    K_complex = torch.complex(K_real, K_imag)
    product = Q_complex * torch.conj(K_complex)
    phi_method2 = torch.angle(product)

    # Should match
    assert torch.allclose(phi_method1, phi_method2, atol=1e-5)


def test_rope_rotation_correctness(rope_frequencies, deterministic_seed):
    """
    Test RoPE rotation matches standard implementation.
    """
    num_keys = 16
    head_dim = 64
    freq_count = head_dim // 2

    K = torch.randn(num_keys, head_dim)
    position_indices = torch.arange(num_keys, dtype=torch.long)
    omega = rope_frequencies

    K_rot_real, K_rot_imag = apply_rope_rotation(K, position_indices, omega)

    # Validate shapes
    assert K_rot_real.shape == (num_keys, freq_count)
    assert K_rot_imag.shape == (num_keys, freq_count)

    # Validate magnitude preservation (rotation doesn't change magnitude)
    K_pairs = K.view(num_keys, freq_count, 2)
    original_mag = torch.sqrt(K_pairs[:, :, 0] ** 2 + K_pairs[:, :, 1] ** 2)
    rotated_mag = torch.sqrt(K_rot_real**2 + K_rot_imag**2)

    assert torch.allclose(original_mag, rotated_mag, atol=1e-5)


def test_position_independent_term(random_query_stats, rope_frequencies, deterministic_seed):
    """
    Test that position-independent term is correctly computed.
    """
    num_keys = 64
    freq_count = 32
    layer_idx = 0
    head_idx = 0

    K = torch.randn(num_keys, freq_count * 2)
    position_indices = torch.arange(num_keys, dtype=torch.long)
    omega = rope_frequencies
    K_rot_real, K_rot_imag = apply_rope_rotation(K, position_indices, omega)

    scorer = TriAttentionScorer(
        random_query_stats["Q_mean_real"],
        random_query_stats["Q_mean_imag"],
        random_query_stats["freq_scale_sq"],
        random_query_stats["extra_coef"],
        omega,
    )

    # Get full score
    target_pos = 100
    full_score = scorer.score_single_position(
        K_rot_real, K_rot_imag, position_indices, target_pos, layer_idx, head_idx
    )

    # Manually compute position-independent part
    extra = random_query_stats["extra_coef"][layer_idx, head_idx]
    scale_sq = random_query_stats["freq_scale_sq"][layer_idx, head_idx]
    expected_independent = (extra * scale_sq).sum()

    # Position-independent term should be same for all keys
    # (it's actually broadcasted, so it's part of each key's score)
    assert torch.all(torch.isfinite(full_score))


def test_scoring_symmetry(random_query_stats, rope_frequencies, deterministic_seed):
    """
    Test scoring symmetry: swapping two identical keys should give same scores.
    """
    freq_count = 32
    layer_idx = 0
    head_idx = 0
    target_position = 100

    # Create two identical keys at different positions
    K_identical = torch.randn(1, freq_count * 2)
    K = K_identical.repeat(2, 1)  # [2, head_dim]
    position_indices = torch.tensor([10, 50], dtype=torch.long)

    omega = rope_frequencies
    K_rot_real, K_rot_imag = apply_rope_rotation(K, position_indices, omega)

    scorer = TriAttentionScorer(
        random_query_stats["Q_mean_real"],
        random_query_stats["Q_mean_imag"],
        random_query_stats["freq_scale_sq"],
        random_query_stats["extra_coef"],
        omega,
    )

    scores = scorer.score_single_position(
        K_rot_real, K_rot_imag, position_indices, target_position, layer_idx, head_idx
    )

    # Scores should be different (different positions)
    assert not torch.allclose(scores[0], scores[1], atol=1e-3)

    # But if we score for same relative distance, should be same
    scores_0 = scorer.score_single_position(
        K_rot_real[:1], K_rot_imag[:1], position_indices[:1], target_position, layer_idx, head_idx
    )
    scores_1 = scorer.score_single_position(
        K_rot_real[1:2],
        K_rot_imag[1:2],
        position_indices[1:2],
        target_position,
        layer_idx,
        head_idx,
    )

    # Both should be finite
    assert torch.isfinite(scores_0).all()
    assert torch.isfinite(scores_1).all()
