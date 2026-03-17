"""
Test TriAttention scoring equivalence against R-KV reference implementation.

This test validates that TriAttention produces equivalent compression results
to the original R-KV SpeckV implementation by comparing:
1. Score computation (position-dependent and position-independent terms)
2. TopK token selection
3. Compression results with identical inputs
"""

import pytest
import torch
import sys
import os
from pathlib import Path

# Add R-KV to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../R-KV"))

try:
    from weian_development.speckv.round_pruning_utils import (
        score_keys_for_round,
        compute_frequency_statistics_from_means,
        to_complex_pairs,
        compute_frequency_scaling,
        build_rotary,
    )
    RKV_AVAILABLE = True
except ImportError:
    RKV_AVAILABLE = False


@pytest.fixture
def sample_model_config():
    """Create a minimal model config for testing."""
    from transformers import AutoConfig

    # Use a real Qwen config for testing
    config = AutoConfig.from_pretrained(
        "Qwen/Qwen2.5-0.5B-Instruct",
        trust_remote_code=True
    )
    return config


@pytest.fixture
def rkv_scoring_inputs(deterministic_seed, cuda_only):
    """
    Generate test inputs compatible with both R-KV and TriAttention scoring.

    Returns a dict with:
    - Q statistics (q_mean_complex, q_abs_mean)
    - K unrotated keys
    - K rotated keys (for TriAttention)
    - Position information
    - RoPE parameters
    """
    batch_size = 1
    num_heads = 4
    seq_len = 100
    head_dim = 64
    freq_count = head_dim // 2

    device = torch.device("cuda")
    dtype = torch.float32

    # Generate Q statistics (these would come from stats file in production)
    q_mean_real = torch.randn(num_heads, freq_count, dtype=dtype, device=device)
    q_mean_imag = torch.randn(num_heads, freq_count, dtype=dtype, device=device)
    q_mean_complex = torch.complex(q_mean_real, q_mean_imag)
    q_abs_mean = torch.abs(q_mean_complex) + torch.rand(num_heads, freq_count, device=device) * 0.5

    # Generate unrotated keys (for R-KV)
    k_unrot = torch.randn(seq_len, head_dim, dtype=dtype, device=device)

    # Build RoPE embeddings to get rotated keys
    from transformers import AutoConfig
    config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
    config.num_attention_heads = num_heads
    config.num_key_value_heads = num_heads
    config.hidden_size = num_heads * head_dim

    rotary = build_rotary(device, Path("Qwen/Qwen2.5-0.5B-Instruct"), dtype, config=config)
    rope_style = getattr(rotary, "_rope_style", "half")

    # Get cos/sin tables for positions
    position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
    dummy = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
    cos_table, sin_table = rotary(dummy, position_ids)
    cos_table = cos_table[0]  # [seq_len, head_dim]
    sin_table = sin_table[0]  # [seq_len, head_dim]

    # Apply RoPE to get K_rot
    k_unrot_expanded = k_unrot.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]
    cos_expanded = cos_table.unsqueeze(0).unsqueeze(0)
    sin_expanded = sin_table.unsqueeze(0).unsqueeze(0)

    # RoPE rotation (half style)
    if rope_style == "half":
        k1 = k_unrot_expanded[..., :head_dim//2]
        k2 = k_unrot_expanded[..., head_dim//2:]
        k_rot = torch.cat([
            k1 * cos_expanded[..., :head_dim//2] - k2 * sin_expanded[..., :head_dim//2],
            k2 * cos_expanded[..., head_dim//2:] + k1 * sin_expanded[..., head_dim//2:]
        ], dim=-1)
    else:
        raise NotImplementedError(f"RoPE style {rope_style} not implemented")

    k_rot = k_rot[0, 0]  # [seq_len, head_dim]

    # Expand to batch and num_heads for TriAttention (need contiguous for Triton kernel)
    k_rot_batch = k_rot.unsqueeze(0).unsqueeze(0).expand(batch_size, num_heads, seq_len, head_dim).contiguous()

    # Frequency scaling - expand to [num_heads, freq_count] for multi-head testing
    freq_scale = compute_frequency_scaling(rotary, head_dim, dtype, device)
    freq_scale_sq = freq_scale.pow(2).unsqueeze(0).expand(num_heads, freq_count).contiguous()

    # RoPE frequencies
    inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float32)
    omega = inv_freq[:freq_count]

    # Offsets for scoring
    num_offsets = 8
    offsets = torch.tensor([float(2**i) for i in range(num_offsets)], dtype=torch.float32, device=device)

    # Position indices
    position_indices = torch.arange(seq_len, dtype=torch.long, device=device)
    round_start = seq_len  # Pretend we're at position seq_len

    return {
        # Q statistics
        "q_mean_complex": q_mean_complex,
        "q_mean_real": q_mean_real,
        "q_mean_imag": q_mean_imag,
        "q_abs_mean": q_abs_mean,

        # K tensors
        "k_unrot": k_unrot,
        "k_rot": k_rot,
        "k_rot_batch": k_rot_batch,

        # Position info
        "position_indices": position_indices,
        "round_start": round_start,

        # RoPE parameters
        "omega": omega,
        "freq_scale_sq": freq_scale_sq,
        "offsets": offsets,
        "rope_style": rope_style,

        # Dimensions
        "batch_size": batch_size,
        "num_heads": num_heads,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "freq_count": freq_count,
    }


@pytest.mark.skipif(not RKV_AVAILABLE, reason="R-KV not available")
def test_rkv_triattention_score_equivalence_single_head(rkv_scoring_inputs):
    """
    Test that TriAttention scoring matches R-KV scoring for a single head.

    This validates the core scoring formula implementation.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    inputs = rkv_scoring_inputs

    # Test for head 0
    head_idx = 0

    # Compute R-KV scores
    # First compute frequency statistics
    amp, phi, extra = compute_frequency_statistics_from_means(
        q_mean_complex=inputs["q_mean_complex"][head_idx],  # [freq_count]
        q_abs_mean=inputs["q_abs_mean"][head_idx],  # [freq_count]
        k_unrot=inputs["k_unrot"],  # [seq_len, head_dim]
        style=inputs["rope_style"],
        disable_mlr=False,
    )
    # amp: [seq_len, freq_count] (unsqueeze(0) is applied inside the function)
    # phi: [seq_len, freq_count]
    # extra: [seq_len, freq_count]

    # Score keys using R-KV function
    rkv_scores = score_keys_for_round(
        key_indices=inputs["position_indices"],
        round_start=inputs["round_start"],
        amp=amp,
        phi=phi,
        omega=inputs["omega"],
        extra=extra,
        offsets=inputs["offsets"],
        aggregation="mean",
        freq_scale_sq=inputs["freq_scale_sq"][head_idx],
        disable_top_n_high_freq=0,
        simulate_bug_phase_offset=0,
        disable_trig=False,
    )  # [seq_len]

    # Compute TriAttention scores
    triattention_scores = speckv_scoring(
        K_rot=inputs["k_rot_batch"][:, head_idx:head_idx+1, :, :],  # [batch, 1, seq_len, head_dim]
        position_indices=inputs["position_indices"],
        q_mean_real=inputs["q_mean_real"][head_idx:head_idx+1],  # [1, freq_count]
        q_mean_imag=inputs["q_mean_imag"][head_idx:head_idx+1],  # [1, freq_count]
        q_abs_mean=inputs["q_abs_mean"][head_idx:head_idx+1],  # [1, freq_count]
        freq_scale_sq=inputs["freq_scale_sq"][head_idx:head_idx+1],  # [1, freq_count]
        omega=inputs["omega"],
        offsets=inputs["offsets"],
        round_start=inputs["round_start"],
        aggregation="mean",
        rope_style=inputs["rope_style"],  # Use same RoPE style as R-KV
    )  # [batch, 1, seq_len]

    triattention_scores = triattention_scores[0, 0]  # [seq_len]

    # Compare scores
    max_abs_error = (rkv_scores - triattention_scores).abs().max().item()
    mean_abs_error = (rkv_scores - triattention_scores).abs().mean().item()
    correlation = torch.corrcoef(torch.stack([rkv_scores, triattention_scores]))[0, 1].item()

    print(f"\n=== R-KV vs TriAttention Score Comparison (Single Head) ===")
    print(f"Max absolute error: {max_abs_error:.6e}")
    print(f"Mean absolute error: {mean_abs_error:.6e}")
    print(f"Correlation: {correlation:.6f}")
    print(f"R-KV scores range: [{rkv_scores.min().item():.4f}, {rkv_scores.max().item():.4f}]")
    print(f"TriAttention scores range: [{triattention_scores.min().item():.4f}, {triattention_scores.max().item():.4f}]")

    # Assert strong equivalence
    assert correlation > 0.99, f"Correlation too low: {correlation}"
    assert max_abs_error < 1e-3, f"Max error too high: {max_abs_error}"


@pytest.mark.skipif(not RKV_AVAILABLE, reason="R-KV not available")
@pytest.mark.parametrize("aggregation", ["mean", "max"])
def test_rkv_triattention_score_equivalence_all_heads(rkv_scoring_inputs, aggregation):
    """
    Test that TriAttention scoring matches R-KV scoring across all heads.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    inputs = rkv_scoring_inputs
    num_heads = inputs["num_heads"]

    all_rkv_scores = []

    # Compute R-KV scores for each head
    for head_idx in range(num_heads):
        amp, phi, extra = compute_frequency_statistics_from_means(
            q_mean_complex=inputs["q_mean_complex"][head_idx],
            q_abs_mean=inputs["q_abs_mean"][head_idx],
            k_unrot=inputs["k_unrot"],
            style=inputs["rope_style"],
            disable_mlr=False,
        )

        rkv_scores = score_keys_for_round(
            key_indices=inputs["position_indices"],
            round_start=inputs["round_start"],
            amp=amp,
            phi=phi,
            omega=inputs["omega"],
            extra=extra,
            offsets=inputs["offsets"],
            aggregation=aggregation,
            freq_scale_sq=inputs["freq_scale_sq"][head_idx],
            disable_top_n_high_freq=0,
            simulate_bug_phase_offset=0,
            disable_trig=False,
        )
        all_rkv_scores.append(rkv_scores)

    rkv_scores_all = torch.stack(all_rkv_scores, dim=0)  # [num_heads, seq_len]

    # Compute TriAttention scores (all heads at once)
    triattention_scores = speckv_scoring(
        K_rot=inputs["k_rot_batch"],  # [batch, num_heads, seq_len, head_dim]
        position_indices=inputs["position_indices"],
        q_mean_real=inputs["q_mean_real"],  # [num_heads, freq_count]
        q_mean_imag=inputs["q_mean_imag"],
        q_abs_mean=inputs["q_abs_mean"],
        freq_scale_sq=inputs["freq_scale_sq"],
        omega=inputs["omega"],
        offsets=inputs["offsets"],
        round_start=inputs["round_start"],
        aggregation=aggregation,
        rope_style=inputs["rope_style"],  # Use same RoPE style as R-KV
    )  # [batch, num_heads, seq_len]

    triattention_scores = triattention_scores[0]  # [num_heads, seq_len]

    # Compare scores per head
    max_abs_errors = []
    correlations = []

    for head_idx in range(num_heads):
        rkv = rkv_scores_all[head_idx]
        tri = triattention_scores[head_idx]

        max_err = (rkv - tri).abs().max().item()
        corr = torch.corrcoef(torch.stack([rkv, tri]))[0, 1].item()

        max_abs_errors.append(max_err)
        correlations.append(corr)

    print(f"\n=== R-KV vs TriAttention Score Comparison (All Heads, aggregation={aggregation}) ===")
    print(f"Max absolute error across heads: {max(max_abs_errors):.6e}")
    print(f"Mean absolute error across heads: {sum(max_abs_errors) / len(max_abs_errors):.6e}")
    print(f"Min correlation: {min(correlations):.6f}")
    print(f"Mean correlation: {sum(correlations) / len(correlations):.6f}")

    # Assert strong equivalence for all heads
    assert all(c > 0.99 for c in correlations), f"Some correlations too low: {correlations}"
    assert max(max_abs_errors) < 1e-3, f"Some errors too high: {max_abs_errors}"


@pytest.mark.skipif(not RKV_AVAILABLE, reason="R-KV not available")
def test_rkv_triattention_topk_equivalence(rkv_scoring_inputs):
    """
    Test that TriAttention and R-KV select the same top-k tokens.

    This is the most critical test - we want to ensure the same tokens
    are selected for compression.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    inputs = rkv_scoring_inputs
    num_heads = inputs["num_heads"]
    seq_len = inputs["seq_len"]
    budget = seq_len // 2  # Keep half the tokens

    all_rkv_indices = []

    # Compute R-KV TopK for each head
    for head_idx in range(num_heads):
        amp, phi, extra = compute_frequency_statistics_from_means(
            q_mean_complex=inputs["q_mean_complex"][head_idx],
            q_abs_mean=inputs["q_abs_mean"][head_idx],
            k_unrot=inputs["k_unrot"],
            style=inputs["rope_style"],
            disable_mlr=False,
        )

        rkv_scores = score_keys_for_round(
            key_indices=inputs["position_indices"],
            round_start=inputs["round_start"],
            amp=amp,
            phi=phi,
            omega=inputs["omega"],
            extra=extra,
            offsets=inputs["offsets"],
            aggregation="mean",
            freq_scale_sq=inputs["freq_scale_sq"][head_idx],
            disable_top_n_high_freq=0,
            simulate_bug_phase_offset=0,
            disable_trig=False,
        )

        # Get top-k indices
        _, rkv_indices = rkv_scores.topk(budget, dim=-1)
        all_rkv_indices.append(rkv_indices)

    rkv_topk_indices = torch.stack(all_rkv_indices, dim=0)  # [num_heads, budget]

    # Compute TriAttention TopK
    triattention_scores = speckv_scoring(
        K_rot=inputs["k_rot_batch"],
        position_indices=inputs["position_indices"],
        q_mean_real=inputs["q_mean_real"],
        q_mean_imag=inputs["q_mean_imag"],
        q_abs_mean=inputs["q_abs_mean"],
        freq_scale_sq=inputs["freq_scale_sq"],
        omega=inputs["omega"],
        offsets=inputs["offsets"],
        round_start=inputs["round_start"],
        aggregation="mean",
        rope_style=inputs["rope_style"],  # Use same RoPE style as R-KV
    )[0]  # [num_heads, seq_len]

    _, triattention_topk_indices = triattention_scores.topk(budget, dim=-1)  # [num_heads, budget]

    # Compare TopK indices per head
    overlap_ratios = []

    for head_idx in range(num_heads):
        rkv_set = set(rkv_topk_indices[head_idx].cpu().tolist())
        tri_set = set(triattention_topk_indices[head_idx].cpu().tolist())

        overlap = len(rkv_set & tri_set)
        overlap_ratio = overlap / budget
        overlap_ratios.append(overlap_ratio)

    print(f"\n=== R-KV vs TriAttention TopK Selection Comparison ===")
    print(f"Budget: {budget}/{seq_len}")
    print(f"TopK overlap per head: {overlap_ratios}")
    print(f"Min overlap: {min(overlap_ratios):.2%}")
    print(f"Mean overlap: {sum(overlap_ratios) / len(overlap_ratios):.2%}")

    # Assert high overlap (should be nearly 100% with identical inputs)
    assert all(r > 0.95 for r in overlap_ratios), f"TopK overlap too low: {overlap_ratios}"


@pytest.mark.skipif(not RKV_AVAILABLE, reason="R-KV not available")
def test_rkv_triattention_position_handling(rkv_scoring_inputs):
    """
    Test that TriAttention correctly handles out-of-order positions.

    This validates the position-aware phase correction in TriAttention.
    """
    from triattention.kernels.triton_scoring import speckv_scoring

    inputs = rkv_scoring_inputs
    head_idx = 0

    # Create shuffled positions
    seq_len = inputs["seq_len"]
    shuffled_indices = torch.randperm(seq_len, device=inputs["position_indices"].device)
    position_indices_shuffled = inputs["position_indices"][shuffled_indices]

    # Shuffle K accordingly
    k_unrot_shuffled = inputs["k_unrot"][shuffled_indices]
    k_rot_shuffled = inputs["k_rot"][shuffled_indices]
    k_rot_batch_shuffled = k_rot_shuffled.unsqueeze(0).unsqueeze(0).expand(
        inputs["batch_size"], 1, seq_len, inputs["head_dim"]
    )

    # Compute R-KV scores with shuffled data
    amp, phi, extra = compute_frequency_statistics_from_means(
        q_mean_complex=inputs["q_mean_complex"][head_idx],
        q_abs_mean=inputs["q_abs_mean"][head_idx],
        k_unrot=k_unrot_shuffled,
        style=inputs["rope_style"],
        disable_mlr=False,
    )

    rkv_scores_shuffled = score_keys_for_round(
        key_indices=position_indices_shuffled,
        round_start=inputs["round_start"],
        amp=amp,
        phi=phi,
        omega=inputs["omega"],
        extra=extra,
        offsets=inputs["offsets"],
        aggregation="mean",
        freq_scale_sq=inputs["freq_scale_sq"][head_idx],
        disable_top_n_high_freq=0,
        simulate_bug_phase_offset=0,
        disable_trig=False,
    )

    # Compute TriAttention scores with shuffled data
    triattention_scores_shuffled = speckv_scoring(
        K_rot=k_rot_batch_shuffled,
        position_indices=position_indices_shuffled,
        q_mean_real=inputs["q_mean_real"][head_idx:head_idx+1],
        q_mean_imag=inputs["q_mean_imag"][head_idx:head_idx+1],
        q_abs_mean=inputs["q_abs_mean"][head_idx:head_idx+1],
        freq_scale_sq=inputs["freq_scale_sq"][head_idx:head_idx+1],
        omega=inputs["omega"],
        offsets=inputs["offsets"],
        round_start=inputs["round_start"],
        aggregation="mean",
        rope_style=inputs["rope_style"],  # Use same RoPE style as R-KV
    )[0, 0]

    # Compare
    max_abs_error = (rkv_scores_shuffled - triattention_scores_shuffled).abs().max().item()
    correlation = torch.corrcoef(torch.stack([rkv_scores_shuffled, triattention_scores_shuffled]))[0, 1].item()

    print(f"\n=== R-KV vs TriAttention Position Handling Test ===")
    print(f"Max absolute error (shuffled): {max_abs_error:.6e}")
    print(f"Correlation (shuffled): {correlation:.6f}")

    # Assert equivalence is maintained even with shuffled positions
    assert correlation > 0.99, f"Correlation too low with shuffled positions: {correlation}"
    assert max_abs_error < 1e-3, f"Max error too high with shuffled positions: {max_abs_error}"


if __name__ == "__main__":
    # Allow running as standalone script for quick testing
    pytest.main([__file__, "-v", "-s"])
