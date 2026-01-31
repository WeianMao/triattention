"""
Test script to verify numerical equivalence of TriAttention scoring optimizations.

Validates three key optimizations:
1. RoPE correction avoidance
2. Fast coefficient computation
3. Trigonometric identity expansion
"""

import torch
import numpy as np


def test_optimization1_rope_correction():
    """
    Verify: cos((t-p)·ω + φ) = cos(t·ω + φ_rot) where φ = φ_rot + p·ω

    Original: cos((t-p)·ω + φ)
    Optimized: cos(t·ω + φ_rot)
    """
    print("\n" + "="*60)
    print("Test 1: RoPE Correction Equivalence")
    print("="*60)

    # Generate random test data
    batch_size = 8
    num_heads = 32
    head_dim = 128
    seq_len = 100

    # Random positions (current and key positions)
    t = torch.randint(0, seq_len, (batch_size,), dtype=torch.float32)  # current position
    p = torch.randint(0, seq_len, (batch_size,), dtype=torch.float32)  # key position

    # RoPE frequencies (typical values for half head_dim due to pair rotation)
    dim = head_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(0, dim, dtype=torch.float32) / dim))
    omega = omega.unsqueeze(0)  # [1, dim]

    # Random φ_rot (phase from rotated K)
    phi_rot = torch.rand(batch_size, num_heads, dim) * 2 * np.pi - np.pi

    # Original algorithm: compute φ = φ_rot + p·ω
    p_expanded = p.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
    phi = phi_rot + p_expanded * omega  # [batch, num_heads, dim]

    # Original scoring: cos((t-p)·ω + φ)
    t_expanded = t.unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1]
    original_score = torch.cos((t_expanded - p_expanded) * omega + phi)

    # Optimized scoring: cos(t·ω + φ_rot)
    optimized_score = torch.cos(t_expanded * omega + phi_rot)

    # Compute errors
    abs_error = torch.abs(original_score - optimized_score)
    rel_error = abs_error / (torch.abs(original_score) + 1e-8)

    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    print(f"Max absolute error: {max_abs_error:.2e}")
    print(f"Mean absolute error: {mean_abs_error:.2e}")
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")

    # Test passes if errors are within float32 precision
    tolerance = 1e-5
    assert max_abs_error < tolerance, f"Optimization 1 failed: max error {max_abs_error:.2e} > {tolerance:.2e}"
    print(f"✓ PASS: Errors within tolerance ({tolerance:.2e})")

    return max_abs_error, max_rel_error


def test_optimization2_fast_coefficients():
    """
    Verify: A·s²·cos(φ) = s²·Re and A·s²·sin(φ) = s²·Im
    where A = |z|, z = Re + i·Im, φ = atan2(Im, Re)
    """
    print("\n" + "="*60)
    print("Test 2: Fast Coefficient Computation")
    print("="*60)

    # Generate random test data
    batch_size = 16
    num_heads = 32
    dim = 64

    # Random complex numbers (from K vector pairs)
    Re = torch.randn(batch_size, num_heads, dim)
    Im = torch.randn(batch_size, num_heads, dim)

    # Random scale factors
    s = torch.rand(batch_size, num_heads, dim) * 2.0 + 0.5  # [0.5, 2.5]

    # Original algorithm
    A = torch.sqrt(Re**2 + Im**2)  # magnitude
    phi = torch.atan2(Im, Re)  # phase
    original_A_coef = A * s**2 * torch.cos(phi)
    original_B_coef = A * s**2 * torch.sin(phi)

    # Optimized algorithm (direct computation)
    optimized_A_coef = s**2 * Re
    optimized_B_coef = s**2 * Im

    # Compute errors for A_coef
    abs_error_A = torch.abs(original_A_coef - optimized_A_coef)
    rel_error_A = abs_error_A / (torch.abs(original_A_coef) + 1e-8)

    # Compute errors for B_coef
    abs_error_B = torch.abs(original_B_coef - optimized_B_coef)
    rel_error_B = abs_error_B / (torch.abs(original_B_coef) + 1e-8)

    print("A_coef comparison:")
    print(f"  Max absolute error: {abs_error_A.max().item():.2e}")
    print(f"  Mean absolute error: {abs_error_A.mean().item():.2e}")
    print(f"  Max relative error: {rel_error_A.max().item():.2e}")
    print(f"  Mean relative error: {rel_error_A.mean().item():.2e}")

    print("\nB_coef comparison:")
    print(f"  Max absolute error: {abs_error_B.max().item():.2e}")
    print(f"  Mean absolute error: {abs_error_B.mean().item():.2e}")
    print(f"  Max relative error: {rel_error_B.max().item():.2e}")
    print(f"  Mean relative error: {rel_error_B.mean().item():.2e}")

    # Test passes if errors are within float32 precision
    tolerance = 1e-5
    max_error = max(abs_error_A.max().item(), abs_error_B.max().item())
    assert max_error < tolerance, f"Optimization 2 failed: max error {max_error:.2e} > {tolerance:.2e}"
    print(f"\n✓ PASS: Errors within tolerance ({tolerance:.2e})")

    return max_error, max(rel_error_A.max().item(), rel_error_B.max().item())


def test_optimization3_trig_identity():
    """
    Verify: A·s²·cos(t·ω + φ) = A_coef·cos(t·ω) - B_coef·sin(t·ω)
    where A_coef = A·s²·cos(φ), B_coef = A·s²·sin(φ)
    """
    print("\n" + "="*60)
    print("Test 3: Trigonometric Identity Expansion")
    print("="*60)

    # Generate random test data
    batch_size = 12
    num_heads = 32
    dim = 64

    # Random amplitude and phase
    A = torch.rand(batch_size, num_heads, dim) * 2.0 + 0.5
    s = torch.rand(batch_size, num_heads, dim) * 1.5 + 0.5
    phi = torch.rand(batch_size, num_heads, dim) * 2 * np.pi - np.pi

    # Random position and frequency
    t = torch.rand(batch_size, 1, 1) * 100
    omega = torch.rand(1, 1, dim) * 0.1

    # Compute coefficients
    A_coef = A * s**2 * torch.cos(phi)
    B_coef = A * s**2 * torch.sin(phi)

    # Original: A·s²·cos(t·ω + φ)
    original_score = A * s**2 * torch.cos(t * omega + phi)

    # Optimized: A_coef·cos(t·ω) - B_coef·sin(t·ω)
    optimized_score = A_coef * torch.cos(t * omega) - B_coef * torch.sin(t * omega)

    # Compute errors
    abs_error = torch.abs(original_score - optimized_score)
    rel_error = abs_error / (torch.abs(original_score) + 1e-8)

    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    print(f"Max absolute error: {max_abs_error:.2e}")
    print(f"Mean absolute error: {mean_abs_error:.2e}")
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")

    # Test passes if errors are within float32 precision
    tolerance = 1e-5
    assert max_abs_error < tolerance, f"Optimization 3 failed: max error {max_abs_error:.2e} > {tolerance:.2e}"
    print(f"✓ PASS: Errors within tolerance ({tolerance:.2e})")

    return max_abs_error, max_rel_error


def test_full_pipeline():
    """
    Verify complete scoring pipeline: original vs optimized algorithm.
    Simulates realistic TriAttention scoring scenario.
    """
    print("\n" + "="*60)
    print("Test 4: Full Pipeline Integration")
    print("="*60)

    # Realistic parameters
    batch_size = 4
    num_heads = 32
    head_dim = 128
    num_keys = 50  # number of cached keys to score

    # Current query position
    t = torch.randint(100, 200, (batch_size,), dtype=torch.float32)

    # Key positions (from KV cache)
    p = torch.randint(0, 100, (batch_size, num_keys), dtype=torch.float32)

    # RoPE frequencies
    dim = head_dim // 2
    omega = 1.0 / (10000 ** (torch.arange(0, dim, dtype=torch.float32) / dim))
    omega = omega.unsqueeze(0).unsqueeze(0)  # [1, 1, dim]

    # Simulated K vectors (rotated by RoPE at position p)
    # In reality these come from KV cache, here we generate random complex pairs
    K_real = torch.randn(batch_size, num_heads, num_keys, dim)
    K_imag = torch.randn(batch_size, num_heads, num_keys, dim)

    # Scale factors (from attention mechanism)
    scale = torch.rand(batch_size, num_heads, num_keys, dim) * 1.0 + 0.5

    # === ORIGINAL ALGORITHM ===
    # Step 1: Compute amplitude and phase
    A_orig = torch.sqrt(K_real**2 + K_imag**2)
    phi_rot_orig = torch.atan2(K_imag, K_real)

    # Step 2: Correct phase φ = φ_rot + p·ω
    p_exp = p.unsqueeze(1).unsqueeze(-1)  # [batch, 1, num_keys, 1]
    phi_orig = phi_rot_orig + p_exp * omega

    # Step 3: Compute position-dependent score
    t_exp = t.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # [batch, 1, 1, 1]
    score_per_dim_orig = A_orig * scale**2 * torch.cos((t_exp - p_exp) * omega + phi_orig)
    score_orig = score_per_dim_orig.sum(dim=-1)  # sum over dimensions

    # === OPTIMIZED ALGORITHM ===
    # Step 1: Precompute position-independent coefficients (no φ needed)
    A_coef = scale**2 * K_real
    B_coef = scale**2 * K_imag

    # Step 2: Compute φ_rot for position correction
    phi_rot_opt = torch.atan2(K_imag, K_real)

    # Step 3: Compute score using trig identity
    # cos(t·ω + φ_rot) = cos(t·ω)·cos(φ_rot) - sin(t·ω)·sin(φ_rot)
    t_omega = t_exp * omega  # [batch, 1, 1, dim]
    cos_t_omega = torch.cos(t_omega)
    sin_t_omega = torch.sin(t_omega)

    # For each key position, compute score
    score_per_dim_opt = A_coef * cos_t_omega - B_coef * sin_t_omega
    score_opt = score_per_dim_opt.sum(dim=-1)  # sum over dimensions

    # === COMPARE RESULTS ===
    abs_error = torch.abs(score_orig - score_opt)
    rel_error = abs_error / (torch.abs(score_orig) + 1e-8)

    max_abs_error = abs_error.max().item()
    mean_abs_error = abs_error.mean().item()
    max_rel_error = rel_error.max().item()
    mean_rel_error = rel_error.mean().item()

    print(f"Shape of scores: {score_orig.shape}")
    print(f"Max absolute error: {max_abs_error:.2e}")
    print(f"Mean absolute error: {mean_abs_error:.2e}")
    print(f"Max relative error: {max_rel_error:.2e}")
    print(f"Mean relative error: {mean_rel_error:.2e}")

    # Sample comparison (first batch, first head)
    print("\nSample scores (first batch, first head, first 5 keys):")
    print(f"Original:  {score_orig[0, 0, :5].tolist()}")
    print(f"Optimized: {score_opt[0, 0, :5].tolist()}")
    print(f"Difference: {(score_orig[0, 0, :5] - score_opt[0, 0, :5]).tolist()}")

    # Test passes if errors are within float32 precision
    tolerance = 5e-4  # relaxed due to accumulation across multiple operations
    assert max_abs_error < tolerance, f"Full pipeline failed: max error {max_abs_error:.2e} > {tolerance:.2e}"
    print(f"\n✓ PASS: Errors within tolerance ({tolerance:.2e})")

    return max_abs_error, max_rel_error


def test_dtype_precision():
    """
    Test optimizations under different dtypes (float32, float16, bfloat16).
    """
    print("\n" + "="*60)
    print("Test 5: Dtype Precision Comparison")
    print("="*60)

    dtypes = [torch.float32, torch.float16, torch.bfloat16]
    results = {}

    for dtype in dtypes:
        print(f"\nTesting with dtype: {dtype}")

        # Generate test data
        batch_size = 4
        num_heads = 16
        dim = 64

        A = torch.rand(batch_size, num_heads, dim, dtype=dtype) * 2.0
        s = torch.rand(batch_size, num_heads, dim, dtype=dtype) * 1.5
        phi = torch.rand(batch_size, num_heads, dim, dtype=dtype) * 2 * np.pi - np.pi
        t = torch.rand(batch_size, 1, 1, dtype=dtype) * 100
        omega = torch.rand(1, 1, dim, dtype=dtype) * 0.1

        # Original
        original = A * s**2 * torch.cos(t * omega + phi)

        # Optimized
        A_coef = A * s**2 * torch.cos(phi)
        B_coef = A * s**2 * torch.sin(phi)
        optimized = A_coef * torch.cos(t * omega) - B_coef * torch.sin(t * omega)

        # Compute error (in float32 for accurate comparison)
        abs_error = torch.abs(original.float() - optimized.float())
        max_error = abs_error.max().item()
        mean_error = abs_error.mean().item()

        results[dtype] = (max_error, mean_error)
        print(f"  Max error: {max_error:.2e}")
        print(f"  Mean error: {mean_error:.2e}")

        # Expected tolerances
        if dtype == torch.float32:
            tolerance = 1e-5
        elif dtype == torch.float16:
            tolerance = 2e-2  # ~2% error acceptable for float16
        else:  # bfloat16 has only 7 mantissa bits
            tolerance = 1e-1  # ~10% error acceptable for bfloat16

        assert max_error < tolerance, f"Dtype {dtype} failed: {max_error:.2e} > {tolerance:.2e}"
        print(f"  ✓ PASS: Within {dtype} tolerance ({tolerance:.2e})")

    return results


if __name__ == "__main__":
    print("\n" + "="*70)
    print("TriAttention Optimization Equivalence Test Suite")
    print("="*70)

    try:
        # Run individual optimization tests
        abs_err1, rel_err1 = test_optimization1_rope_correction()
        abs_err2, rel_err2 = test_optimization2_fast_coefficients()
        abs_err3, rel_err3 = test_optimization3_trig_identity()

        # Run full pipeline test
        abs_err4, rel_err4 = test_full_pipeline()

        # Run dtype precision test
        dtype_results = test_dtype_precision()

        # Summary
        print("\n" + "="*70)
        print("TEST SUMMARY")
        print("="*70)
        print(f"Test 1 (RoPE Correction):       Max Abs Error = {abs_err1:.2e}, Max Rel Error = {rel_err1:.2e}")
        print(f"Test 2 (Fast Coefficients):     Max Abs Error = {abs_err2:.2e}, Max Rel Error = {rel_err2:.2e}")
        print(f"Test 3 (Trig Identity):         Max Abs Error = {abs_err3:.2e}, Max Rel Error = {rel_err3:.2e}")
        print(f"Test 4 (Full Pipeline):         Max Abs Error = {abs_err4:.2e}, Max Rel Error = {rel_err4:.2e}")
        print("\nDtype Precision:")
        for dtype, (max_err, mean_err) in dtype_results.items():
            print(f"  {str(dtype):20s}: Max = {max_err:.2e}, Mean = {mean_err:.2e}")

        print("\n" + "="*70)
        print("ALL TESTS PASSED ✓")
        print("="*70)
        print("\nConclusion: The optimized TriAttention scoring algorithm is")
        print("numerically equivalent to the original algorithm within floating")
        print("point precision limits.")

    except AssertionError as e:
        print("\n" + "="*70)
        print(f"TEST FAILED: {e}")
        print("="*70)
        raise
    except Exception as e:
        print("\n" + "="*70)
        print(f"UNEXPECTED ERROR: {e}")
        print("="*70)
        raise
