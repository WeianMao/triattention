"""
Debug script to investigate Triton-PyTorch scoring equivalence.

This script performs detailed comparison of Triton kernel vs PyTorch reference
with intermediate value inspection and detailed error reporting.
"""

import torch
import sys
from triattention.kernels.triton_scoring import speckv_scoring


def generate_small_test_case():
    """Generate a small, controlled test case for debugging."""
    torch.manual_seed(42)

    batch = 1
    num_heads = 2
    seq_len = 4
    freq_count = 4
    head_dim = freq_count * 2
    num_offsets = 2

    # K_rot: [batch, num_heads, seq_len, head_dim]
    K_rot = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32)

    # Q statistics
    q_mean_real = torch.randn(num_heads, freq_count, dtype=torch.float32)
    q_mean_imag = torch.randn(num_heads, freq_count, dtype=torch.float32)
    q_abs_mean = torch.rand(num_heads, freq_count, dtype=torch.float32) * 2.0 + 0.5
    freq_scale_sq = torch.rand(num_heads, freq_count, dtype=torch.float32) * 2.0 + 0.5

    # Position indices
    position_indices = torch.arange(seq_len, dtype=torch.long)

    # RoPE frequencies
    base = 10000.0
    omega = 1.0 / (base ** (torch.arange(0, freq_count, dtype=torch.float32) / freq_count))

    # Offsets
    offsets = torch.tensor([1.0, 2.0], dtype=torch.float32)

    # Round start
    round_start = 10

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


def compute_pytorch_reference_detailed(inputs, aggregation="max"):
    """
    Compute scores using PyTorch reference with detailed intermediate values.
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

    print("\n=== PyTorch Reference Implementation ===")
    print(f"Shape: batch={batch}, num_heads={num_heads}, seq_len={seq_len}, freq_count={freq_count}")

    # Split K_rot into complex pairs
    k_pairs = K_rot.view(batch, num_heads, seq_len, freq_count, 2)
    k_r = k_pairs[..., 0]  # [batch, num_heads, seq_len, freq_count]
    k_i = k_pairs[..., 1]

    print(f"\nK_rot shape: {K_rot.shape}")
    print(f"k_r shape: {k_r.shape}")
    print(f"k_r[0,0,0,:]: {k_r[0,0,0,:]}")
    print(f"k_i[0,0,0,:]: {k_i[0,0,0,:]}")

    # Compute |K_rot|
    k_abs = torch.sqrt(k_r ** 2 + k_i ** 2 + 1e-8)
    print(f"\nk_abs[0,0,0,:]: {k_abs[0,0,0,:]}")

    # Compute |Q_mean_complex|
    q_mean_abs = torch.sqrt(q_mean_real ** 2 + q_mean_imag ** 2 + 1e-8)
    print(f"\nq_mean_real[0,:]: {q_mean_real[0,:]}")
    print(f"q_mean_imag[0,:]: {q_mean_imag[0,:]}")
    print(f"q_mean_abs[0,:]: {q_mean_abs[0,:]}")

    # Expand Q statistics
    q_r_exp = q_mean_real.unsqueeze(0).unsqueeze(2)
    q_i_exp = q_mean_imag.unsqueeze(0).unsqueeze(2)
    q_abs_exp = q_abs_mean.unsqueeze(0).unsqueeze(2)
    q_mean_abs_exp = q_mean_abs.unsqueeze(0).unsqueeze(2)
    freq_scale_exp = freq_scale_sq.unsqueeze(0).unsqueeze(2)

    # Complex product: Q_mean * conj(K_rot)
    prod_real = q_r_exp * k_r + q_i_exp * k_i
    prod_imag = q_i_exp * k_r - q_r_exp * k_i

    print(f"\nprod_real[0,0,0,:]: {prod_real[0,0,0,:]}")
    print(f"prod_imag[0,0,0,:]: {prod_imag[0,0,0,:]}")

    # Coefficients
    A_coef = freq_scale_exp * prod_real
    B_coef = freq_scale_exp * prod_imag

    print(f"\nA_coef[0,0,0,:]: {A_coef[0,0,0,:]}")
    print(f"B_coef[0,0,0,:]: {B_coef[0,0,0,:]}")

    # Additive term
    extra_coef = (q_abs_exp - q_mean_abs_exp) * k_abs * freq_scale_exp
    extra_sum = extra_coef.sum(dim=-1)

    print(f"\nextra_coef[0,0,0,:]: {extra_coef[0,0,0,:]}")
    print(f"extra_sum[0,0,:]: {extra_sum[0,0,:]}")

    # Handle positions
    if position_indices.ndim == 1:
        positions = position_indices[None, None, :, None]
    else:
        positions = position_indices[:, None, :, None]

    print(f"\npositions: {positions.squeeze()}")
    print(f"round_start: {round_start}")

    # Compute scores for each offset
    scores_per_offset = []

    for i, offset in enumerate(offsets):
        print(f"\n--- Offset {i}: {offset.item()} ---")
        t = round_start + offset.item()
        print(f"t = {t}")

        # Phase correction
        delta_t = t - positions
        print(f"delta_t: {delta_t.squeeze()}")

        phase = delta_t * omega[None, None, None, :]
        print(f"omega: {omega}")
        print(f"phase[0,0,0,:]: {phase[0,0,0,:]}")

        # Compute cos and sin
        cos_vals = torch.cos(phase)
        sin_vals = torch.sin(phase)

        print(f"cos_vals[0,0,0,:]: {cos_vals[0,0,0,:]}")
        print(f"sin_vals[0,0,0,:]: {sin_vals[0,0,0,:]}")

        # Base scores
        base_scores = (A_coef * cos_vals - B_coef * sin_vals).sum(dim=-1)
        print(f"base_scores[0,0,:]: {base_scores[0,0,:]}")

        # Combined score
        combined = base_scores + extra_sum
        print(f"combined[0,0,:]: {combined[0,0,:]}")

        scores_per_offset.append(combined)

    # Aggregate
    scores_stacked = torch.stack(scores_per_offset, dim=0)

    print(f"\n--- Aggregation: {aggregation} ---")
    print(f"scores_stacked shape: {scores_stacked.shape}")

    if aggregation == "max":
        scores = scores_stacked.max(dim=0).values
    else:
        scores = scores_stacked.mean(dim=0)

    print(f"Final scores[0,0,:]: {scores[0,0,:]}")

    return scores


def main():
    """Main debug function."""
    print("=" * 80)
    print("Triton-PyTorch Scoring Equivalence Debug")
    print("=" * 80)

    # Generate test case
    inputs = generate_small_test_case()

    # Test both aggregation modes
    for aggregation in ["mean", "max"]:
        print("\n" + "=" * 80)
        print(f"Testing aggregation mode: {aggregation}")
        print("=" * 80)

        # Compute PyTorch reference
        scores_pytorch = compute_pytorch_reference_detailed(inputs, aggregation=aggregation)

        # Compute Triton
        print("\n=== Triton Kernel ===")
        inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
        scores_triton = speckv_scoring(**inputs_cuda, aggregation=aggregation)
        scores_triton_cpu = scores_triton.cpu()

        print(f"Triton scores[0,0,:]: {scores_triton_cpu[0,0,:]}")

        # Compare
        print("\n=== Comparison ===")
        diff = scores_pytorch - scores_triton_cpu
        max_abs_error = diff.abs().max().item()
        mean_abs_error = diff.abs().mean().item()

        print(f"Difference: {diff[0,0,:]}")
        print(f"Max absolute error: {max_abs_error:.6e}")
        print(f"Mean absolute error: {mean_abs_error:.6e}")

        if max_abs_error > 1e-4:
            print(f"\n❌ FAILED: Error too large (max_abs_error={max_abs_error:.6e} > 1e-4)")
            return False
        else:
            print(f"\n✓ PASSED: Error within tolerance")

    print("\n" + "=" * 80)
    print("All tests passed!")
    print("=" * 80)
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
