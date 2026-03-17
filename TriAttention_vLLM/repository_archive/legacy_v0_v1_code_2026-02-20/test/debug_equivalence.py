"""
Debug script to isolate FP32 equivalence issues between Triton and PyTorch.

Compares intermediate computation steps to find where numerical divergence occurs.
"""

import torch
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from triattention.kernels.triton_scoring import speckv_scoring


def generate_simple_inputs():
    """Generate simple deterministic inputs for debugging."""
    torch.manual_seed(42)

    batch, num_heads, seq_len, freq_count = 1, 2, 8, 4
    num_offsets = 2
    head_dim = freq_count * 2

    # Small values for easier debugging
    K_rot = torch.randn(batch, num_heads, seq_len, head_dim, dtype=torch.float32) * 0.1

    q_mean_real = torch.randn(num_heads, freq_count, dtype=torch.float32) * 0.1
    q_mean_imag = torch.randn(num_heads, freq_count, dtype=torch.float32) * 0.1
    q_abs_mean = torch.rand(num_heads, freq_count, dtype=torch.float32) * 0.5
    freq_scale_sq = torch.ones(num_heads, freq_count, dtype=torch.float32)

    position_indices = torch.arange(seq_len, dtype=torch.long)

    base = 10000.0
    omega = 1.0 / (base ** (torch.arange(0, freq_count, dtype=torch.float32) / freq_count))

    offsets = torch.tensor([1.0, 2.0], dtype=torch.float32)
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


def compute_pytorch_detailed(inputs, aggregation="max"):
    """
    PyTorch reference with detailed intermediate outputs.
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

    # Split K_rot into complex pairs
    k_pairs = K_rot.view(batch, num_heads, seq_len, freq_count, 2)
    k_r = k_pairs[..., 0]
    k_i = k_pairs[..., 1]

    print("=" * 60)
    print("PYTORCH IMPLEMENTATION - DETAILED")
    print("=" * 60)
    print(f"\nInput shapes:")
    print(f"  K_rot: {K_rot.shape}")
    print(f"  k_r: {k_r.shape}")
    print(f"  k_i: {k_i.shape}")
    print(f"  q_mean_real: {q_mean_real.shape}")
    print(f"  position_indices: {position_indices.shape}")

    # Compute |K_rot|
    k_abs = torch.sqrt(k_r ** 2 + k_i ** 2 + 1e-8)
    print(f"\n|K_rot| sample [0,0,0,:]:")
    print(f"  {k_abs[0, 0, 0, :].cpu().numpy()}")

    # Compute |Q_mean_complex|
    q_mean_abs = torch.sqrt(q_mean_real ** 2 + q_mean_imag ** 2 + 1e-8)
    print(f"\n|Q_mean| [head 0]:")
    print(f"  {q_mean_abs[0, :].cpu().numpy()}")

    # Expand Q statistics
    q_r_exp = q_mean_real.unsqueeze(0).unsqueeze(2)
    q_i_exp = q_mean_imag.unsqueeze(0).unsqueeze(2)
    q_abs_exp = q_abs_mean.unsqueeze(0).unsqueeze(2)
    q_mean_abs_exp = q_mean_abs.unsqueeze(0).unsqueeze(2)
    freq_scale_exp = freq_scale_sq.unsqueeze(0).unsqueeze(2)

    # Complex product: Q_mean * conj(K_rot)
    prod_real = q_r_exp * k_r + q_i_exp * k_i
    prod_imag = q_i_exp * k_r - q_r_exp * k_i

    print(f"\nComplex product (real) [0,0,0,:]:")
    print(f"  {prod_real[0, 0, 0, :].cpu().numpy()}")
    print(f"Complex product (imag) [0,0,0,:]:")
    print(f"  {prod_imag[0, 0, 0, :].cpu().numpy()}")

    # Coefficients
    A_coef = freq_scale_exp * prod_real
    B_coef = freq_scale_exp * prod_imag

    print(f"\nA_coef [0,0,0,:]:")
    print(f"  {A_coef[0, 0, 0, :].cpu().numpy()}")
    print(f"B_coef [0,0,0,:]:")
    print(f"  {B_coef[0, 0, 0, :].cpu().numpy()}")

    # Extra term
    extra_coef = (q_abs_exp - q_mean_abs_exp) * k_abs * freq_scale_exp
    extra_sum = extra_coef.sum(dim=-1)

    print(f"\nextra_coef [0,0,0,:]:")
    print(f"  {extra_coef[0, 0, 0, :].cpu().numpy()}")
    print(f"extra_sum [0,0,:]:")
    print(f"  {extra_sum[0, 0, :].cpu().numpy()}")

    # Handle positions
    if position_indices.ndim == 1:
        positions = position_indices[None, None, :, None]
    else:
        positions = position_indices[:, None, :, None]

    print(f"\npositions (reshaped) [0,0,:,0]:")
    print(f"  {positions[0, 0, :, 0].cpu().numpy()}")

    # Compute scores for each offset
    scores_per_offset = []

    for off_idx, offset in enumerate(offsets):
        t = round_start + offset.item()
        print(f"\n--- Offset {off_idx}: {offset.item()}, t={t} ---")

        delta_t = t - positions
        print(f"delta_t [0,0,:,0]: {delta_t[0, 0, :, 0].cpu().numpy()}")

        phase = delta_t * omega[None, None, None, :]
        print(f"phase [0,0,0,:]: {phase[0, 0, 0, :].cpu().numpy()}")

        cos_vals = torch.cos(phase)
        sin_vals = torch.sin(phase)
        print(f"cos [0,0,0,:]: {cos_vals[0, 0, 0, :].cpu().numpy()}")
        print(f"sin [0,0,0,:]: {sin_vals[0, 0, 0, :].cpu().numpy()}")

        base_scores = (A_coef * cos_vals - B_coef * sin_vals).sum(dim=-1)
        print(f"base_scores [0,0,:]: {base_scores[0, 0, :].cpu().numpy()}")

        combined = base_scores + extra_sum
        print(f"combined [0,0,:]: {combined[0, 0, :].cpu().numpy()}")

        scores_per_offset.append(combined)

    scores_stacked = torch.stack(scores_per_offset, dim=0)

    if aggregation == "max":
        scores = scores_stacked.max(dim=0).values
    else:
        scores = scores_stacked.mean(dim=0)

    print(f"\nFinal scores (aggregation={aggregation}) [0,0,:]:")
    print(f"  {scores[0, 0, :].cpu().numpy()}")

    return scores


def test_triton_pytorch_detailed():
    """Compare Triton and PyTorch with detailed diagnostics."""
    print("\n" + "=" * 80)
    print("DEBUGGING TRITON VS PYTORCH EQUIVALENCE")
    print("=" * 80)

    inputs = generate_simple_inputs()

    # PyTorch reference
    scores_pytorch = compute_pytorch_detailed(inputs, aggregation="max")

    # Triton
    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}
    scores_triton = speckv_scoring(**inputs_cuda, aggregation="max")
    scores_triton_cpu = scores_triton.cpu()

    print("\n" + "=" * 60)
    print("TRITON IMPLEMENTATION - OUTPUT")
    print("=" * 60)
    print(f"\nTriton scores [0,0,:]:")
    print(f"  {scores_triton_cpu[0, 0, :].numpy()}")

    # Comparison
    print("\n" + "=" * 60)
    print("COMPARISON")
    print("=" * 60)
    diff = scores_pytorch - scores_triton_cpu
    print(f"\nPyTorch scores [0,0,:]:")
    print(f"  {scores_pytorch[0, 0, :].numpy()}")
    print(f"\nTriton scores [0,0,:]:")
    print(f"  {scores_triton_cpu[0, 0, :].numpy()}")
    print(f"\nDifference (PyTorch - Triton) [0,0,:]:")
    print(f"  {diff[0, 0, :].numpy()}")
    print(f"\nMax absolute error: {diff.abs().max().item():.6e}")
    print(f"Mean absolute error: {diff.abs().mean().item():.6e}")

    # Test with mean aggregation as well
    print("\n" + "=" * 80)
    print("TESTING MEAN AGGREGATION")
    print("=" * 80)

    scores_pytorch_mean = compute_pytorch_detailed(inputs, aggregation="mean")
    scores_triton_mean = speckv_scoring(**inputs_cuda, aggregation="mean").cpu()

    diff_mean = scores_pytorch_mean - scores_triton_mean
    print(f"\nDifference (mean agg) [0,0,:]:")
    print(f"  {diff_mean[0, 0, :].numpy()}")
    print(f"Max absolute error: {diff_mean.abs().max().item():.6e}")


if __name__ == "__main__":
    test_triton_pytorch_detailed()
