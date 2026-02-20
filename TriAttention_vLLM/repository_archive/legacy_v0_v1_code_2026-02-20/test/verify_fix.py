"""
Verify that the Triton kernel fix resolves the FP32 equivalence issue.

Runs the same test as test_basic_equivalence_fp32 without pytest.
"""

import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from triattention.kernels.triton_scoring import speckv_scoring


def generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets, dtype=torch.float32):
    """Generate random test inputs for scoring."""
    head_dim = freq_count * 2

    K_rot = torch.randn(batch, num_heads, seq_len, head_dim, dtype=dtype)
    q_mean_real = torch.randn(num_heads, freq_count, dtype=dtype)
    q_mean_imag = torch.randn(num_heads, freq_count, dtype=dtype)
    q_abs_mean = torch.rand(num_heads, freq_count, dtype=dtype) * 2.0 + 0.5
    freq_scale_sq = torch.rand(num_heads, freq_count, dtype=dtype) * 2.0 + 0.5
    position_indices = torch.arange(seq_len, dtype=torch.long)

    base = 10000.0
    omega = 1.0 / (base ** (torch.arange(0, freq_count, dtype=torch.float32) / freq_count))
    offsets = torch.tensor([float(2**i) for i in range(num_offsets)], dtype=torch.float32)
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
    """PyTorch reference implementation."""
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

    # Split K_rot into complex pairs (INTERLEAVED format)
    k_pairs = K_rot.view(batch, num_heads, seq_len, freq_count, 2)
    k_r = k_pairs[..., 0]
    k_i = k_pairs[..., 1]

    k_abs = torch.sqrt(k_r ** 2 + k_i ** 2 + 1e-8)
    q_mean_abs = torch.sqrt(q_mean_real ** 2 + q_mean_imag ** 2 + 1e-8)

    q_r_exp = q_mean_real.unsqueeze(0).unsqueeze(2)
    q_i_exp = q_mean_imag.unsqueeze(0).unsqueeze(2)
    q_abs_exp = q_abs_mean.unsqueeze(0).unsqueeze(2)
    q_mean_abs_exp = q_mean_abs.unsqueeze(0).unsqueeze(2)
    freq_scale_exp = freq_scale_sq.unsqueeze(0).unsqueeze(2)

    prod_real = q_r_exp * k_r + q_i_exp * k_i
    prod_imag = q_i_exp * k_r - q_r_exp * k_i

    A_coef = freq_scale_exp * prod_real
    B_coef = freq_scale_exp * prod_imag

    extra_coef = (q_abs_exp - q_mean_abs_exp) * k_abs * freq_scale_exp
    extra_sum = extra_coef.sum(dim=-1)

    if position_indices.ndim == 1:
        positions = position_indices[None, None, :, None]
    else:
        positions = position_indices[:, None, :, None]

    scores_per_offset = []

    for offset in offsets:
        t = round_start + offset.item()
        delta_t = t - positions
        phase = delta_t * omega[None, None, None, :]
        cos_vals = torch.cos(phase)
        sin_vals = torch.sin(phase)
        base_scores = (A_coef * cos_vals - B_coef * sin_vals).sum(dim=-1)
        combined = base_scores + extra_sum
        scores_per_offset.append(combined)

    scores_stacked = torch.stack(scores_per_offset, dim=0)

    if aggregation == "max":
        scores = scores_stacked.max(dim=0).values
    else:
        scores = scores_stacked.mean(dim=0)

    return scores


def test_equivalence(aggregation, batch, num_heads, seq_len, freq_count, num_offsets):
    """Test Triton-PyTorch equivalence."""
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)

    inputs = generate_test_inputs(batch, num_heads, seq_len, freq_count, num_offsets, dtype=torch.float32)
    inputs_cuda = {k: v.cuda() if isinstance(v, torch.Tensor) else v for k, v in inputs.items()}

    scores_pytorch = compute_pytorch_reference(inputs, aggregation=aggregation)
    scores_triton = speckv_scoring(**inputs_cuda, aggregation=aggregation)
    scores_triton_cpu = scores_triton.cpu()

    rtol, atol = 1e-4, 1e-4
    max_error = (scores_pytorch - scores_triton_cpu).abs().max().item()
    mean_error = (scores_pytorch - scores_triton_cpu).abs().mean().item()

    passed = torch.allclose(scores_pytorch, scores_triton_cpu, rtol=rtol, atol=atol)

    return passed, max_error, mean_error


def main():
    """Run all tests."""
    print("=" * 80)
    print("VERIFYING TRITON KERNEL FIX")
    print("=" * 80)

    test_configs = [
        ("basic_fp32_mean", "mean", 2, 4, 64, 32, 8),
        ("basic_fp32_max", "max", 2, 4, 64, 32, 8),
        ("batch_1", "mean", 1, 4, 64, 32, 8),
        ("batch_8", "mean", 8, 4, 64, 32, 8),
        ("seq_len_256", "max", 2, 4, 256, 32, 8),
        ("heads_32", "mean", 2, 32, 64, 32, 8),
        ("offsets_1", "max", 2, 4, 64, 32, 1),
        ("offsets_32", "mean", 2, 4, 64, 32, 32),
    ]

    all_passed = True
    results = []

    for name, agg, batch, heads, seq_len, freq, offs in test_configs:
        passed, max_err, mean_err = test_equivalence(agg, batch, heads, seq_len, freq, offs)
        results.append((name, passed, max_err, mean_err))
        all_passed = all_passed and passed

        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"\n{status} {name:20s} max_err={max_err:.2e} mean_err={mean_err:.2e}")

    print("\n" + "=" * 80)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("=" * 80)
        print("\nThe Triton kernel now produces numerically equivalent results to PyTorch!")
        return 0
    else:
        print("✗ SOME TESTS FAILED")
        print("=" * 80)
        for name, passed, max_err, mean_err in results:
            if not passed:
                print(f"  FAILED: {name} (max_err={max_err:.2e})")
        return 1


if __name__ == "__main__":
    sys.exit(main())
