"""
Debug script to understand differences between R-KV and TriAttention scoring.
"""

import torch
import sys
import os
from pathlib import Path

# Add R-KV to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../R-KV"))
# Add TriAttention to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from weian_development.speckv.round_pruning_utils import (
    score_keys_for_round,
    compute_frequency_statistics_from_means,
    build_rotary,
    compute_frequency_scaling,
)
from triattention.kernels.triton_scoring import speckv_scoring
from transformers import AutoConfig

# Simple test case
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

device = torch.device("cuda")
dtype = torch.float32

seq_len = 10
head_dim = 64
freq_count = head_dim // 2

# Generate simple Q statistics
q_mean_real = torch.randn(freq_count, dtype=dtype, device=device)
q_mean_imag = torch.randn(freq_count, dtype=dtype, device=device)
q_mean_complex = torch.complex(q_mean_real, q_mean_imag)
q_abs_mean = torch.abs(q_mean_complex) + torch.rand(freq_count, device=device) * 0.5

# Generate unrotated K
k_unrot = torch.randn(seq_len, head_dim, dtype=dtype, device=device)

print("="*80)
print("DEBUG: R-KV vs TriAttention Scoring")
print("="*80)

# Build RoPE
config = AutoConfig.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct", trust_remote_code=True)
config.num_attention_heads = 1
config.num_key_value_heads = 1
config.hidden_size = head_dim

rotary = build_rotary(device, Path("Qwen/Qwen2.5-0.5B-Instruct"), dtype, config=config)
rope_style = getattr(rotary, "_rope_style", "half")
print(f"\nRoPE style: {rope_style}")

# Get RoPE cos/sin
position_ids = torch.arange(seq_len, device=device, dtype=torch.long).unsqueeze(0)
dummy = torch.zeros(1, seq_len, head_dim, device=device, dtype=dtype)
cos_table, sin_table = rotary(dummy, position_ids)
cos_table = cos_table[0]
sin_table = sin_table[0]

# Apply RoPE to K (half style)
k1 = k_unrot[..., :head_dim//2]
k2 = k_unrot[..., head_dim//2:]
k_rot_half = torch.cat([
    k1 * cos_table[..., :head_dim//2] - k2 * sin_table[..., :head_dim//2],
    k2 * cos_table[..., head_dim//2:] + k1 * sin_table[..., head_dim//2:]
], dim=-1)

print(f"\nK_unrot shape: {k_unrot.shape}")
print(f"K_rot shape: {k_rot_half.shape}")

# Now let's look at how R-KV expects K
# R-KV uses K_unrot to compute complex pairs
from weian_development.speckv.round_pruning_utils import to_complex_pairs

k_complex_rkv = to_complex_pairs(k_unrot, style=rope_style)
print(f"\nR-KV K_complex shape: {k_complex_rkv.shape}")
print(f"K_complex first 3 values: {k_complex_rkv[:3]}")

# How TriAttention sees K_rot
# TriAttention expects K_rot in interleaved format: [r0, i0, r1, i1, ...]
# But we're providing it in half format: [r0, r1, ..., i0, i1, ...]

# Let's convert K_rot to interleaved format manually
k_rot_pairs_half = k_rot_half.reshape(seq_len, freq_count, 2)
k_rot_real_half = k_rot_pairs_half[..., 0]
k_rot_imag_half = k_rot_pairs_half[..., 1]

# Create interleaved K_rot
k_rot_interleaved = torch.empty(seq_len, head_dim, dtype=dtype, device=device)
k_rot_interleaved[..., 0::2] = k_rot_real_half  # Real parts at even indices
k_rot_interleaved[..., 1::2] = k_rot_imag_half  # Imag parts at odd indices

print(f"\nK_rot (half format) first element: {k_rot_half[0, :8]}")
print(f"K_rot (interleaved) first element: {k_rot_interleaved[0, :8]}")

# Compute R-KV scores
print("\n" + "="*80)
print("R-KV Scoring")
print("="*80)

amp, phi, extra = compute_frequency_statistics_from_means(
    q_mean_complex=q_mean_complex,
    q_abs_mean=q_abs_mean,
    k_unrot=k_unrot,
    style=rope_style,
    disable_mlr=False,
)

print(f"amp shape: {amp.shape}")
print(f"phi shape: {phi.shape}")
print(f"extra shape: {extra.shape}")

freq_scale = compute_frequency_scaling(rotary, head_dim, dtype, device)
freq_scale_sq = freq_scale.pow(2)
inv_freq = rotary.inv_freq.to(device=device, dtype=torch.float32)
omega = inv_freq[:freq_count]

num_offsets = 4
offsets = torch.tensor([float(2**i) for i in range(num_offsets)], dtype=torch.float32, device=device)
position_indices = torch.arange(seq_len, dtype=torch.long, device=device)
round_start = seq_len

rkv_scores = score_keys_for_round(
    key_indices=position_indices,
    round_start=round_start,
    amp=amp,
    phi=phi,
    omega=omega,
    extra=extra,
    offsets=offsets,
    aggregation="mean",
    freq_scale_sq=freq_scale_sq,
    disable_top_n_high_freq=0,
    simulate_bug_phase_offset=0,
    disable_trig=False,
)

print(f"R-KV scores: {rkv_scores}")
print(f"R-KV scores range: [{rkv_scores.min().item():.4f}, {rkv_scores.max().item():.4f}]")

# Compute TriAttention scores (using interleaved K_rot)
print("\n" + "="*80)
print("TriAttention Scoring (interleaved)")
print("="*80)

k_rot_batch_interleaved = k_rot_interleaved.unsqueeze(0).unsqueeze(0)  # [1, 1, seq_len, head_dim]

triattention_scores_interleaved = speckv_scoring(
    K_rot=k_rot_batch_interleaved,
    position_indices=position_indices,
    q_mean_real=q_mean_real.unsqueeze(0),
    q_mean_imag=q_mean_imag.unsqueeze(0),
    q_abs_mean=q_abs_mean.unsqueeze(0),
    freq_scale_sq=freq_scale_sq.unsqueeze(0),
    omega=omega,
    offsets=offsets,
    round_start=round_start,
    aggregation="mean",
)[0, 0]

print(f"TriAttention scores: {triattention_scores_interleaved}")
print(f"TriAttention scores range: [{triattention_scores_interleaved.min().item():.4f}, {triattention_scores_interleaved.max().item():.4f}]")

# Compare
diff = (rkv_scores - triattention_scores_interleaved).abs()
print("\n" + "="*80)
print("Comparison (interleaved)")
print("="*80)
print(f"Max abs diff: {diff.max().item():.6e}")
print(f"Mean abs diff: {diff.mean().item():.6e}")
corr = torch.corrcoef(torch.stack([rkv_scores, triattention_scores_interleaved]))[0, 1].item()
print(f"Correlation: {corr:.6f}")

# Try with half format too
print("\n" + "="*80)
print("TriAttention Scoring (half format)")
print("="*80)

k_rot_batch_half = k_rot_half.unsqueeze(0).unsqueeze(0)

triattention_scores_half = speckv_scoring(
    K_rot=k_rot_batch_half,
    position_indices=position_indices,
    q_mean_real=q_mean_real.unsqueeze(0),
    q_mean_imag=q_mean_imag.unsqueeze(0),
    q_abs_mean=q_abs_mean.unsqueeze(0),
    freq_scale_sq=freq_scale_sq.unsqueeze(0),
    omega=omega,
    offsets=offsets,
    round_start=round_start,
    aggregation="mean",
)[0, 0]

print(f"TriAttention scores: {triattention_scores_half}")
print(f"TriAttention scores range: [{triattention_scores_half.min().item():.4f}, {triattention_scores_half.max().item():.4f}]")

diff_half = (rkv_scores - triattention_scores_half).abs()
print("\n" + "="*80)
print("Comparison (half format)")
print("="*80)
print(f"Max abs diff: {diff_half.max().item():.6e}")
print(f"Mean abs diff: {diff_half.mean().item():.6e}")
corr_half = torch.corrcoef(torch.stack([rkv_scores, triattention_scores_half]))[0, 1].item()
print(f"Correlation: {corr_half:.6f}")
