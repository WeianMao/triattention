"""Check which complex number format K_rot uses."""

import torch

# Create a simple test tensor
K_rot = torch.arange(16, dtype=torch.float32).reshape(1, 1, 2, 8)
print("K_rot raw data:")
print(K_rot[0, 0])

# Interleaved format interpretation: [r0, i0, r1, i1, r2, i2, r3, i3]
print("\nInterleaved format (view as pairs):")
k_pairs = K_rot.view(1, 1, 2, 4, 2)
k_r_interleaved = k_pairs[..., 0]
k_i_interleaved = k_pairs[..., 1]
print(f"k_r: {k_r_interleaved[0, 0]}")
print(f"k_i: {k_i_interleaved[0, 0]}")

# Split format interpretation: [r0, r1, r2, r3, i0, i1, i2, i3]
print("\nSplit format (half_dim split):")
head_dim = 8
half_dim = 4
k_r_split = K_rot[..., :half_dim]
k_i_split = K_rot[..., half_dim:]
print(f"k_r: {k_r_split[0, 0]}")
print(f"k_i: {k_i_split[0, 0]}")

# What does RoPE actually produce?
print("\n" + "="*60)
print("RoPE format check:")

# Typical RoPE application on tensor [batch, heads, seq, dim]
# RoPE operates on pairs: x[:, :, :, 0::2] and x[:, :, :, 1::2]
# This means INTERLEAVED format

test_input = torch.arange(8, dtype=torch.float32).reshape(1, 1, 1, 8)
print(f"Input: {test_input[0, 0, 0]}")

# RoPE extracts pairs like this:
x_real = test_input[..., 0::2]  # [0, 2, 4, 6]
x_imag = test_input[..., 1::2]  # [1, 3, 5, 7]
print(f"RoPE real (0::2): {x_real[0, 0, 0]}")
print(f"RoPE imag (1::2): {x_imag[0, 0, 0]}")

print("\nConclusion: RoPE uses INTERLEAVED format [r0, i0, r1, i1, ...]")
