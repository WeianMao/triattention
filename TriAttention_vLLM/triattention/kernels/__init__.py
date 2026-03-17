"""Triton kernels for TriAttention KV cache compression.

This package contains optimized Triton kernels for:
- Scoring: Frequency-based importance scoring with RoPE phase correction (Phase 1)
- TopK: Top-k selection (Phase 2, optional)
- Gather: KV cache gathering (Phase 2, optional)
- Fused: Fused TopK + Gather (Phase 2, optional)

Key Feature:
- position_indices support: Correctly handles out-of-order KV cache via position tracking
"""

from .triton_scoring import speckv_scoring, speckv_scoring_kernel

__all__ = [
    "speckv_scoring",
    "speckv_scoring_kernel",
]
