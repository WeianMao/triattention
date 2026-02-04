"""
Triton kernel for frequency-domain KV cache scoring with RoPE phase correction.

Implements the optimized scoring formula from R-KV with key improvements:
1. Avoid RoPE inversion: Use K_rot directly, RoPE position info is "baked in"
2. Single K read: Load K once, iterate offsets in registers
3. Precomputed trig tables: cos/sin values computed at vLLM init, not at scoring time

Formula:
    score = base_scores + additive

    base_scores = sum_over_freq(A * cos(t*omega) - B * sin(t*omega))
    additive = sum_over_freq((|q_abs_mean| - |q_mean_complex|) * |k_rot| * freq_scale^2)

    Where:
        t = round_start + offset (query position)
        A = freq_scale^2 * Re(Q_mean * conj(K_rot))
        B = freq_scale^2 * Im(Q_mean * conj(K_rot))

Key Insight:
    Since K_rot = K_unrot * e^{i*p*omega}, the position p is already encoded in K_rot.
    The phase only depends on query position t, NOT on key position p.
    This is mathematically equivalent to R-KV's formula using K_unrot.

See docs/RKV_EQUIVALENCE_FIX.md for full derivation.
"""

import torch
import triton
import triton.language as tl
from typing import Tuple, Optional


# =============================================================================
# Precomputed Trigonometric Tables
# =============================================================================

class TrigTableCache:
    """
    Cache for precomputed cos/sin tables used in scoring.

    Created once at vLLM initialization, used for all subsequent scoring calls.
    Memory usage: ~8MB for 128K seq, interval=128, freq=64
    """

    def __init__(
        self,
        max_seq_len: int,
        compress_interval: int,
        offsets: torch.Tensor,
        omega: torch.Tensor,
        device: torch.device,
    ):
        """
        Precompute all possible cos/sin values for scoring.

        Args:
            max_seq_len: Maximum sequence length (e.g., 131072 for 128K)
            compress_interval: Compression triggers every N steps (e.g., 128)
            offsets: Offset values for multi-position scoring [num_offsets]
            omega: Angular frequencies [freq_count]
            device: CUDA device
        """
        self.compress_interval = compress_interval
        self.device = device
        self.num_offsets = offsets.shape[0]
        self.freq_count = omega.shape[0]

        # All possible round_start values: [interval, 2*interval, 3*interval, ...]
        round_starts = torch.arange(
            compress_interval,
            max_seq_len + 1,
            compress_interval,
            device=device,
            dtype=torch.float32
        )
        self.num_positions = round_starts.shape[0]

        # Compute all t = round_start + offset combinations
        # t: [num_positions, num_offsets]
        t = round_starts[:, None] + offsets[None, :].to(device=device, dtype=torch.float32)

        # Compute phase = t * omega for all combinations
        # phase: [num_positions, num_offsets, freq_count]
        phase = t[:, :, None] * omega[None, None, :].to(device=device, dtype=torch.float32)

        # Precompute cos and sin
        self.cos_table = torch.cos(phase).contiguous()  # [num_positions, num_offsets, freq_count]
        self.sin_table = torch.sin(phase).contiguous()  # [num_positions, num_offsets, freq_count]

        # Memory usage estimation
        mem_bytes = 2 * self.cos_table.numel() * 4  # float32
        self.memory_mb = mem_bytes / (1024 * 1024)

    def get_trig_values(self, round_start: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get precomputed cos/sin values for a given round_start.

        Args:
            round_start: Current compression round start position

        Returns:
            cos_vals: [num_offsets, freq_count]
            sin_vals: [num_offsets, freq_count]
        """
        if round_start % self.compress_interval != 0:
            raise ValueError(
                f"round_start {round_start} must be a multiple of compress_interval {self.compress_interval}"
            )
        if round_start < self.compress_interval or round_start > self.compress_interval * self.num_positions:
            raise ValueError(
                f"round_start {round_start} out of range. "
                f"Expected in [{self.compress_interval}, {self.compress_interval * self.num_positions}]"
            )
        idx = (round_start // self.compress_interval) - 1
        return self.cos_table[idx], self.sin_table[idx]

    def __repr__(self) -> str:
        return (
            f"TrigTableCache(positions={self.num_positions}, offsets={self.num_offsets}, "
            f"freq={self.freq_count}, memory={self.memory_mb:.1f}MB)"
        )


def create_trig_cache(
    max_seq_len: int,
    compress_interval: int,
    offsets: torch.Tensor,
    omega: torch.Tensor,
    device: torch.device,
    warn_threshold_mb: float = 100.0,
) -> TrigTableCache:
    """
    Factory function to create TrigTableCache with memory warning.

    Args:
        max_seq_len: Maximum sequence length
        compress_interval: Compression interval
        offsets: Offset values [num_offsets]
        omega: Angular frequencies [freq_count]
        device: CUDA device
        warn_threshold_mb: Warn if memory exceeds this threshold

    Returns:
        TrigTableCache instance
    """
    cache = TrigTableCache(max_seq_len, compress_interval, offsets, omega, device)

    if cache.memory_mb > warn_threshold_mb:
        import warnings
        warnings.warn(
            f"TrigTableCache uses {cache.memory_mb:.1f}MB, exceeds threshold {warn_threshold_mb}MB. "
            f"Consider increasing compress_interval or reducing max_seq_len.",
            UserWarning
        )

    return cache


@triton.jit
def speckv_scoring_kernel(
    # Input tensors
    K_rot_ptr,          # [batch, num_heads, seq_len, head_dim] - Rotated keys
    position_indices_ptr,  # [batch, seq_len] or [seq_len] - Original positions of each key
    q_mean_real_ptr,    # [num_heads, freq_count] - Real part of Q mean
    q_mean_imag_ptr,    # [num_heads, freq_count] - Imaginary part of Q mean
    q_abs_mean_ptr,     # [num_heads, freq_count] - |Q_abs_mean|
    freq_scale_sq_ptr,  # [num_heads, freq_count] - Frequency scaling squared
    omega_ptr,          # [freq_count] - Angular frequencies
    offsets_ptr,        # [num_offsets] - Offset values
    round_start,        # Scalar - Current round start position
    # Output tensor
    scores_ptr,         # [batch, num_heads, seq_len] - Output scores
    # Strides
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_pb,
    stride_pn,
    # Dimensions
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    freq_count,
    num_offsets: tl.constexpr,  # Must be constexpr for static_range
    aggregation_mode: tl.constexpr,  # 0 = max, 1 = mean
    rope_style: tl.constexpr,  # 0 = interleaved [r0,i0,r1,i1,...], 1 = half [r0,r1,...,i0,i1,...]
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Compute frequency-domain scores for KV cache compression.

    Grid: (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_N))

    Each program processes BLOCK_N tokens for one (batch, head) pair.

    RoPE styles:
        - interleaved (0): K_rot layout is [r0, i0, r1, i1, ...] (HuggingFace default)
        - half (1): K_rot layout is [r0, r1, ..., i0, i1, ...] (Qwen models)
    """
    # Program indices
    bh_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    batch_idx = bh_idx // num_heads
    head_idx = bh_idx % num_heads

    # Token offsets for this block
    n_start = block_idx * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < seq_len

    # Frequency offsets
    f_offs = tl.arange(0, BLOCK_F)
    f_mask = f_offs < freq_count

    # Half dimension for complex pairing (front/back halves)
    half_dim = head_dim // 2

    # NOTE: position_indices is NOT loaded from GPU memory
    # Since K_rot already contains RoPE rotation, phase calculation uses t*omega + phi_rot
    # where phi_rot is computed from K_rot. No need to track original positions.
    # This saves GPU memory bandwidth and storage.

    # Load omega (angular frequencies)
    # NOTE: Cast to fp32 for tl.cos/tl.sin compatibility (bf16 causes compilation error)
    omega = tl.load(omega_ptr + f_offs, mask=f_mask, other=0.0).to(tl.float32)  # [BLOCK_F]

    # === Load Q statistics for this head (shared across all tokens) ===
    # NOTE: Cast to fp32 for tl.sqrt compatibility (bf16 input causes compilation error)
    q_base = head_idx * freq_count
    q_r = tl.load(q_mean_real_ptr + q_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)
    q_i = tl.load(q_mean_imag_ptr + q_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)
    q_abs = tl.load(q_abs_mean_ptr + q_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)
    freq_scale = tl.load(freq_scale_sq_ptr + q_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)

    # Compute |Q_mean_complex| from real/imag parts
    q_mean_abs = tl.sqrt(q_r * q_r + q_i * q_i + 1e-8)

    # === Process tokens in block (vectorized over BLOCK_N) ===
    # Load K_rot for all tokens in block
    # Shape: [BLOCK_N, BLOCK_F]
    k_base = batch_idx * stride_kb + head_idx * stride_kh

    if rope_style == 0:
        # INTERLEAVED format: [r0, i0, r1, i1, r2, i2, ...]
        # Real parts at even indices: 0, 2, 4, ... (f * 2)
        k_r_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + (f_offs[None, :] * 2)
        # Imaginary parts at odd indices: 1, 3, 5, ... (f * 2 + 1)
        k_i_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + (f_offs[None, :] * 2 + 1)
    else:
        # HALF format: [r0, r1, ..., i0, i1, ...] (Qwen models)
        # Real parts in first half: 0, 1, ..., freq_count-1
        k_r_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + f_offs[None, :]
        # Imaginary parts in second half: freq_count, freq_count+1, ...
        k_i_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + (f_offs[None, :] + half_dim)

    # Cast K to fp32 for tl.sqrt compatibility
    k_r = tl.load(k_r_ptrs, mask=n_mask[:, None] & f_mask[None, :], other=0.0).to(tl.float32)
    k_i = tl.load(k_i_ptrs, mask=n_mask[:, None] & f_mask[None, :], other=0.0).to(tl.float32)

    # === Compute position-independent coefficients ===
    # Broadcast: q [BLOCK_F] -> [BLOCK_N, BLOCK_F]
    # Complex product: Q_mean * conj(K_rot)
    prod_real = q_r[None, :] * k_r + q_i[None, :] * k_i  # [BLOCK_N, BLOCK_F]
    prod_imag = q_i[None, :] * k_r - q_r[None, :] * k_i  # [BLOCK_N, BLOCK_F]

    # Coefficients for cos/sin expansion
    A_coef = freq_scale[None, :] * prod_real  # [BLOCK_N, BLOCK_F]
    B_coef = freq_scale[None, :] * prod_imag  # [BLOCK_N, BLOCK_F]

    # |K_rot| for additive term
    k_abs = tl.sqrt(k_r * k_r + k_i * k_i + 1e-8)  # [BLOCK_N, BLOCK_F]

    # Additive term (MLR - Magnitude Linear Regression)
    # Formula matches R-KV: extra = (q_abs_mean - q_mean_abs) * k_abs * freq_scale
    # NOTE: disable_mlr is not yet supported as a kernel constexpr parameter
    # The wrapper should fall back to PyTorch implementation if disable_mlr=True
    extra_coef = (q_abs[None, :] - q_mean_abs[None, :]) * k_abs * freq_scale[None, :]
    extra_sum = tl.sum(extra_coef, axis=1)  # [BLOCK_N]

    # === Compute scores for all offsets ===
    # Initialize accumulator: [BLOCK_N, num_offsets]
    max_scores = tl.full([BLOCK_N], value=-1e10, dtype=tl.float32)
    mean_scores = tl.zeros([BLOCK_N], dtype=tl.float32)

    # Iterate over offsets (static_range ensures compile-time unrolling)
    for off_idx in tl.static_range(num_offsets):
        # Load offset value
        # NOTE: Cast to fp32 for tl.cos/tl.sin compatibility (bf16 causes compilation error)
        offset = tl.load(offsets_ptr + off_idx).to(tl.float32)

        # Compute phase for query position t
        # Since K_rot already contains RoPE rotation (K_rot = K_unrot * e^{i*p*omega}),
        # the phase only depends on query position t, NOT on key position p.
        # Formula: phase = t * omega (where t = round_start + offset)
        # The position information from K_rot is already captured in prod_real/prod_imag
        # which are used to compute A_coef and B_coef.
        # See docs/RKV_EQUIVALENCE_FIX.md for mathematical derivation.
        t = round_start + offset  # scalar (query position)
        phase = t * omega  # [BLOCK_F]
        # Note: position_indices parameter kept for potential future use,
        # but NOT used in phase calculation when using K_rot

        # Compute cos and sin of phase (broadcast to all tokens)
        cos_vals = tl.cos(phase)  # [BLOCK_F]
        sin_vals = tl.sin(phase)  # [BLOCK_F]

        # Compute base scores: dot product over frequencies
        # A * cos - B * sin, shape: [BLOCK_N]
        # Broadcast cos/sin [BLOCK_F] to match A_coef/B_coef [BLOCK_N, BLOCK_F]
        base_scores = tl.sum(A_coef * cos_vals[None, :] - B_coef * sin_vals[None, :], axis=1)

        # Combined score
        combined = base_scores + extra_sum  # [BLOCK_N]

        # Update aggregators
        max_scores = tl.maximum(max_scores, combined)
        mean_scores = mean_scores + combined

    # Finalize aggregation
    if aggregation_mode == 0:  # max
        final_scores = max_scores
    else:  # mean
        final_scores = mean_scores / num_offsets

    # === Store results ===
    out_base = batch_idx * (num_heads * seq_len) + head_idx * seq_len
    out_ptrs = scores_ptr + out_base + n_offs
    tl.store(out_ptrs, final_scores, mask=n_mask)


# =============================================================================
# Kernel with Precomputed Trig Tables (Optimized Version)
# =============================================================================

@triton.jit
def speckv_scoring_kernel_cached(
    # Input tensors
    K_rot_ptr,          # [batch, num_heads, seq_len, head_dim] - Rotated keys
    position_indices_ptr,  # [batch, seq_len] or [seq_len] - Original positions of each key
    q_mean_real_ptr,    # [num_heads, freq_count] - Real part of Q mean
    q_mean_imag_ptr,    # [num_heads, freq_count] - Imaginary part of Q mean
    q_abs_mean_ptr,     # [num_heads, freq_count] - |Q_abs_mean|
    freq_scale_sq_ptr,  # [num_heads, freq_count] - Frequency scaling squared
    omega_ptr,          # [freq_count] - Angular frequencies (needed for position correction)
    cos_table_ptr,      # [num_offsets, freq_count] - Precomputed cos(t*omega) values
    sin_table_ptr,      # [num_offsets, freq_count] - Precomputed sin(t*omega) values
    round_start,        # Scalar - Current round start position
    # Output tensor
    scores_ptr,         # [batch, num_heads, seq_len] - Output scores
    # Strides
    stride_kb,
    stride_kh,
    stride_kn,
    stride_kd,
    stride_pb,
    stride_pn,
    # Dimensions
    batch_size,
    num_heads,
    seq_len,
    head_dim,
    freq_count,
    num_offsets: tl.constexpr,
    aggregation_mode: tl.constexpr,  # 0 = max, 1 = mean
    rope_style: tl.constexpr,  # 0 = interleaved, 1 = half
    # Block sizes
    BLOCK_N: tl.constexpr,
    BLOCK_F: tl.constexpr,
):
    """
    Optimized scoring kernel using precomputed cos/sin tables.

    Since K_rot already contains RoPE rotation (K_rot = K_unrot * e^{i*p*omega}),
    the phase only depends on query position t. The precomputed tables store
    cos(t*omega) and sin(t*omega), which can be used directly after applying
    the phi_rot correction.

    Grid: (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_N))

    RoPE styles:
        - interleaved (0): K_rot layout is [r0, i0, r1, i1, ...]
        - half (1): K_rot layout is [r0, r1, ..., i0, i1, ...]
    """
    # Program indices
    bh_idx = tl.program_id(0)
    block_idx = tl.program_id(1)

    batch_idx = bh_idx // num_heads
    head_idx = bh_idx % num_heads

    # Token offsets for this block
    n_start = block_idx * BLOCK_N
    n_offs = n_start + tl.arange(0, BLOCK_N)
    n_mask = n_offs < seq_len

    # Frequency offsets
    f_offs = tl.arange(0, BLOCK_F)
    f_mask = f_offs < freq_count

    # Half dimension for half-style RoPE
    half_dim = head_dim // 2

    # NOTE: position_indices is NOT loaded - see first kernel for explanation.
    # Phase calculation uses t*omega + phi_rot, no need for per-token positions.

    # Load omega (needed for position correction)
    # NOTE: Cast to fp32 for consistency with main kernel (bf16 causes trig function errors)
    omega = tl.load(omega_ptr + f_offs, mask=f_mask, other=0.0).to(tl.float32)  # [BLOCK_F]

    # === Load Q statistics for this head ===
    # NOTE: Cast to fp32 for tl.sqrt compatibility (bf16 input causes compilation error)
    q_base = head_idx * freq_count
    q_r = tl.load(q_mean_real_ptr + q_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)
    q_i = tl.load(q_mean_imag_ptr + q_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)
    q_abs = tl.load(q_abs_mean_ptr + q_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)
    freq_scale = tl.load(freq_scale_sq_ptr + q_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)

    # Compute |Q_mean_complex|
    q_mean_abs = tl.sqrt(q_r * q_r + q_i * q_i + 1e-8)

    # === Load K_rot for this block ===
    k_base = batch_idx * stride_kb + head_idx * stride_kh

    if rope_style == 0:
        # INTERLEAVED format: [r0, i0, r1, i1, ...]
        k_r_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + (f_offs[None, :] * 2)
        k_i_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + (f_offs[None, :] * 2 + 1)
    else:
        # HALF format: [r0, r1, ..., i0, i1, ...]
        k_r_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + f_offs[None, :]
        k_i_ptrs = K_rot_ptr + k_base + n_offs[:, None] * stride_kn + (f_offs[None, :] + half_dim)

    # Cast K to fp32 for tl.sqrt compatibility
    k_r = tl.load(k_r_ptrs, mask=n_mask[:, None] & f_mask[None, :], other=0.0).to(tl.float32)
    k_i = tl.load(k_i_ptrs, mask=n_mask[:, None] & f_mask[None, :], other=0.0).to(tl.float32)

    # === Compute position-independent coefficients ===
    prod_real = q_r[None, :] * k_r + q_i[None, :] * k_i
    prod_imag = q_i[None, :] * k_r - q_r[None, :] * k_i

    A_coef = freq_scale[None, :] * prod_real
    B_coef = freq_scale[None, :] * prod_imag

    # Additive term (MLR - Magnitude Linear Regression)
    k_abs = tl.sqrt(k_r * k_r + k_i * k_i + 1e-8)
    extra_coef = (q_abs[None, :] - q_mean_abs[None, :]) * k_abs * freq_scale[None, :]
    extra_sum = tl.sum(extra_coef, axis=1)

    # === Compute scores using precomputed tables with position correction ===
    max_scores = tl.full([BLOCK_N], value=-1e10, dtype=tl.float32)
    mean_scores = tl.zeros([BLOCK_N], dtype=tl.float32)

    for off_idx in tl.static_range(num_offsets):
        # Load precomputed cos(t*omega) and sin(t*omega) where t = round_start + offset
        cos_base = off_idx * freq_count
        sin_base = off_idx * freq_count

        # NOTE: Cast to fp32 for numerical stability
        cos_t_omega = tl.load(cos_table_ptr + cos_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)  # [BLOCK_F]
        sin_t_omega = tl.load(sin_table_ptr + sin_base + f_offs, mask=f_mask, other=0.0).to(tl.float32)  # [BLOCK_F]

        # Since K_rot already contains RoPE rotation, phase only depends on t.
        # Precomputed table has cos(t*omega), sin(t*omega).
        # The position information from K_rot is already in A_coef and B_coef,
        # so we can use the precomputed trig values directly.

        # Compute base scores
        # Broadcast cos/sin [BLOCK_F] to match A_coef/B_coef [BLOCK_N, BLOCK_F]
        base_scores = tl.sum(A_coef * cos_t_omega[None, :] - B_coef * sin_t_omega[None, :], axis=1)
        combined = base_scores + extra_sum

        max_scores = tl.maximum(max_scores, combined)
        mean_scores = mean_scores + combined

    # Finalize
    if aggregation_mode == 0:
        final_scores = max_scores
    else:
        final_scores = mean_scores / num_offsets

    # Store results
    out_base = batch_idx * (num_heads * seq_len) + head_idx * seq_len
    out_ptrs = scores_ptr + out_base + n_offs
    tl.store(out_ptrs, final_scores, mask=n_mask)


# =============================================================================
# Python Wrapper Functions
# =============================================================================

def speckv_scoring(
    K_rot: torch.Tensor,
    position_indices: Optional[torch.Tensor],
    q_mean_real: torch.Tensor,
    q_mean_imag: torch.Tensor,
    q_abs_mean: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    omega: torch.Tensor,
    offsets: torch.Tensor,
    round_start: int,
    aggregation: str = "max",
    trig_cache: Optional[TrigTableCache] = None,
    rope_style: str = "interleaved",
    disable_mlr: bool = False,
) -> torch.Tensor:
    """
    Python wrapper for Triton scoring kernel.

    Args:
        K_rot: Rotated keys, shape [batch, num_heads, seq_len, head_dim]
        position_indices: DEPRECATED - no longer used in phase calculation.
            Kept for API compatibility. Pass None to avoid GPU memory waste.
            When using K_rot, phase = t*omega + phi_rot (no position needed).
        q_mean_real: Real part of Q mean, shape [num_heads, freq_count]
        q_mean_imag: Imaginary part of Q mean, shape [num_heads, freq_count]
        q_abs_mean: Absolute mean of Q, shape [num_heads, freq_count]
        freq_scale_sq: Frequency scaling squared, shape [num_heads, freq_count]
        omega: Angular frequencies, shape [freq_count]
            Required for both cached and non-cached kernels (cached needs it for position correction)
        offsets: Offset values, shape [num_offsets] (only used when trig_cache is None)
        round_start: Current round start position (scalar)
        aggregation: "max" or "mean" for offset aggregation
        trig_cache: Optional precomputed trig table cache. If provided, uses optimized
            kernel with table lookup instead of on-the-fly trig computation.
        rope_style: RoPE format for K_rot. Options:
            - "interleaved": [r0, i0, r1, i1, ...] (HuggingFace default)
            - "half": [r0, r1, ..., i0, i1, ...] (Qwen models)
        disable_mlr: If True, falls back to PyTorch implementation.
            Triton kernel does not support disable_mlr=True.

    Returns:
        scores: Shape [batch, num_heads, seq_len]
    """
    if disable_mlr:
        raise NotImplementedError(
            "disable_mlr=True is not supported in Triton kernel. "
            "Use compute_scores_pytorch() instead by setting config.use_triton_scoring=False."
        )

    batch_size, num_heads, seq_len, head_dim = K_rot.shape
    freq_count = q_mean_real.shape[1]

    # Validate inputs
    assert K_rot.is_contiguous(), "K_rot must be contiguous"
    assert head_dim % 2 == 0, "head_dim must be even for complex pairing"
    assert head_dim // 2 == freq_count, f"freq_count {freq_count} != head_dim//2 {head_dim//2}"

    # Allocate output
    scores = torch.empty(
        (batch_size, num_heads, seq_len),
        device=K_rot.device,
        dtype=torch.float32
    )

    # Launch configuration
    BLOCK_N = 32
    BLOCK_F = triton.next_power_of_2(freq_count)
    grid = (batch_size * num_heads, triton.cdiv(seq_len, BLOCK_N))
    aggregation_mode = 0 if aggregation == "max" else 1
    rope_style_mode = 0 if rope_style == "interleaved" else 1

    # Handle position_indices - create minimal dummy tensor if None
    # NOTE: position_indices is NOT used in the kernel anymore (phase = t*omega + phi_rot)
    # We still need to pass a valid pointer for kernel signature compatibility
    if position_indices is None:
        # Create minimal dummy tensor (1 element) to satisfy kernel signature
        # The kernel doesn't actually load from this tensor anymore
        position_indices = torch.zeros(1, dtype=torch.int32, device=K_rot.device)
        stride_pb = 0
        stride_pn = 0  # stride=0 means repeated same value, safe since kernel doesn't load
    elif position_indices.ndim == 1:
        position_indices = position_indices.contiguous()
        stride_pb = 0
        stride_pn = 1
    else:
        assert position_indices.shape == (batch_size, seq_len)
        position_indices = position_indices.contiguous()
        stride_pb = position_indices.stride(0)
        stride_pn = position_indices.stride(1)

    if trig_cache is not None:
        # Use optimized kernel with precomputed tables
        cos_table, sin_table = trig_cache.get_trig_values(round_start)
        num_offsets = trig_cache.num_offsets

        speckv_scoring_kernel_cached[grid](
            K_rot,
            position_indices,
            q_mean_real,
            q_mean_imag,
            q_abs_mean,
            freq_scale_sq,
            omega,
            cos_table,
            sin_table,
            round_start,
            scores,
            # Strides
            K_rot.stride(0),
            K_rot.stride(1),
            K_rot.stride(2),
            K_rot.stride(3),
            stride_pb,
            stride_pn,
            # Dimensions
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            freq_count,
            num_offsets,
            aggregation_mode=aggregation_mode,
            rope_style=rope_style_mode,
            BLOCK_N=BLOCK_N,
            BLOCK_F=BLOCK_F,
        )
    else:
        # Use original kernel with on-the-fly trig computation
        num_offsets = offsets.shape[0]

        speckv_scoring_kernel[grid](
            K_rot,
            position_indices,
            q_mean_real,
            q_mean_imag,
            q_abs_mean,
            freq_scale_sq,
            omega,
            offsets,
            round_start,
            scores,
            # Strides
            K_rot.stride(0),
            K_rot.stride(1),
            K_rot.stride(2),
            K_rot.stride(3),
            stride_pb,
            stride_pn,
            # Dimensions
            batch_size,
            num_heads,
            seq_len,
            head_dim,
            freq_count,
            num_offsets,
            aggregation_mode=aggregation_mode,
            rope_style=rope_style_mode,
            BLOCK_N=BLOCK_N,
            BLOCK_F=BLOCK_F,
        )

    return scores
