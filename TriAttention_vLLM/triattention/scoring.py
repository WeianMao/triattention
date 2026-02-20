"""Scoring logic for TriAttention KV cache compression.

This module provides the Python wrapper for Triton scoring kernels.
The actual Triton kernel implementation is in kernels/triton_scoring.py.

Design Alignment:
- Follows SpeckV scoring formula from R-KV
- Phase 1: Implements Triton-accelerated scoring
- Supports per_head and per_layer scoring modes
"""
from typing import Dict, Optional

import torch

from .config import TriAttentionConfig


def compute_scores(
    key_states: torch.Tensor,
    cache_positions: Optional[torch.Tensor],
    head_stats: Dict[str, torch.Tensor],
    omega: torch.Tensor,
    offsets: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    config: TriAttentionConfig,
    round_start: Optional[int] = None,
    trig_cache=None,
) -> torch.Tensor:
    """Compute importance scores for KV tokens.

    This is the main entry point for scoring. It dispatches to either:
    - Triton kernel (Phase 1 default)
    - PyTorch fallback (for debugging/comparison)

    Args:
        key_states: K cache [batch, num_kv_heads, seq_len, head_dim]
        cache_positions: DEPRECATED - only used to infer round_start if not provided.
            Pass None to avoid GPU memory allocation. Set round_start explicitly instead.
        head_stats: Per-head frequency statistics
            - 'q_mean_complex': [num_kv_heads, freq_count, 2] (real, imag)
            - 'freq_scale_sq': [num_kv_heads, freq_count]
            - 'extra_coef': [num_kv_heads, freq_count] (optional, for MLR term)
        omega: Angular frequencies [freq_count]
        offsets: Scoring offsets [num_offsets]
        freq_scale_sq: Frequency scaling factors [num_kv_heads, freq_count]
        config: TriAttention configuration
        round_start: Current round start position. If None, inferred from cache_positions.max()

    Returns:
        Importance scores:
        - [batch, num_kv_heads, seq_len] for per_head mode
        - [batch, seq_len] for per_layer mode
    """
    if config.use_triton_scoring and not config.disable_mlr:
        return compute_scores_triton(
            key_states=key_states,
            cache_positions=cache_positions,
            head_stats=head_stats,
            omega=omega,
            offsets=offsets,
            freq_scale_sq=freq_scale_sq,
            config=config,
            round_start=round_start,
            trig_cache=trig_cache,
        )
    else:
        return compute_scores_pytorch(
            key_states=key_states,
            cache_positions=cache_positions,
            head_stats=head_stats,
            omega=omega,
            offsets=offsets,
            freq_scale_sq=freq_scale_sq,
            config=config,
            round_start=round_start,
        )


def compute_scores_triton(
    key_states: torch.Tensor,
    cache_positions: Optional[torch.Tensor],
    head_stats: Dict[str, torch.Tensor],
    omega: torch.Tensor,
    offsets: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    config: TriAttentionConfig,
    round_start: Optional[int] = None,
    trig_cache=None,
) -> torch.Tensor:
    """Compute scores using Triton kernel.

    Args:
        key_states: K cache [batch, num_kv_heads, seq_len, head_dim]
            NOTE: In vLLM, keys are stored AFTER RoPE rotation (K_rot)
        cache_positions: DEPRECATED - not used in scoring anymore.
            Only used to infer round_start if round_start is None.
            Pass None to avoid GPU memory waste.
        head_stats: Per-head frequency statistics
            - 'q_mean_complex': [num_kv_heads, freq_count, 2] (real, imag)
            - 'q_abs_mean': [num_kv_heads, freq_count]
        omega: Angular frequencies [freq_count]
        offsets: Scoring offsets [num_offsets]
        freq_scale_sq: Frequency scaling factors [num_kv_heads, freq_count]
        config: TriAttention configuration
        round_start: Current round start position. If None, inferred from cache_positions.

    Returns:
        Importance scores [batch, num_kv_heads, seq_len] or [batch, seq_len]
    """
    from .kernels.triton_scoring import speckv_scoring

    batch_size, num_kv_heads, seq_len, head_dim = key_states.shape
    freq_count = head_dim // 2

    # Extract Q statistics from head_stats
    q_mean_complex = head_stats['q_mean_complex']  # [num_kv_heads, freq_count, 2]
    q_mean_real = q_mean_complex[..., 0].contiguous()  # [num_kv_heads, freq_count]
    q_mean_imag = q_mean_complex[..., 1].contiguous()  # [num_kv_heads, freq_count]

    # q_abs_mean: either from stats or compute from q_mean_complex
    if 'q_abs_mean' in head_stats:
        q_abs_mean = head_stats['q_abs_mean'].contiguous()
    else:
        q_abs_mean = torch.sqrt(q_mean_real ** 2 + q_mean_imag ** 2 + 1e-8)

    # Determine round_start
    if round_start is None:
        if cache_positions is not None:
            round_start = cache_positions.max().item()
        else:
            # Fallback: use seq_len - 1 as round_start
            round_start = seq_len - 1

    # Keep K as-is to avoid an extra full-tensor copy on scoring hot path.
    K_rot = key_states
    freq_scale_sq_input = freq_scale_sq.contiguous().to(dtype=key_states.dtype)
    omega_input = omega.contiguous().to(dtype=key_states.dtype)
    offsets_input = offsets.contiguous().to(dtype=key_states.dtype)

    # NOTE: position_indices is NOT used in Triton kernel anymore
    # Phase calculation uses t*omega + phi_rot, no per-token positions needed
    # Pass None to avoid GPU memory waste
    position_indices = None

    active_trig_cache = None
    active_trig_values = None
    if trig_cache is not None:
        divide_length = int(getattr(config, "divide_length", 0))
        max_positions = int(getattr(trig_cache, "num_positions", 0))
        max_round_start = divide_length * max_positions if divide_length > 0 else 0
        if (
            divide_length > 0
            and round_start >= divide_length
            and max_round_start >= divide_length
        ):
            base_round_start = (int(round_start) // divide_length) * divide_length
            if divide_length <= base_round_start <= max_round_start:
                residual = int(round_start) - base_round_start
                if residual == 0:
                    active_trig_cache = trig_cache
                else:
                    cos_base, sin_base = trig_cache.get_trig_values(base_round_start)
                    omega_fp32 = omega_input.to(dtype=torch.float32)
                    residual_phase = float(residual) * omega_fp32
                    cos_residual = torch.cos(residual_phase).unsqueeze(0)
                    sin_residual = torch.sin(residual_phase).unsqueeze(0)
                    # cos(a+b)=cos(a)cos(b)-sin(a)sin(b), sin(a+b)=sin(a)cos(b)+cos(a)sin(b)
                    cos_shifted = (
                        cos_base.to(dtype=torch.float32) * cos_residual
                        - sin_base.to(dtype=torch.float32) * sin_residual
                    ).contiguous()
                    sin_shifted = (
                        sin_base.to(dtype=torch.float32) * cos_residual
                        + cos_base.to(dtype=torch.float32) * sin_residual
                    ).contiguous()
                    active_trig_values = (cos_shifted, sin_shifted)

    # Call Triton kernel
    scores = speckv_scoring(
        K_rot=K_rot,
        position_indices=position_indices,
        q_mean_real=q_mean_real.to(dtype=key_states.dtype),
        q_mean_imag=q_mean_imag.to(dtype=key_states.dtype),
        q_abs_mean=q_abs_mean.to(dtype=key_states.dtype),
        freq_scale_sq=freq_scale_sq_input,
        omega=omega_input,
        offsets=offsets_input,
        round_start=round_start,
        aggregation=config.score_aggregation,
        disable_mlr=config.disable_mlr,
        trig_cache=active_trig_cache,
        trig_values=active_trig_values,
    )  # [batch, num_kv_heads, seq_len]

    # For per_layer mode, aggregate across heads
    if config.pruning_mode == "per_layer":
        scores = scores.mean(dim=1)  # [batch, seq_len]

    return scores.to(dtype=config.topk_dtype)


def compute_scores_pytorch(
    key_states: torch.Tensor,
    cache_positions: Optional[torch.Tensor],
    head_stats: Dict[str, torch.Tensor],
    omega: torch.Tensor,
    offsets: torch.Tensor,
    freq_scale_sq: torch.Tensor,
    config: TriAttentionConfig,
    round_start: Optional[int] = None,
) -> torch.Tensor:
    """Compute scores using PyTorch (fallback/reference implementation).

    This implements the SpeckV scoring formula in pure PyTorch for:
    1. Reference correctness checking
    2. Debugging Triton kernel
    3. Fallback when Triton is unavailable

    Scoring Formula:
    Since K is stored after RoPE rotation (K_rot = K_unrot * e^{i*p*omega}),
    the position p is already "baked in" to K_rot. Therefore:

        score = sum_over_frequencies(
            freq_scale[f]^2 * (A * cos(t*omega[f]) - B * sin(t*omega[f]))
        ) + extra_term

    Where:
        - t = round_start + offset (query position)
        - A = Re(Q_mean * conj(K_rot))
        - B = Im(Q_mean * conj(K_rot))
        - phi = arg(Q_mean * conj(K_rot)) is used to adjust the phase
        - extra_term = position-independent magnitude-LR term

    The key insight: phase only depends on t, NOT on key position p.
    See docs/RKV_EQUIVALENCE_FIX.md for full mathematical derivation.

    Args:
        key_states: K cache [batch, num_kv_heads, seq_len, head_dim]
        cache_positions: DEPRECATED - only used to infer round_start if not provided.
        head_stats: Per-head frequency statistics
        omega: Angular frequencies [freq_count]
        offsets: Scoring offsets [num_offsets]
        freq_scale_sq: Frequency scaling factors [num_kv_heads, freq_count]
        config: TriAttention configuration
        round_start: Current round start position. If None, inferred from cache_positions.

    Returns:
        Importance scores [batch, num_kv_heads, seq_len] or [batch, seq_len]
    """
    batch_size, num_kv_heads, seq_len, head_dim = key_states.shape
    freq_count = head_dim // 2

    # Extract Q statistics
    q_mean_complex = head_stats['q_mean_complex']  # [num_kv_heads, freq_count, 2]
    q_mean_real = q_mean_complex[..., 0]  # [num_kv_heads, freq_count]
    q_mean_imag = q_mean_complex[..., 1]  # [num_kv_heads, freq_count]

    # Compute |Q_mean_complex| from complex components
    q_mean_abs = torch.sqrt(q_mean_real ** 2 + q_mean_imag ** 2 + 1e-8)  # [num_kv_heads, freq_count]

    # Get q_abs_mean (mean of |Q_complex| across queries)
    if 'q_abs_mean' in head_stats:
        q_abs_mean = head_stats['q_abs_mean']  # [num_kv_heads, freq_count]
    else:
        # If not provided, assume it equals q_mean_abs (no MLR effect)
        q_abs_mean = q_mean_abs

    # Convert K to complex representation (assuming RoPE 'half' style)
    # K shape: [batch, num_kv_heads, seq_len, head_dim]
    # Split into pairs: [batch, num_kv_heads, seq_len, freq_count, 2]
    k_pairs = key_states.reshape(batch_size, num_kv_heads, seq_len, freq_count, 2)
    k_real = k_pairs[..., 0]  # [batch, num_kv_heads, seq_len, freq_count]
    k_imag = k_pairs[..., 1]  # [batch, num_kv_heads, seq_len, freq_count]
    k_abs = torch.sqrt(k_real ** 2 + k_imag ** 2)  # [batch, num_kv_heads, seq_len, freq_count]

    # Compute amplitude term: |Q_mean| * |K|
    # Expand q_abs_mean: [num_kv_heads, freq_count] -> [batch, num_kv_heads, seq_len, freq_count]
    q_abs_expanded = q_abs_mean.unsqueeze(0).unsqueeze(2)
    amp = q_abs_expanded * k_abs  # [batch, num_kv_heads, seq_len, freq_count]

    # Compute phase difference: phi = atan2(q_imag, q_real) - atan2(k_imag, k_real)
    # For numerical stability, use complex division formula
    q_real_expanded = q_mean_real.unsqueeze(0).unsqueeze(2)
    q_imag_expanded = q_mean_imag.unsqueeze(0).unsqueeze(2)

    # Phase = atan2(q_imag * k_real - q_real * k_imag, q_real * k_real + q_imag * k_imag)
    numerator = q_imag_expanded * k_real - q_real_expanded * k_imag
    denominator = q_real_expanded * k_real + q_imag_expanded * k_imag
    phi = torch.atan2(numerator, denominator)  # [batch, num_kv_heads, seq_len, freq_count]

    # Determine round_start
    if round_start is None:
        if cache_positions is not None:
            if cache_positions.ndim == 1:
                round_start = cache_positions.max().item()
            else:
                round_start = cache_positions.max().item()
        else:
            # Fallback: use seq_len - 1
            round_start = seq_len - 1

    # Initialize scores
    num_offsets = offsets.shape[0]
    scores_per_offset = []

    # Compute scores for each offset
    for offset_val in offsets:
        # Since K already contains RoPE rotation (K_rot = K_unrot * e^{i*p*omega}),
        # phase only depends on query position t, not on key position p.
        # Correct formula: phase = t * omega + phi
        # where t = round_start + offset (query position)
        t = round_start + offset_val.item()

        # omega: [freq_count] -> [1, 1, 1, freq_count]
        omega_expanded = omega.unsqueeze(0).unsqueeze(0).unsqueeze(0)

        # phase = t * omega + phi
        # t is scalar, omega: [1, 1, 1, freq_count]
        # phi: [batch, num_kv_heads, seq_len, freq_count]
        phase = t * omega_expanded + phi  # [batch, num_kv_heads, seq_len, freq_count]

        # Position-dependent term: amp * freq_scale_sq * cos(phase)
        # freq_scale_sq: [num_kv_heads, freq_count] -> [1, num_kv_heads, 1, freq_count]
        freq_scale_expanded = freq_scale_sq.unsqueeze(0).unsqueeze(2)

        if not config.disable_trig:
            position_term = amp * freq_scale_expanded * torch.cos(phase)
        else:
            # If trigonometric term disabled, use only magnitude
            position_term = amp * freq_scale_expanded

        # Sum over frequencies
        score_offset = position_term.sum(dim=-1)  # [batch, num_kv_heads, seq_len]
        scores_per_offset.append(score_offset)

    # Aggregate scores across offsets
    if config.score_aggregation == "mean":
        scores = torch.stack(scores_per_offset, dim=0).mean(dim=0)
    elif config.score_aggregation == "max":
        scores = torch.stack(scores_per_offset, dim=0).max(dim=0).values
    else:
        raise ValueError(f"Unsupported score_aggregation: {config.score_aggregation}")

    # Add position-independent term (MLR - Magnitude Linear Regression)
    # Formula matches R-KV: extra = (q_abs_mean - q_mean_abs) * k_abs * freq_scale_sq
    # If disable_mlr=True, use simplified version: extra = q_abs_mean * k_abs * freq_scale_sq
    if config.disable_mlr:
        # Simplified version: only magnitude product
        extra_coef = q_abs_mean  # [num_kv_heads, freq_count]
    else:
        # MLR version: difference term captures magnitude variation
        extra_coef = q_abs_mean - q_mean_abs  # [num_kv_heads, freq_count]

    # Compute extra term: sum over frequencies
    extra_coef_expanded = extra_coef.unsqueeze(0).unsqueeze(2)  # [1, num_kv_heads, 1, freq_count]
    extra_term = (k_abs * extra_coef_expanded * freq_scale_expanded).sum(dim=-1)  # [batch, num_kv_heads, seq_len]
    scores = scores + extra_term

    # For per_layer mode, aggregate across heads
    if config.pruning_mode == "per_layer":
        scores = scores.mean(dim=1)  # [batch, seq_len]

    return scores
