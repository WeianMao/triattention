"""
TriAttention MLX — Native Apple Silicon Port
=============================================
Ports the core TriAttention trigonometric KV compression algorithm to MLX.

Key insight from the paper:
  Pre-RoPE Q/K vectors concentrate around fixed centers across positions.
  These centers determine "distance preferences" via a trigonometric series.
  → Score keys by how well their positions match the center's distance preference.
  → No need for post-RoPE queries (which are unstable due to rotation).

This MLX port integrates with mlx-lm's KV cache system:
  - Patches the model's forward pass
  - Compresses KV cache when it exceeds kv_budget
  - Uses pre-RoPE frequency statistics for scoring
  - Works with Apple Silicon unified memory (M1/M2/M3/M4 Max)

References:
  Paper:  https://arxiv.org/abs/2604.04921
  Origin: https://github.com/WeianMao/triattention
  Fork:   https://github.com/DeadByDawn101/triattention

Contributed by: @DeadByDawn101 (MLX port) — April 2026
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
import numpy as np


# ─────────────────────────── Config ──────────────────────────────────────────

@dataclass
class TriAttentionMLXConfig:
    """Configuration for TriAttention MLX compression."""
    
    stats_path: Optional[Path] = None
    """Path to precomputed frequency statistics (.npz). 
    If None, runs without stats (uses norm-only scoring — less accurate)."""
    
    kv_budget: int = 2048
    """Maximum number of KV pairs to keep per layer."""
    
    divide_length: int = 128
    """Compress every N decode steps."""
    
    score_aggregation: str = "mean"
    """How to aggregate scores across heads: 'mean' or 'max'."""
    
    prefill_pin: bool = True
    """Always preserve prefill (prompt) tokens — never evict them."""
    
    disable_trig: bool = False
    """If True, use norm-only scoring (no trigonometric term). Faster but less accurate."""
    
    disable_mlr: bool = False
    """If True, skip the mean-log-ratio extra term."""
    
    head_dim: int = 256
    """Attention head dimension (model-specific)."""
    
    rope_theta: float = 10000.0
    """RoPE base frequency."""


# ─────────────────────────── Frequency Stats ─────────────────────────────────

@dataclass
class HeadFrequencyStats:
    """Pre-RoPE frequency statistics for one attention head."""
    q_mean_real: mx.array     # [head_dim/2] — real part of Q center
    q_mean_imag: mx.array     # [head_dim/2] — imag part of Q center  
    q_abs_mean: mx.array      # [head_dim/2] — |Q| mean for norm term
    layer_idx: int
    head_idx: int


def load_stats(stats_path: Path) -> Dict[Tuple[int, int], HeadFrequencyStats]:
    """Load precomputed frequency statistics from .npz file."""
    data = np.load(str(stats_path))
    stats = {}
    
    # Expected format: {layer}_{head}_q_mean_real, {layer}_{head}_q_mean_imag, {layer}_{head}_q_abs_mean
    keys = set()
    for k in data.files:
        parts = k.split("_")
        if len(parts) >= 4 and parts[0] == "l" and parts[2] == "h":
            keys.add((int(parts[1]), int(parts[3])))
    
    for layer_idx, head_idx in keys:
        prefix = f"l_{layer_idx}_h_{head_idx}"
        stats[(layer_idx, head_idx)] = HeadFrequencyStats(
            q_mean_real=mx.array(data[f"{prefix}_q_mean_real"]),
            q_mean_imag=mx.array(data[f"{prefix}_q_mean_imag"]),
            q_abs_mean=mx.array(data[f"{prefix}_q_abs_mean"]),
            layer_idx=layer_idx,
            head_idx=head_idx,
        )
    
    return stats


# ─────────────────────────── Scoring Engine ──────────────────────────────────

def build_inv_freq(head_dim: int, rope_theta: float) -> mx.array:
    """Build inverse frequencies for RoPE."""
    i = mx.arange(0, head_dim // 2, dtype=mx.float32)
    return 1.0 / (rope_theta ** (2 * i / head_dim))


def invert_rope_mlx(k: mx.array, positions: mx.array, inv_freq: mx.array) -> mx.array:
    """
    Remove RoPE rotation from key vectors to get pre-RoPE representation.
    
    k: [seq_len, head_dim]
    positions: [seq_len]
    inv_freq: [head_dim/2]
    
    Returns: k_pre_rope [seq_len, head_dim]
    """
    # Build rotation angles: [seq_len, head_dim/2]
    theta = mx.outer(positions.astype(mx.float32), inv_freq)  # [seq_len, head_dim/2]
    
    cos_t = mx.cos(theta)  # [seq_len, head_dim/2]
    sin_t = mx.sin(theta)  # [seq_len, head_dim/2]
    
    # Split k into two halves (standard RoPE layout)
    k = k.astype(mx.float32)
    k1, k2 = k[..., :k.shape[-1]//2], k[..., k.shape[-1]//2:]
    
    # Inverse rotation: multiply by transpose of rotation matrix
    # R^-1 = R^T: [cos, sin; -sin, cos]^T = [cos, -sin; sin, cos]
    k1_orig = k1 * cos_t + k2 * sin_t
    k2_orig = -k1 * sin_t + k2 * cos_t
    
    return mx.concatenate([k1_orig, k2_orig], axis=-1)


def score_keys_trig(
    k_pre: mx.array,
    positions: mx.array,
    stats: HeadFrequencyStats,
    inv_freq: mx.array,
    absolute_position: int,
    disable_trig: bool = False,
    disable_mlr: bool = False,
) -> mx.array:
    """
    Score keys using trigonometric frequency analysis.
    
    Core TriAttention algorithm:
      score(key_i) = Σ_f amp_f * cos(omega_f * (pos_i - current_pos) + phi_f)
                   + alpha * ||k_pre_i||
    
    k_pre: [seq_len, head_dim] — pre-RoPE keys
    positions: [seq_len] — absolute positions of each key
    stats: frequency statistics for this head
    inv_freq: [head_dim/2]
    absolute_position: current decode position
    
    Returns: scores [seq_len]
    """
    seq_len = k_pre.shape[0]
    
    # Norm term: ||k_pre|| (always active unless both disabled)
    k_norms = mx.sqrt(mx.sum(k_pre ** 2, axis=-1) + 1e-8)  # [seq_len]
    
    if disable_trig:
        return k_norms
    
    # Build trigonometric score
    # Amplitude and phase from Q center statistics
    q_complex_real = stats.q_mean_real  # [head_dim/2]
    q_complex_imag = stats.q_mean_imag  # [head_dim/2]
    
    # amp = |Q_center| per frequency
    amp = mx.sqrt(q_complex_real ** 2 + q_complex_imag ** 2 + 1e-8)  # [head_dim/2]
    
    # phi = angle of Q_center per frequency
    phi = mx.arctan2(q_complex_imag, q_complex_real)  # [head_dim/2]
    
    # Distance offsets: current_pos - key_pos
    offsets = float(absolute_position) - positions.astype(mx.float32)  # [seq_len]
    
    # Phase at each key position: omega * offset + phi
    # [seq_len, head_dim/2] = outer product
    phase = mx.outer(offsets, inv_freq) + phi[None, :]  # [seq_len, head_dim/2]
    
    # Weighted cosine sum: amp * cos(phase)
    trig_scores = mx.sum(amp[None, :] * mx.cos(phase), axis=-1)  # [seq_len]
    
    # Frequency scaling: weight higher-frequency components less
    freq_scale = inv_freq / (inv_freq.max() + 1e-8)
    freq_scale_sq = freq_scale ** 2
    trig_scores_scaled = mx.sum(amp[None, :] * freq_scale_sq[None, :] * mx.cos(phase), axis=-1)
    
    # MLR extra term: correlation between k_pre norm and Q abs mean
    if not disable_mlr:
        q_abs = stats.q_abs_mean  # [head_dim/2]
        k_abs = mx.abs(k_pre[..., :k_pre.shape[-1]//2]) + mx.abs(k_pre[..., k_pre.shape[-1]//2:])
        k_abs = k_abs.reshape(seq_len, -1, 2).mean(axis=-1)  # [seq_len, head_dim/2]
        # Clip to avoid log(0)
        ratio = mx.clip(k_abs / (q_abs[None, :] + 1e-8), 1e-6, 1e6)
        mlr = mx.sum(q_abs[None, :] * mx.log(ratio + 1e-8), axis=-1)  # [seq_len]
        return trig_scores_scaled + 0.1 * mlr + 0.01 * k_norms
    
    return trig_scores_scaled + 0.01 * k_norms


# ─────────────────────────── Compressor ──────────────────────────────────────

class TriAttentionMLX:
    """
    TriAttention KV cache compressor for MLX models.
    
    Integrates with mlx-lm's KV cache by hooking into the model's decode loop.
    """
    
    def __init__(self, config: TriAttentionMLXConfig):
        self.config = config
        self.inv_freq = build_inv_freq(config.head_dim, config.rope_theta)
        
        # Load stats if provided
        self.stats: Dict[Tuple[int, int], HeadFrequencyStats] = {}
        if config.stats_path and Path(config.stats_path).exists():
            self.stats = load_stats(Path(config.stats_path))
            print(f"[TriAttention MLX] Loaded stats for {len(self.stats)} heads")
        else:
            print("[TriAttention MLX] No stats file — using norm-only scoring")
        
        # State
        self.cache_positions: List[int] = []
        self.absolute_position: int = 0
        self.prefix_length: int = 0
        self.step_count: int = 0
    
    def reset(self):
        """Reset for new generation."""
        self.cache_positions = []
        self.absolute_position = 0
        self.prefix_length = 0
        self.step_count = 0
    
    def should_compress(self, cache_len: int) -> bool:
        """Check if we should compress now."""
        effective = cache_len
        if self.config.prefill_pin:
            effective = max(0, cache_len - self.prefix_length)
        return (
            effective >= self.config.kv_budget
            and self.step_count > 0
            and (self.step_count % self.config.divide_length == 0)
        )
    
    def score_layer(
        self,
        keys: mx.array,
        positions: mx.array,
        layer_idx: int,
    ) -> mx.array:
        """
        Score all keys in one layer.
        
        keys: [num_heads, seq_len, head_dim] (pre-RoPE or post-RoPE)
        positions: [seq_len]
        
        Returns: scores [seq_len]
        """
        num_heads = keys.shape[0]
        all_scores = []
        
        for head_idx in range(num_heads):
            k_head = keys[head_idx]  # [seq_len, head_dim]
            
            if (layer_idx, head_idx) in self.stats:
                stats = self.stats[(layer_idx, head_idx)]
                # Invert RoPE first (keys in cache are post-RoPE)
                k_pre = invert_rope_mlx(k_head, positions, self.inv_freq)
                scores = score_keys_trig(
                    k_pre, positions, stats, self.inv_freq,
                    self.absolute_position,
                    disable_trig=self.config.disable_trig,
                    disable_mlr=self.config.disable_mlr,
                )
            else:
                # Fallback: norm-only scoring
                scores = mx.sqrt(mx.sum(k_head.astype(mx.float32) ** 2, axis=-1) + 1e-8)
            
            all_scores.append(scores)
        
        # Aggregate across heads
        stacked = mx.stack(all_scores, axis=0)  # [num_heads, seq_len]
        if self.config.score_aggregation == "max":
            return stacked.max(axis=0)
        return stacked.mean(axis=0)
    
    def compress_cache(
        self,
        kv_cache: List[Tuple[mx.array, mx.array]],
    ) -> List[Tuple[mx.array, mx.array]]:
        """
        Compress KV cache by evicting low-importance tokens.
        
        kv_cache: list of (keys, values) per layer
                  keys: [batch, num_heads, seq_len, head_dim]
        
        Returns: compressed kv_cache
        """
        if not kv_cache:
            return kv_cache
        
        seq_len = kv_cache[0][0].shape[2]
        positions = mx.array(self.cache_positions[:seq_len], dtype=mx.int32)
        
        # Compute scores across all layers
        all_scores = []
        for layer_idx, (keys, _) in enumerate(kv_cache):
            # keys: [1, num_heads, seq_len, head_dim]
            layer_keys = keys[0]  # [num_heads, seq_len, head_dim]
            layer_scores = self.score_layer(layer_keys, positions, layer_idx)
            all_scores.append(layer_scores)
        
        # Aggregate across layers
        score_matrix = mx.stack(all_scores, axis=0)  # [num_layers, seq_len]
        global_scores = score_matrix.mean(axis=0)  # [seq_len]
        
        # Always keep prefill tokens
        prefix = self.prefix_length if self.config.prefill_pin else 0
        decode_scores = global_scores[prefix:]
        decode_len = decode_scores.shape[0]
        
        decode_budget = max(0, self.config.kv_budget - prefix)
        
        if decode_len <= decode_budget:
            return kv_cache  # Nothing to compress
        
        # Top-k selection on decode tokens
        keep_k = min(decode_budget, decode_len)
        top_indices = mx.argsort(-decode_scores)[:keep_k]  # descending
        top_indices_sorted = mx.sort(top_indices)
        
        # Add back prefill offset
        decode_keep_abs = top_indices_sorted + prefix
        
        # Combine: all prefill + selected decode
        prefill_indices = mx.arange(prefix)
        keep_indices = mx.concatenate([prefill_indices, decode_keep_abs])
        
        # Apply compression to each layer
        new_cache = []
        for keys, values in kv_cache:
            # keys: [1, num_heads, seq_len, head_dim]
            k_new = keys[:, :, keep_indices, :]
            v_new = values[:, :, keep_indices, :]
            new_cache.append((k_new, v_new))
        
        # Update position tracking
        keep_list = keep_indices.tolist()
        self.cache_positions = [self.cache_positions[i] for i in keep_list]
        
        return new_cache


# ─────────────────────────── Model Patcher ───────────────────────────────────

def apply_triattention_mlx(
    model,
    stats_path: Optional[str] = None,
    kv_budget: int = 2048,
    divide_length: int = 128,
    score_aggregation: str = "mean",
    prefill_pin: bool = True,
    disable_trig: bool = False,
    disable_mlr: bool = False,
    rope_theta: float = 10000.0,
) -> TriAttentionMLX:
    """
    Apply TriAttention KV compression to an mlx-lm model.
    
    Usage:
        model, tokenizer = mlx_lm.load("deadbydawn101/gemma-4-E4B-Agentic-Opus-Reasoning-GeminiCLI-mlx-4bit")
        compressor = apply_triattention_mlx(
            model,
            stats_path="triattention/calibration/gemma4_e4b_stats.npz",
            kv_budget=2048,
        )
        
        # Then in your generation loop, call compressor.step(kv_cache) after each decode step
    
    Args:
        model: mlx-lm model instance
        stats_path: Path to precomputed .npz stats (optional but recommended)
        kv_budget: Max KV pairs to keep per layer (default 2048)
        divide_length: Compress every N steps (default 128)
        score_aggregation: 'mean' or 'max' across heads
        prefill_pin: Keep all prompt tokens (default True)
        disable_trig: Use norm-only scoring (faster, less accurate)
        disable_mlr: Skip mean-log-ratio term
        rope_theta: RoPE base frequency (match your model)
    
    Returns:
        TriAttentionMLX compressor instance
    """
    # Detect head_dim from model
    head_dim = 256  # default for Gemma 4
    try:
        if hasattr(model, "args"):
            head_dim = getattr(model.args, "head_dim", 
                      getattr(model.args, "hidden_size", 2048) // 
                      getattr(model.args, "num_attention_heads", 8))
            rope_theta = float(getattr(model.args, "rope_theta", rope_theta))
    except Exception:
        pass
    
    config = TriAttentionMLXConfig(
        stats_path=Path(stats_path) if stats_path else None,
        kv_budget=kv_budget,
        divide_length=divide_length,
        score_aggregation=score_aggregation,
        prefill_pin=prefill_pin,
        disable_trig=disable_trig,
        disable_mlr=disable_mlr,
        head_dim=head_dim,
        rope_theta=rope_theta,
    )
    
    compressor = TriAttentionMLX(config)
    
    # Attach to model for easy access
    model._triattention_mlx = compressor
    
    print(
        f"[TriAttention MLX] Applied — budget={kv_budget}, "
        f"divide_length={divide_length}, head_dim={head_dim}, "
        f"stats={'loaded' if compressor.stats else 'none (norm-only)'}"
    )
    
    return compressor


# ─────────────────────────── Generation Loop Helper ──────────────────────────

def triattention_generate_step(
    compressor: TriAttentionMLX,
    kv_cache: List[Tuple[mx.array, mx.array]],
    is_prefill: bool = False,
    current_position: int = 0,
) -> List[Tuple[mx.array, mx.array]]:
    """
    Call this after each decode step to manage KV compression.
    
    Args:
        compressor: TriAttentionMLX instance
        kv_cache: Current KV cache from model
        is_prefill: True for first (prompt) forward pass
        current_position: Current absolute position in sequence
    
    Returns:
        (possibly compressed) KV cache
    """
    if is_prefill:
        seq_len = kv_cache[0][0].shape[2] if kv_cache else 0
        compressor.reset()
        compressor.cache_positions = list(range(seq_len))
        compressor.absolute_position = seq_len
        compressor.prefix_length = seq_len
        return kv_cache
    
    # Track new position
    compressor.cache_positions.append(current_position)
    compressor.absolute_position = current_position + 1
    compressor.step_count += 1
    
    # Compress if needed
    cache_len = kv_cache[0][0].shape[2] if kv_cache else 0
    if compressor.should_compress(cache_len):
        kv_cache = compressor.compress_cache(kv_cache)
    
    return kv_cache
