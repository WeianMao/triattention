"""Scoring utilities for sglang TriAttention integration.

Provides two capabilities that the main scheduler_hooks.py scoring path
needs but does not implement inline:

1. **TrigCache integration** (007 report section 8-7): the shared
   ``TriAttentionCompressor`` already creates and owns a
   ``TrigTableCache`` during ``_lazy_init()``.  When
   ``scheduler_hooks._compress_single_request`` calls
   ``compressor._compute_scores()``, the trig cache is used
   automatically.  This module provides :func:`get_trig_cache` as a
   helper for any future code path that bypasses the compressor
   (e.g., a standalone scoring utility) and needs the cache reference.

2. **Chunked scoring** (007 report section 8-8): for long sequences
   where ``effective_len > SCORE_CHUNK_MAX_TOKENS``, gathering the
   entire KV tensor onto the GPU at once can cause OOM.
   :func:`chunked_score` implements chunked gather + score + concat,
   matching vLLM's ``selector_hf.py`` semantics.

Usage by scheduler_hooks.py (after controller integration)
----------------------------------------------------------
Replace the inner scoring loop in ``_compress_single_request`` with::

    from triattention.sglang.scoring_utils import chunked_score

    aggregated_scores = chunked_score(
        compressor=compressor,
        k_buffers=k_buffers,
        req_to_token=req_to_token,
        req_pool_idx=req_pool_idx,
        effective_len=effective_len,
        available_layers=available_layers,
        aggregation_mode=aggregation_mode,
        score_chunk_max_tokens=rc.score_chunk_max_tokens,
    )

NOTE: scheduler_hooks.py integration is deferred -- group A agent is
currently modifying that file.  The controller should integrate after
group A's changes are complete.
"""

from __future__ import annotations

import logging
import math
from typing import TYPE_CHECKING, List, Optional

import torch

if TYPE_CHECKING:
    from triattention.vllm.core.compressor import TriAttentionCompressor
    from triattention.sglang.stats_loader import StatsBundle

logger = logging.getLogger(__name__)

# Default chunk size matching vLLM's TriAttentionRuntimeConfig default.
DEFAULT_SCORE_CHUNK_MAX_TOKENS: int = 4096


# ---------------------------------------------------------------------------
# 0. HF-aligned per-attention-head scoring with GQA max aggregation
# ---------------------------------------------------------------------------


def score_layer_hf_aligned(
    layer_k: torch.Tensor,
    layer_idx: int,
    stats_bundle: "StatsBundle",
    compressor: "TriAttentionCompressor",
    tp_rank: int = 0,
    tp_size: int = 1,
) -> torch.Tensor:
    """Score a single layer's KV tokens at attention-head granularity.

    This function replicates the HF reference scoring path:
      1. Expand K from ``[1, num_kv_heads_local, S, D]`` to
         ``[1, num_attention_heads_local, S, D]`` via ``repeat_interleave``.
      2. Score each attention head independently using per-attention-head
         stats from *stats_bundle*, sliced to the local TP shard.
      3. Aggregate within each GQA group using **max** (not mean),
         matching HF's ``_select_per_head_independent``.

    Parameters
    ----------
    layer_k : Tensor
        Dense key tensor ``[1, num_kv_heads_local, seq_len, head_dim]``.
        Keys are stored AFTER RoPE rotation (K_rot).  With TP>1,
        ``num_kv_heads_local`` is the per-shard count.
    layer_idx : int
        Layer index for stats lookup.
    stats_bundle : StatsBundle
        Loaded at full attention-head granularity (e.g. 64 heads).
    compressor : TriAttentionCompressor
        Used for omega, offsets, round_start, and scoring config.
    tp_rank : int
        Tensor-parallel rank of this shard (0-indexed).
    tp_size : int
        Total number of tensor-parallel shards.

    Returns
    -------
    Tensor
        Scores ``[1, num_kv_heads_local, seq_len]`` after GQA max aggregation.
    """
    from triattention.vllm.core.scoring import compute_scores

    num_kv_heads_local = layer_k.shape[1]
    gqa_group_size = stats_bundle.gqa_group_size
    num_attention_heads_total = stats_bundle.num_attention_heads

    # TP sharding: compute which attention heads belong to this shard.
    num_attention_heads_local = num_attention_heads_total // tp_size
    head_start = tp_rank * num_attention_heads_local
    head_end = head_start + num_attention_heads_local

    # Step 1: Expand K to local attention-head granularity.
    # [1, num_kv_heads_local, S, D] -> [1, num_attention_heads_local, S, D]
    layer_k_expanded = layer_k.repeat_interleave(gqa_group_size, dim=1)

    # Step 2: Build per-attention-head stats for this shard.
    layer_stats_full = stats_bundle.head_stats.get(layer_idx)
    if layer_stats_full is None:
        return torch.zeros(
            1, num_kv_heads_local, layer_k.shape[2],
            device=layer_k.device, dtype=torch.float32,
        )

    # Slice stats to local shard's attention heads.
    q_mean_complex_local = layer_stats_full["q_mean_complex"][head_start:head_end]
    q_abs_mean_local = layer_stats_full["q_abs_mean"][head_start:head_end]

    # freq_scale_sq is shared across heads -- expand to local head count.
    freq_scale_sq_1d = layer_stats_full["freq_scale_sq"]  # [freq_count]
    freq_scale_sq_expanded = freq_scale_sq_1d.unsqueeze(0).expand(
        num_attention_heads_local, -1
    ).contiguous()

    head_stats_for_scoring = {
        "q_mean_complex": q_mean_complex_local,
        "q_abs_mean": q_abs_mean_local,
    }

    # Step 3: Compute per-attention-head scores.
    round_start = compressor.state.get_round_start()

    scores_local = compute_scores(
        key_states=layer_k_expanded,
        cache_positions=None,
        head_stats=head_stats_for_scoring,
        omega=compressor.omega,
        offsets=compressor.offsets,
        freq_scale_sq=freq_scale_sq_expanded,
        config=compressor.config,
        round_start=round_start,
        trig_cache=compressor.trig_cache,
    )
    # scores_local: [1, num_attention_heads_local, seq_len] (per_head mode)

    if scores_local.dim() == 2:
        scores_local = scores_local.unsqueeze(1).expand(1, num_kv_heads_local, -1)
        return scores_local.to(dtype=torch.float32)

    # Step 4: GQA aggregate -- max within each group of gqa_group_size.
    seq_len = scores_local.shape[-1]
    scores_grouped = scores_local.view(1, num_kv_heads_local, gqa_group_size, seq_len)
    scores_kv = scores_grouped.max(dim=2).values  # [1, num_kv_heads_local, S]

    return scores_kv.to(dtype=torch.float32)


# ---------------------------------------------------------------------------
# 1. TrigCache helper
# ---------------------------------------------------------------------------

def get_trig_cache(compressor: "TriAttentionCompressor"):
    """Return the compressor's precomputed trig cache, or None.

    The ``TriAttentionCompressor`` creates a ``TrigTableCache`` during
    ``_lazy_init()`` when ``config.use_trig_cache=True`` and Triton
    scoring is enabled.  This helper exposes that cache for external
    callers without reaching into private attributes.

    Parameters
    ----------
    compressor : TriAttentionCompressor
        Must have been lazily initialized (``_lazy_init()`` called).

    Returns
    -------
    TrigTableCache or None
        The cache if available, otherwise None.
    """
    compressor._lazy_init()
    return getattr(compressor, "trig_cache", None)


# ---------------------------------------------------------------------------
# 2. Chunked scoring
# ---------------------------------------------------------------------------

def chunked_score(
    *,
    compressor: "TriAttentionCompressor",
    k_buffers: List[torch.Tensor],
    req_to_token: torch.Tensor,
    req_pool_idx: int,
    effective_len: int,
    available_layers: List[int],
    aggregation_mode: str = "mean",
    score_chunk_max_tokens: int = DEFAULT_SCORE_CHUNK_MAX_TOKENS,
    normalize_scores: bool = False,
) -> Optional[torch.Tensor]:
    """Score a request's KV cache in chunks to limit GPU memory usage.

    For sequences shorter than *score_chunk_max_tokens*, this is
    equivalent to a single gather + score pass.  For longer sequences
    it splits the token range into chunks, gathers and scores each
    chunk independently, then concatenates the per-chunk scores.

    The mathematical result is identical to scoring the full sequence
    at once because the scoring function is *position-wise* -- each
    token's score depends only on its own key vector and the
    precomputed statistics, not on other tokens' scores.

    Parameters
    ----------
    compressor : TriAttentionCompressor
        Initialized compressor (``_lazy_init()`` already called).
    k_buffers : list[Tensor]
        Per-layer key cache, each ``[total_tokens, H, D]``.
    req_to_token : Tensor
        ``[max_reqs, max_context_len]``, int32.
    req_pool_idx : int
        Row in req_to_token for this request.
    effective_len : int
        Number of valid token positions.
    available_layers : list[int]
        Layer indices that have frequency stats.
    aggregation_mode : str
        Cross-layer aggregation: ``"mean"`` or ``"max"``.
    score_chunk_max_tokens : int
        Maximum tokens to gather at once.
    normalize_scores : bool
        If True, apply z-score normalization per layer before
        cross-layer aggregation.

    Returns
    -------
    Tensor or None
        Aggregated scores ``[1, num_kv_heads, effective_len]`` (or
        ``[1, effective_len]`` depending on compressor pruning mode).
        Returns None if no layers produced valid scores.
    """
    if effective_len <= 0:
        return None

    chunk_size = max(1, score_chunk_max_tokens)
    num_chunks = math.ceil(effective_len / chunk_size)

    # Slot indices for the full sequence (compute once).
    device = k_buffers[0].device
    all_slots = req_to_token[req_pool_idx, :effective_len].long().to(device)

    # Accumulator for cross-layer aggregation.
    aggregated_scores: Optional[torch.Tensor] = None
    layer_count = 0

    for layer_idx in available_layers:
        # Score this layer in chunks.
        chunk_results: list[torch.Tensor] = []

        for chunk_idx in range(num_chunks):
            start = chunk_idx * chunk_size
            end = min(start + chunk_size, effective_len)
            chunk_len = end - start

            # Gather dense keys for this chunk from this layer only.
            chunk_slots = all_slots[start:end]

            # k_buffers[layer_idx] is [total_tokens, H, D].
            gathered = k_buffers[layer_idx].index_select(0, chunk_slots)
            # gathered: [chunk_len, H, D] -> [1, H, chunk_len, D]
            chunk_k = gathered.permute(1, 0, 2).unsqueeze(0)

            # Score this chunk.
            chunk_scores = compressor._compute_scores(
                key_states=chunk_k,
                layer_idx=layer_idx,
            )
            # chunk_scores: [1, H, chunk_len] or [1, chunk_len]
            chunk_results.append(chunk_scores)

        # Concatenate chunks along the sequence dimension.
        if len(chunk_results) == 1:
            layer_scores = chunk_results[0]
        else:
            layer_scores = torch.cat(chunk_results, dim=-1)
        # layer_scores: [1, H, effective_len] or [1, effective_len]

        # Per-layer normalization (before cross-layer aggregation).
        if normalize_scores:
            from triattention.vllm.core.utils import normalize_scores as norm_fn
            layer_scores = norm_fn(layer_scores)

        # Cross-layer aggregation.
        if aggregated_scores is None:
            aggregated_scores = layer_scores.clone()
        else:
            if aggregation_mode == "max":
                aggregated_scores = torch.maximum(
                    aggregated_scores, layer_scores
                )
            else:
                # mean: accumulate sum, divide later.
                aggregated_scores.add_(layer_scores)
        layer_count += 1

    if aggregated_scores is None or layer_count <= 0:
        return None

    # Finalize mean aggregation.
    if aggregation_mode != "max" and layer_count > 1:
        aggregated_scores.div_(float(layer_count))

    return aggregated_scores
