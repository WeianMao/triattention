"""Scheduler-side monkey-patch hooks for TriAttention.

This module provides patched versions of three ``Scheduler`` methods:

* ``__init__``                    -- attach TriAttention state (config,
  planner, compressor, effective-length tracker) after the original init.
* ``get_next_batch_to_run``       -- trigger-check and KV compression
  execution for running decode requests.
* ``process_batch_result_decode`` -- update effective-length bookkeeping
  and sync ``kv_committed_len`` / ``kv_allocated_len`` after decode.

Design decisions
----------------
**Compression location**: sglang runs scheduler + worker in the same
process via a synchronous event loop (``event_loop_normal``).  The
scheduler thread has direct access to GPU tensors (KV pool, req_to_token)
without cross-process IPC.  Therefore we execute compression *inside*
the scheduler loop, in ``get_next_batch_to_run``, before the batch is
forwarded to the worker.  This is the simplest correct approach and
avoids the signal/event handshake that vLLM's multi-process architecture
requires.

**Overlap mode**: Phase 1 only supports ``event_loop_normal``.  If
overlap scheduling is detected (``self.enable_overlap is True``),
TriAttention logs a warning and disables itself -- no crash.

**Radix cache**: If radix cache is enabled and the safety check is not
bypassed, TriAttention logs a warning and disables itself.
"""

from __future__ import annotations

import functools
import logging
from typing import Callable, Optional, Set

import torch

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------
# E.1 — Scheduler.__init__ patch
# -----------------------------------------------------------------------


def _patched_scheduler_init(original_init: Callable) -> Callable:
    """Return a wrapper around ``Scheduler.__init__`` that appends
    TriAttention runtime state after the original initialisation.

    Injected state (all prefixed with ``_triattention_``):
      - ``_triattention_enabled``: bool, master switch.
      - ``_triattention_config``: ``SglangIntegrationConfig``.
      - ``_triattention_tracker``: ``EffectiveLengthTracker``.
      - ``_triattention_compressor``: ``TriAttentionCompressor``.
      - ``_triattention_planner``: ``CompressionPlanner``.
      - ``_triattention_stats``: ``StatsBundle``.
      - ``_triattention_prefill_lens``: dict[str, int] — per-request.
      - ``_triattention_step``: int — monotonic step counter.
    """

    @functools.wraps(original_init)
    def wrapped_init(self, *args, **kwargs):
        # Run the original Scheduler.__init__ first.
        original_init(self, *args, **kwargs)

        # ---- TriAttention initialisation ----
        try:
            _init_triattention_state(self)
        except Exception:
            logger.exception(
                "TriAttention init failed — disabling compression for "
                "this scheduler instance.  sglang continues normally."
            )
            self._triattention_enabled = False

    return wrapped_init


def _init_triattention_state(scheduler) -> None:
    """Attach all TriAttention runtime state to *scheduler*.

    Called exactly once during ``Scheduler.__init__`` patch.
    """
    from triattention.sglang.config import load_sglang_config
    from triattention.sglang.effective_length import EffectiveLengthTracker
    from triattention.sglang.stats_loader import (
        StatsBundle,
        load_stats,
        validate_stats_against_model,
    )
    from triattention.vllm.core.compressor import TriAttentionCompressor
    from triattention.vllm.core.config import TriAttentionConfig
    from triattention.vllm.runtime.planner import CompressionPlanner

    # 1. Load configuration from env vars.
    config = load_sglang_config()
    config.log_summary()

    # 2. Master enable check.
    if not config.enable_triattention:
        logger.info("TriAttention is disabled via environment variable.")
        scheduler._triattention_enabled = False
        return

    # 3. Overlap mode check (P-014: Phase 1 normal only).
    if getattr(scheduler, "enable_overlap", False):
        logger.warning(
            "TriAttention does not support overlap scheduling mode in "
            "Phase 1.  Disabling TriAttention.  Launch sglang with "
            "--disable-overlap-schedule to use TriAttention."
        )
        scheduler._triattention_enabled = False
        return

    # 4. Speculative decoding check.
    spec_algo = getattr(scheduler, "spec_algorithm", None)
    if spec_algo is not None and not spec_algo.is_none():
        logger.warning(
            "TriAttention does not support speculative decoding in "
            "Phase 1.  Disabling TriAttention."
        )
        scheduler._triattention_enabled = False
        return

    # 5. Radix cache check.
    disable_radix = getattr(scheduler, "disable_radix_cache", True)
    if not disable_radix and not config.disable_radix_cache_check:
        logger.warning(
            "Radix cache is enabled but TriAttention requires it to be "
            "disabled (--disable-radix-cache).  Disabling TriAttention.  "
            "Set TRIATTN_SGLANG_DISABLE_RADIX_CACHE_CHECK=1 to bypass."
        )
        scheduler._triattention_enabled = False
        return

    # 6. Load frequency stats.
    rc = config.runtime_config
    stats: Optional[StatsBundle] = None
    # Extract num_kv_heads early for both stats loading and the compressor.
    model_config = getattr(scheduler, "model_config", None)
    _num_kv_heads: Optional[int] = None
    if model_config is not None:
        _num_kv_heads = model_config.get_total_num_kv_heads() or None
    if rc.sparse_stats_path is not None:
        stats = load_stats(
            stats_path=str(rc.sparse_stats_path),
            device=torch.device(f"cuda:{scheduler.gpu_id}"),
            num_kv_heads=_num_kv_heads,
        )
        # Validate stats against the model if model_config is available.
        if model_config is not None:
            try:
                num_layers = getattr(
                    model_config.hf_text_config, "num_hidden_layers", 0
                )
                num_kv_heads = model_config.get_total_num_kv_heads()
                head_dim = getattr(model_config, "head_dim", 0)
                if num_layers > 0 and num_kv_heads > 0 and head_dim > 0:
                    validate_stats_against_model(
                        stats, num_layers, num_kv_heads, head_dim
                    )
            except ValueError as e:
                logger.error(
                    "Stats / model mismatch: %s — disabling TriAttention.", e
                )
                scheduler._triattention_enabled = False
                return

    # 7. Build the compressor (algorithm core — framework-agnostic).
    # Pass num_kv_heads so the compressor's internal stats loading
    # applies GQA aggregation correctly.
    compressor_config = TriAttentionConfig(
        kv_budget=rc.kv_budget,
        divide_length=rc.divide_length,
        window_size=rc.window_size,
        protect_prefill=rc.protect_prefill,
        include_prefill_in_budget=rc.include_prefill_in_budget,
        stats_path=str(rc.sparse_stats_path) if rc.sparse_stats_path else None,
        device=f"cuda:{scheduler.gpu_id}",
        sparse_normalize_scores=rc.sparse_normalize_scores,
        pruning_mode=rc.pruning_mode,
        score_aggregation=rc.sparse_score_aggregation,
        disable_mlr=rc.disable_mlr,
        disable_trig=rc.disable_trig,
        num_kv_heads=_num_kv_heads,
        # Pass model_path so compressor can derive correct rope_theta/inv_freq
        # from model config. Without this, compressor falls back to
        # rope_theta=10000 (wrong for Qwen3-32B which uses 1000000).
        model_path=scheduler.server_args.model_path,
        use_triton_scoring=True,
    )
    compressor = TriAttentionCompressor(compressor_config)

    # 8. Build the planner (trigger logic).
    planner = CompressionPlanner(rc)

    # 9. Build the effective-length tracker.
    tracker = EffectiveLengthTracker()

    # 10. Attach everything to the scheduler instance.
    scheduler._triattention_enabled = True
    scheduler._triattention_config = config
    scheduler._triattention_tracker = tracker
    scheduler._triattention_compressor = compressor
    scheduler._triattention_planner = planner
    scheduler._triattention_stats = stats
    # TP sharding info: used by scoring to slice stats to local shard.
    scheduler._triattention_tp_rank = getattr(scheduler, "tp_rank", 0)
    scheduler._triattention_tp_size = getattr(scheduler, "tp_size", 1)
    scheduler._triattention_prefill_lens: dict[str, int] = {}
    scheduler._triattention_step = 0
    # INV-19 — dedup guard.  Tracks req_ids that were compressed in the
    # *previous* step to prevent consecutive double compression if the
    # effective_len tracking has an off-by-1 issue.
    scheduler._triattention_just_compressed: Set[str] = set()
    # Explicit defer set.  Requests whose prefill is not yet fully
    # complete are added here; they are removed on the first decode step
    # after prefill completes, at which point compression is allowed.
    scheduler._triattention_deferred_first_compression: Set[str] = set()
    # Divide_length cooldown.  Maps req_id -> logical_len at
    # last compression.  Even if effective_len exceeds the budget, the next
    # compression is deferred until logical_len has advanced by at least
    # divide_length tokens since the last compression.  This is a defense-
    # in-depth mechanism matching vLLM's last_compression_step guard.
    scheduler._triattention_last_compress_logical: dict[str, int] = {}

    # 11. Register the tracker with the input_patches module so that
    # prepare_for_decode (which runs on ScheduleBatch, not Scheduler)
    # can look it up.
    from triattention.sglang.input_patches import set_active_tracker
    set_active_tracker(tracker)

    logger.info(
        "TriAttention scheduler hooks initialized: budget=%d "
        "divide_length=%d window_size=%d protect_prefill=%s",
        rc.kv_budget,
        rc.divide_length,
        rc.window_size,
        rc.protect_prefill,
    )


# -----------------------------------------------------------------------
# E.2 — get_next_batch_to_run patch
# -----------------------------------------------------------------------


def _patched_get_next_batch_to_run(original_fn: Callable) -> Callable:
    """Return a wrapper around ``Scheduler.get_next_batch_to_run`` that
    performs KV compression for eligible decode requests before the
    batch is sent to the worker.

    Compression pipeline (per eligible request):
      1. Check trigger via ``CompressionPlanner.build_signal``.
      2. Gather dense key tensor via ``gather_request_k_dense``.
      3. Score + select via ``TriAttentionCompressor.compress``.
      4. Compact KV in-place via ``compact_request_kv_in_place``.
      5. Reclaim freed slots via ``reclaim_freed_slots``.
      6. Sync KV metadata via ``EffectiveLengthTracker.sync_kv_metadata``.
      7. Update tracker effective length.

    Compression happens *before* ``update_running_batch`` /
    ``prepare_for_decode`` so that ``seq_lens`` in the batch reflects
    effective lengths after compaction.
    """

    @functools.wraps(original_fn)
    def wrapped(self):
        # Fast path: TriAttention disabled or not initialised.
        if not getattr(self, "_triattention_enabled", False):
            return original_fn(self)

        # Sync effective offsets before scheduling.
        # vLLM calls _sync_effective_kv_offsets_before_schedule() before
        # each schedule round.  Without this, already-compressed requests
        # that are NOT processed by the compression pass (e.g. they did
        # not hit the trigger threshold this round) would have stale
        # effective lengths, causing prepare_for_decode to use wrong
        # seq_lens values.
        try:
            _sync_effective_offsets_before_schedule(self)
        except Exception:
            logger.exception(
                "TriAttention pre-schedule offset sync failed — "
                "continuing with potentially stale effective lengths."
            )

        # Run compression before the original scheduling logic.
        # The original method calls update_running_batch → prepare_for_decode
        # internally, which allocates new KV slots and bumps seq_lens.
        # We need compression to happen before that to ensure seq_lens
        # reflects effective (post-compression) lengths.
        try:
            _run_compression_pass(self)
        except Exception:
            logger.exception(
                "TriAttention compression pass failed — skipping this "
                "round.  sglang continues normally."
            )

        # Call the original scheduling logic.
        batch = original_fn(self)

        # Increment step counter.
        self._triattention_step = getattr(self, "_triattention_step", 0) + 1

        return batch

    return wrapped


def _sync_effective_offsets_before_schedule(scheduler) -> None:
    """Sync effective lengths for all running requests before scheduling.

    Equivalent to vLLM's ``_sync_effective_kv_offsets_before_schedule()``.

    For already-compressed requests that are still generating tokens,
    the effective length must be updated to reflect new tokens appended
    since the last observation.  Without this, prepare_for_decode would
    use stale effective lengths for requests that did not trigger
    compression in this round.
    """
    from triattention.sglang.effective_length import EffectiveLengthTracker

    running_batch = getattr(scheduler, "running_batch", None)
    if running_batch is None or running_batch.is_empty():
        return

    tracker = scheduler._triattention_tracker
    if not tracker.has_any_overrides():
        return

    for req in running_batch.reqs:
        req_id = req.rid
        if req_id in tracker._last_logical:
            # Use origin_input_ids + output_ids instead of fill_ids.
            # During decode, sglang sets fill_ids to only the last token
            # ([last_output_id]), so len(fill_ids) == 1.  Using that value
            # corrupts the tracker: effective_len rolls back to
            # min(effective, 1) == 1, causing every subsequent step to see
            # a huge delta and trigger compression.
            logical_len = len(req.origin_input_ids) + len(getattr(req, 'output_ids', []))
            # Retract handling.  If sglang's retract_decode moved this
            # request out and back into the running batch, logical_len
            # may have decreased.  The tracker's observe_logical_progress
            # already handles this via min(effective, logical_len)
            # rollback.  We also sync kv_metadata if effective is tracked
            # and the request was retracted (logical decreased).
            prev_logical = tracker._last_logical.get(req_id, logical_len)
            effective = tracker.observe_logical_progress(req_id, logical_len)
            if logical_len < prev_logical and tracker.has_override(req_id):
                # Retract detected: re-sync kv metadata to match the
                # (potentially clamped) effective length.
                EffectiveLengthTracker.sync_kv_metadata(req, effective)


def _run_compression_pass(scheduler) -> None:
    """Iterate over running requests and compress those that exceed
    the trigger threshold.

    This function directly accesses the GPU KV pool via the scheduler's
    references (same-process architecture).
    """
    running_batch = getattr(scheduler, "running_batch", None)
    if running_batch is None or running_batch.is_empty():
        return

    # Only compress during decode, not prefill.
    if getattr(running_batch, "is_prefill_only", False):
        return

    tracker: "EffectiveLengthTracker" = scheduler._triattention_tracker
    planner: "CompressionPlanner" = scheduler._triattention_planner
    compressor: "TriAttentionCompressor" = scheduler._triattention_compressor
    config = scheduler._triattention_config
    rc = config.runtime_config

    # Access KV pool infrastructure.
    req_to_token_pool = scheduler.req_to_token_pool
    req_to_token = req_to_token_pool.req_to_token
    kv_cache = scheduler.token_to_kv_pool_allocator.get_kvcache()
    k_buffers = kv_cache.k_buffer
    v_buffers = kv_cache.v_buffer
    allocator = scheduler.token_to_kv_pool_allocator

    step = getattr(scheduler, "_triattention_step", 0)
    prefill_lens = scheduler._triattention_prefill_lens
    # INV-19 dedup: capture previous step's compressed set, then clear
    # for this step.
    prev_compressed = scheduler._triattention_just_compressed
    scheduler._triattention_just_compressed = set()
    deferred = scheduler._triattention_deferred_first_compression

    from triattention.sglang.kv_compaction import (
        compact_request_kv_in_place,
        gather_request_k_dense,
        reclaim_freed_slots,
    )
    from triattention.sglang.effective_length import EffectiveLengthTracker

    for req in list(running_batch.reqs):
        req_id = req.rid

        # Register request if not yet tracked.
        if req_id not in tracker._last_logical:
            # Long prefill progression.
            # Register with the CURRENT logical length (which may be
            # large if this is a long-prompt request that just finished
            # prefill).  This sets _last_logical correctly so that the
            # first observe_logical_progress after registration computes
            # delta=0 or delta=1 (not a huge jump).  For chunked
            # prefill, the request may be registered during prefill
            # (logical_len < prefill_len) and then observed again after
            # prefill completes — the deferred set ensures no
            # compression happens until decode, and the tracker sees
            # each increment correctly.
            # Use origin_input_ids + output_ids (fill_ids is stale during decode).
            logical_len = len(req.origin_input_ids) + len(getattr(req, 'output_ids', []))
            tracker.register_request(req_id, logical_len)
            prefill_lens[req_id] = len(req.origin_input_ids)

        # sglang fill_ids is only updated during init_next_round_input
        # (prefill/extend), not during decode.  Use origin_input_ids +
        # output_ids to get the true logical length.
        logical_len = len(req.origin_input_ids) + len(getattr(req, 'output_ids', []))

        # Off-by-1 guard.  In sglang's event loop, compression runs in
        # get_next_batch_to_run BEFORE the
        # decode step appends the new token.  So len(req.fill_ids) at
        # this point reflects the state BEFORE the current decode token
        # is generated.  effective_len should match: it counts KV
        # entries already in the cache, not including the about-to-be-
        # generated token.  The +1 in estimated_cache_len below
        # accounts for the token that WILL be appended after this
        # scheduling round.
        # This is the correct timing — no off-by-1 here.

        # Get effective length (accounts for prior compressions).
        effective_len = tracker.observe_logical_progress(req_id, logical_len)

        # Explicit chunked prefill defer via _deferred_first_compression
        # set.
        # Requests still in prefill are added to the deferred set.
        # Once prefill completes (logical_len > prefill_len), we check
        # if the request is in the deferred set.  If so, we remove it
        # and skip compression for THIS step (the first decode step
        # after prefill), allowing compression only from the second
        # decode step onward.  This ensures KV from the final prefill
        # chunk is fully committed before any compression occurs.
        prefill_len_for_req = len(req.origin_input_ids)
        if logical_len <= prefill_len_for_req:
            # Still in prefill — add to deferred set and skip.
            deferred.add(req_id)
            continue

        if req_id in deferred:
            # First decode step after prefill completes.  Remove from
            # deferred and skip this
            # step to ensure prefill KV is fully settled.
            deferred.discard(req_id)
            logger.debug(
                "TriAttention: deferring first compression for req=%s "
                "(first decode step after prefill).",
                req_id,
            )
            continue

        # INV-19 dedup guard.  If this request was compressed in the
        # immediately preceding step, skip it to
        # avoid double-compression from potential off-by-1 in the
        # trigger condition.
        if req_id in prev_compressed:
            logger.debug(
                "TriAttention: skipping req=%s (compressed in "
                "previous step, dedup guard).",
                req_id,
            )
            continue

        # Divide_length cooldown.  If this request was compressed
        # before, check that at least divide_length new tokens
        # have been generated since the last compression.  This prevents
        # runaway compression even if effective_len tracking has residual
        # off-by-1 issues.
        last_compress_logical = scheduler._triattention_last_compress_logical
        if req_id in last_compress_logical:
            tokens_since = logical_len - last_compress_logical[req_id]
            if tokens_since < rc.divide_length:
                continue

        # Build compression signal via the planner.
        prefill_len = prefill_lens.get(req_id, len(req.origin_input_ids))

        # Long prompt handling.  When the prompt itself exceeds
        # kv_budget, the first decode step after
        # prefill will have effective_len >= kv_budget + divide_length
        # already.  The planner's build_signal handles this naturally
        # (estimated_cache_len will be large enough to trigger).  No
        # special case needed beyond ensuring the deferred set logic
        # above allows this step through.

        # Match the HF count_prompt_tokens=False semantics: trigger on
        # decode-only token count (effective_size = seq_len - prefix).
        decode_only_cache_len = effective_len - prefill_len_for_req + 1
        signal = planner.build_signal(
            req_id=req_id,
            estimated_cache_len=decode_only_cache_len,
            prefill_len=0,  # already subtracted above
            step=step,
        )

        if not signal.should_compress:
            continue

        # ---- Execute compression ----
        logger.info(
            "TriAttention compressing req=%s effective_len=%d "
            "logical_len=%d reason=%s",
            req_id,
            effective_len,
            logical_len,
            signal.reason,
        )

        try:
            _compress_single_request(
                scheduler=scheduler,
                req=req,
                req_id=req_id,
                effective_len=effective_len,
                prefill_len=prefill_len,
                k_buffers=k_buffers,
                v_buffers=v_buffers,
                req_to_token=req_to_token,
                allocator=allocator,
                compressor=compressor,
                tracker=tracker,
                rc=rc,
            )
            # INV-19: record that this req was compressed this step for
            # dedup in the next step.
            scheduler._triattention_just_compressed.add(req_id)
            # Record logical_len at compression for divide_length
            # cooldown.
            scheduler._triattention_last_compress_logical[req_id] = logical_len
        except Exception:
            logger.exception(
                "TriAttention compression failed for req=%s — "
                "skipping, request continues uncompressed.",
                req_id,
            )


def _compress_single_request(
    *,
    scheduler,
    req,
    req_id: str,
    effective_len: int,
    prefill_len: int,
    k_buffers,
    v_buffers,
    req_to_token: torch.Tensor,
    allocator,
    compressor,
    tracker,
    rc,
) -> None:
    """Run the full compression pipeline for a single request.

    Steps:
      1. Validate request state.
      2. Gather dense key tensor from KV pool.
      3. Score each layer independently, aggregate across layers.
      4. Apply window + prefill protection, select top-k.
      5. Post-selection validation (window, prefill, budget invariants).
      6. Compact KV in-place.
      7. Reclaim freed slots.
      8. Sync KV metadata on the Req object.
      9. Update tracker.
    """
    from triattention.sglang.kv_compaction import (
        compact_request_kv_in_place,  # noqa: F811 — kept as fallback
        compact_request_kv_in_place_per_head,
        gather_request_k_dense,
        reclaim_freed_slots,
    )
    from triattention.sglang.effective_length import EffectiveLengthTracker

    # Request state lookup fallback with detailed diagnostics.
    # Validate req_pool_idx before any GPU op.
    req_pool_idx = getattr(req, "req_pool_idx", None)
    if req_pool_idx is None:
        logger.warning(
            "TriAttention: req=%s has no req_pool_idx attribute — "
            "skipping compression (request may have been freed).",
            req_id,
        )
        return
    # Check that req_pool_idx is within the valid range of req_to_token.
    if req_pool_idx < 0 or req_pool_idx >= req_to_token.shape[0]:
        logger.warning(
            "TriAttention: req=%s has invalid req_pool_idx=%d "
            "(valid range 0..%d) — skipping compression.",
            req_id,
            req_pool_idx,
            req_to_token.shape[0] - 1,
        )
        return
    # Check that effective_len does not exceed req_to_token row capacity.
    if effective_len > req_to_token.shape[1]:
        logger.warning(
            "TriAttention: req=%s effective_len=%d exceeds "
            "req_to_token capacity=%d — skipping compression.",
            req_id,
            effective_len,
            req_to_token.shape[1],
        )
        return

    # Scoring pipeline operates at attention-head granularity (e.g. 64
    # heads for Llama-3-8B) matching the HF reference:
    #   - Stats loaded at num_attention_heads (not num_kv_heads)
    #   - K expanded via repeat_interleave to num_attention_heads
    #   - Each attention head scored independently
    #   - GQA aggregation via max within group (not mean)
    #   - Cross-layer aggregation via mean (matching HF per_head mode)

    # Determine which layers have stats for scoring.
    # Use stats_bundle layers (attention-head granularity), not
    # compressor.head_stats (KV-head granularity).
    stats_bundle = scheduler._triattention_stats
    compressor._lazy_init()
    available_layers = sorted(stats_bundle.head_stats.keys())
    if not available_layers:
        logger.warning(
            "TriAttention: no layers with stats for req=%s, skipping.",
            req_id,
        )
        return

    # Step 1: Gather dense key tensors for layers with stats.
    k_dense = gather_request_k_dense(
        k_buffers=k_buffers,
        req_to_token=req_to_token,
        req_pool_idx=req_pool_idx,
        effective_len=effective_len,
        target_layers=available_layers,
    )
    # k_dense shape: [len(available_layers), num_kv_heads, effective_len, head_dim]

    # Set absolute_position so the scoring formula's trig phase terms
    # (cos(t*omega), sin(t*omega)) use the correct decode position.
    logical_len = len(req.origin_input_ids) + len(getattr(req, "output_ids", []))
    compressor.state.absolute_position = logical_len

    # Step 2: Per-attention-head scoring with GQA max aggregation.
    # Cross-layer aggregation: mean of per-layer max-aggregated scores.
    aggregation_mode = getattr(rc, "layer_perhead_aggregation", "mean")

    from triattention.sglang.scoring_utils import score_layer_hf_aligned

    aggregated_scores = None
    layer_count = 0

    for i, layer_idx in enumerate(available_layers):
        layer_k = k_dense[i:i+1]  # [1, num_kv_heads, effective_len, head_dim]

        # Score at attention-head granularity, aggregate to KV heads via max.
        tp_rank = getattr(scheduler, "_triattention_tp_rank", 0)
        tp_size = getattr(scheduler, "_triattention_tp_size", 1)
        layer_scores = score_layer_hf_aligned(
            layer_k=layer_k,
            layer_idx=layer_idx,
            stats_bundle=stats_bundle,
            compressor=compressor,
            tp_rank=tp_rank,
            tp_size=tp_size,
        )
        # layer_scores: [1, num_kv_heads, effective_len]

        # Apply normalization if enabled (per-layer, before cross-layer aggregation).
        if compressor.config.sparse_normalize_scores:
            from triattention.vllm.core.utils import normalize_scores
            layer_scores = normalize_scores(layer_scores)

        if aggregated_scores is None:
            aggregated_scores = layer_scores.clone()
        else:
            if aggregation_mode == "max":
                aggregated_scores = torch.maximum(aggregated_scores, layer_scores)
            else:
                # mean: accumulate sum, divide later
                aggregated_scores.add_(layer_scores)
        layer_count += 1

    if aggregated_scores is None or layer_count <= 0:
        logger.warning(
            "TriAttention: no valid layer scores for req=%s, skipping.",
            req_id,
        )
        return

    # Finalize cross-layer aggregation.
    if aggregation_mode != "max" and layer_count > 1:
        aggregated_scores.div_(float(layer_count))

    # Window protection is disabled when window_size=0.
    window_size = compressor.config.window_size
    if window_size > 0:
        from triattention.vllm.core.utils import protect_window_tokens
        aggregated_scores = protect_window_tokens(
            aggregated_scores, window_size
        )

    # Apply prefill protection if enabled.
    if rc.protect_prefill and prefill_len > 0:
        aggregated_scores[..., :prefill_len] = float("inf")

    # Step 3: Select top-k from aggregated scores.
    k = min(rc.kv_budget, effective_len)
    if k <= 0:
        return

    keep_indices = torch.topk(
        aggregated_scores, k=k, dim=-1, largest=True, sorted=False,
    ).indices

    # Reshape topk output to [num_kv_heads, budget].
    # No head-union/truncate — each head keeps its own top-K directly.
    num_kv_heads = k_buffers[0].shape[1]
    if keep_indices.dim() == 3:
        # [1, num_kv_heads, budget] → [num_kv_heads, budget]
        keep_indices_per_head = keep_indices.squeeze(0)
    elif keep_indices.dim() == 2:
        # [1, budget] — per_layer mode: all heads share same indices
        keep_indices_per_head = keep_indices.squeeze(0).unsqueeze(0).expand(
            num_kv_heads, -1
        )
    else:
        # 1D fallback
        keep_indices_per_head = keep_indices.flatten().unsqueeze(0).expand(
            num_kv_heads, -1
        )

    # Ensure int64 dtype for downstream indexing.
    keep_indices_per_head = keep_indices_per_head.to(dtype=torch.int64)

    # Ensure each head's indices are sorted ascending.
    keep_indices_per_head = torch.sort(
        keep_indices_per_head, dim=-1
    ).values

    budget = keep_indices_per_head.shape[1]

    if budget >= effective_len:
        # No actual compression needed.
        logger.debug(
            "TriAttention: budget=%d >= effective_len=%d for req=%s, "
            "skipping compaction.",
            budget,
            effective_len,
            req_id,
        )
        return

    # INV-02 window-protection check is skipped when window_size=0.

    # INV-03: vectorized per-head prefill protection check.
    # Verify critical prefix tokens are retained by ALL heads.
    if rc.protect_prefill and prefill_len > 0:
        check_count = min(4, prefill_len, budget)
        if check_count > 0:
            critical_prefix = torch.arange(
                0, check_count,
                device=keep_indices_per_head.device,
                dtype=keep_indices_per_head.dtype,
            )  # [C]
            # Each head's first C indices should be exactly 0..C-1
            # (since indices are sorted and prefill positions got +inf).
            first_c = keep_indices_per_head[:, :check_count]  # [H, C]
            prefix_ok = (
                first_c == critical_prefix.unsqueeze(0)
            ).all()
            if not prefix_ok:
                per_head_ok = (
                    first_c == critical_prefix.unsqueeze(0)
                ).all(dim=1)  # [H]
                failing_heads = (~per_head_ok).sum().item()
                logger.error(
                    "TriAttention: INV-03 violation — %d/%d heads "
                    "have critical prefill tokens missing from "
                    "keep_indices for req=%s.  Aborting compression.",
                    failing_heads,
                    num_kv_heads,
                    req_id,
                )
                return

    # Step 4: Per-head compact KV in-place across all layers.
    compact_request_kv_in_place_per_head(
        k_buffers=k_buffers,
        v_buffers=v_buffers,
        req_to_token=req_to_token,
        req_pool_idx=req_pool_idx,
        keep_indices_per_head=keep_indices_per_head,
        effective_len=effective_len,
    )

    # Step 5: Reclaim freed slots.
    freed_slots = reclaim_freed_slots(
        req_to_token=req_to_token,
        req_pool_idx=req_pool_idx,
        budget=budget,
        effective_len=effective_len,
        allocator=allocator,
    )

    # Step 6: Sync kv_committed_len and kv_allocated_len on the Req.
    EffectiveLengthTracker.sync_kv_metadata(req, budget)

    # Step 7: Update tracker with new effective length.
    tracker.update_after_compression(req_id, budget)

    # INV-01 post-compression budget invariant check.
    # After update_after_compression, verify that the tracker's
    # effective length equals the budget we just set.
    tracked_effective = tracker.get_effective_len(req_id)
    if tracked_effective is not None and tracked_effective != budget:
        logger.error(
            "TriAttention: INV-01 post-compression invariant "
            "violation — tracker effective=%d != budget=%d for "
            "req=%s.  Forcing tracker to budget.",
            tracked_effective,
            budget,
            req_id,
        )
        tracker.update_after_compression(req_id, budget)

    logger.info(
        "TriAttention compression complete: req=%s "
        "effective_len %d -> %d (freed %d slots)",
        req_id,
        effective_len,
        budget,
        freed_slots.numel(),
    )


# -----------------------------------------------------------------------
# E.3 — process_batch_result_decode patch
# -----------------------------------------------------------------------


def _patched_process_batch_result_decode(original_fn: Callable) -> Callable:
    """Return a wrapper around ``process_batch_result_decode`` that
    updates the effective-length tracker after each decode step.

    After the original method processes the decode result:
      1. For each request still running: call
         ``observe_logical_progress`` so the tracker knows the new
         logical length (logical +1 per decode step).
      2. For finished requests: remove from tracker.
      3. For newly entered requests: register in tracker.
    """

    @functools.wraps(original_fn)
    def wrapped(self, batch, result):
        # Call original processing first — this updates output_ids,
        # checks finish conditions, frees completed requests, etc.
        original_fn(self, batch, result)

        # Fast path: TriAttention disabled.
        if not getattr(self, "_triattention_enabled", False):
            return

        try:
            _update_tracker_after_decode(self, batch)
        except Exception:
            logger.exception(
                "TriAttention post-decode tracker update failed — "
                "state may be stale but sglang continues."
            )

    return wrapped


def _update_tracker_after_decode(scheduler, batch) -> None:
    """Sync the effective-length tracker after a decode step.

    This is called after ``process_batch_result_decode`` has updated
    ``req.output_ids`` and checked finish conditions.
    """
    tracker = scheduler._triattention_tracker
    prefill_lens = scheduler._triattention_prefill_lens

    for req in batch.reqs:
        req_id = req.rid

        # Use origin_input_ids + output_ids instead of fill_ids.
        # During decode, sglang sets fill_ids to only the
        # last token ([last_output_id]), so len(fill_ids) == 1.  Using
        # that value corrupts the tracker: effective_len rolls back to
        # min(effective, 1) == 1, causing every subsequent step to see
        # a huge delta and trigger compression.
        logical_len = len(req.origin_input_ids) + len(getattr(req, 'output_ids', []))

        if req_id not in tracker._last_logical:
            # Newly seen request — register it.
            tracker.register_request(req_id, logical_len)
            prefill_lens[req_id] = len(req.origin_input_ids)
        else:
            # Update tracker with latest logical length.
            tracker.observe_logical_progress(req_id, logical_len)

        # Clean up ALL per-request state for finished requests.
        # tracker.remove_request pops both _effective and _last_logical.
        # prefill_lens.pop removes the prefill length entry.  The
        # compressor.state is shared (not per-request) and does not
        # accumulate per-request data, so no additional cleanup is needed
        # there.  The planner is stateless (build_signal recomputes each
        # time).  Also clean up dedup and deferred sets for
        # finished/aborted requests.
        if req.finished():
            tracker.remove_request(req_id)
            prefill_lens.pop(req_id, None)
            scheduler._triattention_just_compressed.discard(req_id)
            scheduler._triattention_deferred_first_compression.discard(req_id)
            # Clean up cooldown state for finished requests.
            scheduler._triattention_last_compress_logical.pop(req_id, None)
