"""Compression action execution for TriAttentionModelRunner."""
from __future__ import annotations

import logging
from typing import Any

from .constants import TRITON_SCORING_REQUIRED_MARKER
from .signals import CompressionSignal


def execute_runner_compression_actions(
    *,
    executor: Any,
    state_store: Any,
    scheduler_output: Any,
    signals: dict[str, CompressionSignal],
    strict_no_downgrade: bool,
    allowed_strict_skip_reasons: set[str],
    logger: logging.Logger,
    log_decisions: bool,
) -> list[dict[str, Any]]:
    """Execute compression for triggered requests and emit scheduler-side events."""
    events: list[dict[str, Any]] = []
    for req_id, signal in signals.items():
        if not signal.should_compress:
            continue
        # Guard against V1 batch-queue race: scheduler may emit consecutive
        # compression signals for the same request before update_from_output
        # runs.  The worker-side block table was already shrunk by the first
        # step, so executing again would desync scheduler/worker block counts.
        # Exception: during chunked prefill (scheduled_tokens > 1), each step
        # adds up to 2048 tokens, so consecutive compression is expected and
        # necessary to avoid excessive accumulation.
        req_state = state_store.get(req_id) if hasattr(state_store, "get") else None
        if req_state is not None:
            last_step = getattr(req_state, "last_compression_step", -1)
            compression_count = int(getattr(req_state, "compression_count", 0) or 0)
            sched_tokens = int(getattr(signal, "scheduled_tokens", 1))
            if compression_count > 0 and last_step >= 0 and signal.step - last_step <= 1 and sched_tokens <= 1:
                logger.info(
                    "TriAttention compression skipped (batch-queue dedup) "
                    "req=%s step=%d last_compression_step=%d",
                    req_id, signal.step, last_step,
                )
                if hasattr(state_store, "mark_compression_skipped"):
                    state_store.mark_compression_skipped(
                        req_id=req_id,
                        reason="batch_queue_dedup",
                        step=signal.step,
                    )
                events.append(
                    {
                        "req_id": req_id,
                        "step": signal.step,
                        "status": "skipped",
                        "reason": "batch_queue_dedup",
                        "cache_len_after": getattr(req_state, "current_cache_len", None),
                        "scheduled_tokens": int(getattr(signal, "scheduled_tokens", 1)),
                        "estimated_cache_len": int(getattr(signal, "estimated_cache_len", 0)),
                        "prefill_len": int(getattr(signal, "prefill_len", 0)),
                    }
                )
                continue
        try:
            result = executor.execute(
                req_id=req_id,
                signal=signal,
                scheduler_output=scheduler_output,
            )
        except Exception as exc:  # pragma: no cover - safety fallback
            if strict_no_downgrade:
                logger.exception(
                    "TriAttention strict mode fatal: compression executor exception "
                    "req=%s step=%d",
                    req_id,
                    signal.step,
                )
                raise RuntimeError(
                    f"{TRITON_SCORING_REQUIRED_MARKER}:executor_exception:"
                    f"req={req_id}:step={signal.step}:type={type(exc).__name__}"
                ) from exc
            if TRITON_SCORING_REQUIRED_MARKER in str(exc):
                logger.exception(
                    "TriAttention fatal: Triton scoring is required. "
                    "req=%s step=%d",
                    req_id,
                    signal.step,
                )
                raise
            state_store.mark_compression_skipped(
                req_id=req_id,
                reason=f"executor_exception:{type(exc).__name__}",
                step=signal.step,
            )
            logger.exception(
                "TriAttention compression executor failed req=%s step=%d",
                req_id,
                signal.step,
            )
            events.append(
                {
                    "req_id": req_id,
                    "step": signal.step,
                    "status": "error",
                    "reason": f"executor_exception:{type(exc).__name__}",
                    "cache_len_after": None,
                    "scheduled_tokens": int(getattr(signal, "scheduled_tokens", 1)),
                    "estimated_cache_len": int(getattr(signal, "estimated_cache_len", 0)),
                    "prefill_len": int(getattr(signal, "prefill_len", 0)),
                }
            )
            continue

        if (
            strict_no_downgrade
            and not result.applied
            and result.reason not in allowed_strict_skip_reasons
        ):
            raise RuntimeError(
                f"{TRITON_SCORING_REQUIRED_MARKER}:unexpected_skip:"
                f"req={req_id}:step={signal.step}:reason={result.reason}"
            )

        if result.applied:
            cache_len_after = (
                signal.estimated_cache_len
                if result.cache_len_after is None
                else result.cache_len_after
            )
            details = result.details if isinstance(result.details, dict) else {}
            before_len = details.get("effective_tokens_before")
            budget_total = details.get("budget_total")
            reclaimed_block_count = details.get("reclaimed_block_count")
            recent_unabsorbed_tokens = details.get("recent_unabsorbed_tokens")
            logger.info(
                "TriAttention compression applied req=%s step=%d reason=%s "
                "before=%s after=%d reclaimed_blocks=%s",
                req_id, signal.step, result.reason,
                before_len, cache_len_after, reclaimed_block_count,
            )
            # Resolve scheduler_nct for this request so state can record
            # the num_computed_tokens at compression time (used by
            # build_effective_sparse_overrides for stable delta).
            _sched_nct = None
            _cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
            if _cached_reqs is not None:
                _cr_ids = getattr(_cached_reqs, "req_ids", None)
                _cr_nct = getattr(_cached_reqs, "num_computed_tokens", None)
                if isinstance(_cr_ids, list) and isinstance(_cr_nct, list):
                    try:
                        _idx = _cr_ids.index(req_id)
                        _sched_nct = int(_cr_nct[_idx])
                    except (ValueError, IndexError):
                        pass
            state_store.mark_compressed(
                req_id=req_id,
                step=signal.step,
                cache_len=cache_len_after,
                scheduled_tokens=int(getattr(signal, "scheduled_tokens", 1)),
                scheduler_nct=_sched_nct,
            )
            if log_decisions:
                logger.debug(
                    "TriAttention compression applied req=%s step=%d reason=%s",
                    req_id,
                    signal.step,
                    result.reason,
                )
            if log_decisions and isinstance(before_len, int):
                logger.debug(
                    "TriAttention compression summary req=%s step=%d before=%d after=%d "
                    "budget=%s reclaimed_blocks=%s recent_unabsorbed=%s "
                    "scheduled_tokens=%s estimated_cache_len=%s reason=%s",
                    req_id,
                    signal.step,
                    before_len,
                    cache_len_after,
                    budget_total,
                    reclaimed_block_count,
                    recent_unabsorbed_tokens,
                    int(getattr(signal, "scheduled_tokens", 1)),
                    int(getattr(signal, "estimated_cache_len", 0)),
                    result.reason,
                )
            events.append(
                {
                    "req_id": req_id,
                    "step": signal.step,
                    "status": "applied",
                    "reason": result.reason,
                    "cache_len_after": cache_len_after,
                    "details": result.details,
                    "scheduled_tokens": int(getattr(signal, "scheduled_tokens", 1)),
                    "estimated_cache_len": int(getattr(signal, "estimated_cache_len", 0)),
                    "prefill_len": int(getattr(signal, "prefill_len", 0)),
                    "block_reclaim": (
                        result.details.get("block_reclaim")
                        if isinstance(result.details, dict)
                        else None
                    ),
                }
            )
            continue

        state_store.mark_compression_skipped(
            req_id=req_id,
            reason=result.reason,
            step=signal.step,
        )
        logger.info(
            "TriAttention compression skipped req=%s step=%d reason=%s "
            "cache_len_after=%s details=%s",
            req_id,
            signal.step,
            result.reason,
            result.cache_len_after,
            result.details,
        )
        events.append(
            {
                "req_id": req_id,
                "step": signal.step,
                "status": "skipped",
                "reason": result.reason,
                "cache_len_after": result.cache_len_after,
                "details": result.details,
                "scheduled_tokens": int(getattr(signal, "scheduled_tokens", 1)),
                "estimated_cache_len": int(getattr(signal, "estimated_cache_len", 0)),
                "prefill_len": int(getattr(signal, "prefill_len", 0)),
                "block_reclaim": (
                    result.details.get("block_reclaim")
                    if isinstance(result.details, dict)
                    else None
                ),
            }
        )
    return events
