"""Runner state/lifecycle updates for TriAttention runtime."""

from __future__ import annotations

import logging
import os
from typing import Any

from .debug_trace import trace_event
from .signals import CompressionSignal


_LEGACY_COMPRESSED_ESTIMATE = (
    os.environ.get("TRIATTN_RUNTIME_LEGACY_COMPRESSED_ESTIMATE", "0").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _resolve_full_prefill_len_from_request_like(request_like: Any) -> int:
    candidates: list[int] = []

    prompt_token_ids = getattr(request_like, "prompt_token_ids", None)
    if prompt_token_ids is not None:
        try:
            candidates.append(len(prompt_token_ids))
        except Exception:
            pass

    for attr_name in ("prompt_token_ids_len", "num_prompt_tokens"):
        raw_value = getattr(request_like, attr_name, None)
        if raw_value is None:
            continue
        try:
            candidates.append(int(raw_value))
        except (TypeError, ValueError):
            continue

    prefill_token_ids = getattr(request_like, "prefill_token_ids", None)
    if prefill_token_ids is not None:
        try:
            candidates.append(len(prefill_token_ids))
        except Exception:
            pass

    return max(candidates, default=0)


def register_new_requests(*, state_store: Any, scheduler_output: Any, protect_prefill: bool) -> None:
    for new_req in scheduler_output.scheduled_new_reqs:
        prefill_len = _resolve_full_prefill_len_from_request_like(new_req)
        state_store.ensure(
            req_id=new_req.req_id,
            prefill_len=prefill_len,
            protect_prefill=protect_prefill,
        )


def cleanup_finished_requests(*, state_store: Any, scheduler_output: Any) -> None:
    for req_id in scheduler_output.finished_req_ids:
        state_store.remove(req_id)


def mark_preemptions(*, state_store: Any, scheduler_output: Any) -> None:
    preempted_req_ids = getattr(scheduler_output, "preempted_req_ids", None)
    if not preempted_req_ids:
        return
    for req_id in preempted_req_ids:
        state_store.mark_preempted(req_id)


def mark_resumed(*, state_store: Any, scheduler_output: Any) -> None:
    cached_reqs = getattr(scheduler_output, "scheduled_cached_reqs", None)
    if cached_reqs is None:
        return
    resumed_req_ids = getattr(cached_reqs, "resumed_req_ids", None)
    if resumed_req_ids is None:
        resumed_req_ids = getattr(cached_reqs, "req_ids", []) or []
    for req_id in resumed_req_ids:
        state_store.mark_resumed(req_id)


def consume_runner_signals(
    *,
    state_store: Any,
    scheduler_output: Any,
    last_step: int,
    logger: logging.Logger,
    log_decisions: bool,
) -> tuple[int, dict[str, CompressionSignal]]:
    step = getattr(scheduler_output, "triattention_step", last_step + 1)
    signals: dict[str, CompressionSignal] = getattr(scheduler_output, "triattention_signals", {})

    for req_id, signal in signals.items():
        state = state_store.ensure(
            req_id=req_id,
            prefill_len=signal.prefill_len,
            protect_prefill=signal.protect_prefill,
        )
        scheduled_tokens = max(0, int(getattr(signal, "scheduled_tokens", 1)))
        cache_len_est = max(0, int(signal.estimated_cache_len))
        if int(getattr(state, "compression_count", 0)) > 0:
            # Async mode can temporarily expose stale scheduler estimates (both
            # high and low).  Once a request is compressed, keep runner-side
            # effective length on a local monotonic recurrence and avoid mixing
            # in scheduler snapshots for this step.
            prev_cache_len = max(0, int(getattr(state, "current_cache_len", 0)))
            local_est = prev_cache_len + scheduled_tokens
            cache_len_est = (
                max(0, int(signal.estimated_cache_len))
                if _LEGACY_COMPRESSED_ESTIMATE
                else local_est
            )
        state_store.update_cache_len(
            req_id,
            cache_len_est,
            step=signal.step,
        )
        if signal.should_compress:
            state_store.mark_trigger(req_id, signal.reason, signal.step)
            trace_event(
                "signal_trigger",
                req_id=repr(req_id),
                step=int(signal.step),
                reason=str(signal.reason),
                estimated_cache_len=int(signal.estimated_cache_len),
                scheduled_tokens=int(getattr(signal, "scheduled_tokens", 1)),
            )
            if log_decisions:
                logger.debug(
                    "TriAttention trigger req=%s step=%d reason=%s len=%d mode=%s",
                    req_id,
                    signal.step,
                    signal.reason,
                    signal.estimated_cache_len,
                    state.mode,
                )
    return step, signals
