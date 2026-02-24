"""Runner state/lifecycle updates for TriAttention V2."""

from __future__ import annotations

import logging
from typing import Any

from .signals import CompressionSignal


def register_new_requests(*, state_store: Any, scheduler_output: Any, protect_prefill: bool) -> None:
    for new_req in scheduler_output.scheduled_new_reqs:
        if new_req.prefill_token_ids is not None:
            prefill_len = len(new_req.prefill_token_ids)
        elif new_req.prompt_token_ids is not None:
            prefill_len = len(new_req.prompt_token_ids)
        else:
            prefill_len = 0
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
    resumed_req_ids = scheduler_output.scheduled_cached_reqs.resumed_req_ids
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
        state_store.update_cache_len(
            req_id,
            signal.estimated_cache_len,
            step=signal.step,
        )
        if signal.should_compress:
            state_store.mark_trigger(req_id, signal.reason, signal.step)
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
