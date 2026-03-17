"""Build effective-length/position override metadata for runtime input adapter."""

from __future__ import annotations

from typing import Any

import torch

from .debug_trace import trace_event
from .request_key_compat import get_scheduled_token_items
from .runner_struct_compat import resolve_req_id_to_index


def _has_active_compressed_requests(state_store: Any) -> bool | None:
    if state_store is None:
        return None
    checker = getattr(state_store, "has_active_compressed_requests", None)
    if callable(checker):
        try:
            return bool(checker())
        except Exception:
            return None
    return None


def _applied_req_ids_from_events(
    compression_events: list[dict[str, Any]] | None,
) -> set[Any]:
    out: set[Any] = set()
    if not isinstance(compression_events, list):
        return out
    for event in compression_events:
        if (
            isinstance(event, dict)
            and event.get("status") == "applied"
            and event.get("req_id") is not None
        ):
            out.add(event["req_id"])
    return out


def _scheduler_output_step(scheduler_output: Any) -> int | None:
    try:
        step = getattr(scheduler_output, "triattention_step", None)
    except Exception:
        return None
    return int(step) if isinstance(step, int) else None


def _state_marks_effective_pre_step_base(
    *,
    state: Any,
    scheduler_step: int | None,
) -> bool | None:
    """Return whether `current_cache_len` already equals the effective pre-step base.

    Returns:
      - True/False when request state exposes explicit semantics markers
      - None when markers are absent (caller may use legacy fallback heuristics)
    """
    if state is None:
        return None
    semantics = getattr(state, "current_cache_len_semantics", None)
    if not isinstance(semantics, str):
        return None
    if semantics != "effective_pre_step":
        return False
    state_step = getattr(state, "current_cache_len_step", None)
    if isinstance(scheduler_step, int) and isinstance(state_step, int):
        return state_step == scheduler_step
    return True


def _effective_base_before_step(
    *,
    state: Any,
    abs_progress: int,
    scheduled_tokens: int,
    scheduler_step: int | None,
) -> int:
    current_len = int(getattr(state, "current_cache_len", abs_progress)) if state is not None else abs_progress
    marked = _state_marks_effective_pre_step_base(
        state=state,
        scheduler_step=scheduler_step,
    )
    if marked is True:
        return current_len
    if marked is False:
        return max(0, current_len - max(0, int(scheduled_tokens)))
    if state is not None:
        compression_count = getattr(state, "compression_count", None)
        if isinstance(compression_count, int) and compression_count > 0:
            raise RuntimeError(
                "TRIATTN_EFFECTIVE_BASE_STATE_SEMANTICS_MISSING:"
                "compressed_request_state_missing_current_cache_len_semantics"
            )
    return max(0, current_len - max(0, int(scheduled_tokens)))


def _resolve_abs_progress_for_override(
    *,
    req_state: Any,
    state: Any,
    scheduled_tokens: int,
    scheduler_step: int | None,
) -> int:
    abs_progress = (
        int(getattr(req_state, "num_computed_tokens", 0))
        if req_state is not None
        else 0
    )
    if state is None:
        return abs_progress

    compression_count = getattr(state, "compression_count", None)
    prefill_len = getattr(state, "prefill_len", None)
    if not isinstance(compression_count, int) or compression_count <= 0:
        return abs_progress
    if not isinstance(prefill_len, int) or prefill_len <= 0:
        return abs_progress
    if abs_progress >= prefill_len:
        return abs_progress

    # In chunked-prefill serve mode, the first decode step after the final
    # prefill chunk can transiently expose a request progress counter that is
    # still one prefill chunk behind. Clamp that one-step lag to the full
    # prompt length so sparse position overrides stay aligned with the actual
    # cache contents after prefill compression.
    if int(scheduled_tokens) == 1 and _state_marks_effective_pre_step_base(
        state=state,
        scheduler_step=scheduler_step,
    ) is True:
        trace_event(
            "effective_override_prefill_progress_clamp",
            abs_progress=int(abs_progress),
            prefill_len=int(prefill_len),
            scheduled_tokens=int(scheduled_tokens),
            scheduler_step=int(scheduler_step) if isinstance(scheduler_step, int) else -1,
            compression_count=int(compression_count),
        )
        return int(prefill_len)
    return abs_progress


def _get_cached_sparse_overrides_result(
    *,
    scheduler_output: Any,
    base_runner: Any,
    state_store: Any,
) -> tuple[dict[int, int] | None, dict[int, int] | None, int | None, int] | None:
    if scheduler_output is None:
        return None
    try:
        cache = getattr(scheduler_output, "_triattention_cached_sparse_overrides", None)
    except Exception:
        cache = None
    if not isinstance(cache, dict):
        return None
    key = (id(base_runner), id(state_store))
    value = cache.get(key)
    if (
        isinstance(value, tuple)
        and len(value) == 4
    ):
        return value
    return None


def _set_cached_sparse_overrides_result(
    *,
    scheduler_output: Any,
    base_runner: Any,
    state_store: Any,
    value: tuple[dict[int, int] | None, dict[int, int] | None, int | None, int],
) -> None:
    if scheduler_output is None:
        return
    key = (id(base_runner), id(state_store))
    try:
        cache = getattr(scheduler_output, "_triattention_cached_sparse_overrides", None)
        if not isinstance(cache, dict):
            cache = {}
            setattr(scheduler_output, "_triattention_cached_sparse_overrides", cache)
        cache[key] = value
    except Exception:
        # Best-effort cache only.
        return

def build_effective_sparse_overrides(
    *,
    base_runner: Any,
    state_store: Any,
    scheduler_output: Any,
    compression_events: list[dict[str, Any]] | None = None,
) -> tuple[dict[int, int] | None, dict[int, int] | None, int | None, int]:
    """Build sparse per-request overrides for seq_lens and slot-mapping positions.

    Returns:
      - effective cache length before step by req_state index
      - (effective_base - absolute_base) delta by req_state index for slot mapping
      - single-request seq base fast-path scalar (optional)
      - single-request pos delta fast-path scalar (optional)
    """
    cached = _get_cached_sparse_overrides_result(
        scheduler_output=scheduler_output,
        base_runner=base_runner,
        state_store=state_store,
    )
    if cached is not None:
        return cached
    req_states = getattr(base_runner, "req_states", None)
    requests = getattr(base_runner, "requests", None)
    if req_states is None or not isinstance(requests, dict):
        if not isinstance(requests, dict):
            out = (None, None, None, 0)
            _set_cached_sparse_overrides_result(
                scheduler_output=scheduler_output,
                base_runner=base_runner,
                state_store=state_store,
                value=out,
            )
            return out
    req_id_to_index, _req_index_source = resolve_req_id_to_index(base_runner)
    if not isinstance(req_id_to_index, dict):
        out = (None, None, None, 0)
        _set_cached_sparse_overrides_result(
            scheduler_output=scheduler_output,
            base_runner=base_runner,
            state_store=state_store,
            value=out,
        )
        return out

    has_active_compressed = _has_active_compressed_requests(state_store)
    applied_event_reqs = _applied_req_ids_from_events(compression_events)
    if has_active_compressed is False and applied_event_reqs:
        raise RuntimeError(
            "TRIATTN_COMPRESSION_EVENT_STATE_MISMATCH:"
            "applied_compression_events_present_but_state_store_reports_no_compressed_requests"
        )
    if has_active_compressed is False:
        out = (None, None, None, 0)
        _set_cached_sparse_overrides_result(
            scheduler_output=scheduler_output,
            base_runner=base_runner,
            state_store=state_store,
            value=out,
        )
        return out

    scheduled_items = get_scheduled_token_items(scheduler_output)
    if not scheduled_items:
        out = (None, None, None, 0)
        _set_cached_sparse_overrides_result(
            scheduler_output=scheduler_output,
            base_runner=base_runner,
            state_store=state_store,
            value=out,
        )
        return out

    scheduler_step = _scheduler_output_step(scheduler_output)

    seq_bases: dict[int, int] = {}
    pos_deltas: dict[int, int] = {}
    requests_get = requests.get
    state_get = getattr(state_store, "get", None)
    if not callable(state_get):
        state_get = None
    for _raw_key, req_id, scheduled_tokens in scheduled_items:
        req_idx = req_id_to_index.get(req_id)
        if not isinstance(req_idx, int):
            continue
        state = state_get(req_id) if state_get is not None else None
        if state is not None:
            compression_count = getattr(state, "compression_count", None)
            if isinstance(compression_count, int) and compression_count <= 0:
                # Requests never compressed in their lifetime should continue to
                # follow native vLLM seq_len / slot-mapping semantics.
                continue
        req_state = requests_get(req_id)
        abs_progress = _resolve_abs_progress_for_override(
            req_state=req_state,
            state=state,
            scheduled_tokens=scheduled_tokens,
            scheduler_step=scheduler_step,
        )
        effective_before_step = _effective_base_before_step(
            state=state,
            abs_progress=abs_progress,
            scheduled_tokens=scheduled_tokens,
            scheduler_step=scheduler_step,
        )
        delta = int(effective_before_step - abs_progress)
        trace_event(
            "effective_override_observe",
            req_id=repr(req_id),
            req_idx=int(req_idx),
            scheduled_tokens=int(scheduled_tokens),
            abs_progress=int(abs_progress),
            current_cache_len=int(getattr(state, "current_cache_len", abs_progress)) if state is not None else int(abs_progress),
            current_cache_len_semantics=str(getattr(state, "current_cache_len_semantics", "none")) if state is not None else "none",
            current_cache_len_step=int(getattr(state, "current_cache_len_step", -1)) if state is not None else -1,
            effective_before_step=int(effective_before_step),
            pos_delta=int(delta),
            scheduler_step=int(scheduler_step) if isinstance(scheduler_step, int) else -1,
            compression_count=int(getattr(state, "compression_count", 0)) if state is not None else 0,
        )
        if delta == 0:
            continue
        # Sparse overrides only need rows whose effective base differs from the
        # original vLLM absolute progress. Zero-delta rows can safely fall back
        # to the unmodified base implementation outputs.
        seq_bases[req_idx] = effective_before_step
        pos_deltas[req_idx] = delta
        trace_event(
            "effective_override_row",
            req_id=repr(req_id),
            req_idx=int(req_idx),
            scheduled_tokens=int(scheduled_tokens),
            abs_progress=int(abs_progress),
            current_cache_len=int(getattr(state, "current_cache_len", abs_progress)) if state is not None else int(abs_progress),
            current_cache_len_semantics=str(getattr(state, "current_cache_len_semantics", "none")) if state is not None else "none",
            current_cache_len_step=int(getattr(state, "current_cache_len_step", -1)) if state is not None else -1,
            effective_before_step=int(effective_before_step),
            pos_delta=int(delta),
            scheduler_step=int(scheduler_step) if isinstance(scheduler_step, int) else -1,
            compression_count=int(getattr(state, "compression_count", 0)) if state is not None else 0,
        )

    single_seq_base: int | None = None
    single_pos_delta = 0
    if len(seq_bases) == 1:
        single_seq_base = next(iter(seq_bases.values()))
        if pos_deltas:
            single_pos_delta = int(next(iter(pos_deltas.values())))

    out = ((seq_bases or None), (pos_deltas or None), single_seq_base, single_pos_delta)
    _set_cached_sparse_overrides_result(
        scheduler_output=scheduler_output,
        base_runner=base_runner,
        state_store=state_store,
        value=out,
    )
    return out


def build_effective_seq_len_override_tensor(
    *,
    base_runner: Any,
    state_store: Any,
    scheduler_output: Any,
    compression_events: list[dict[str, Any]] | None = None,
) -> torch.Tensor | None:
    """Build per-req_state effective cache lengths for the current scheduled step."""
    req_states = getattr(base_runner, "req_states", None)
    requests = getattr(base_runner, "requests", None)
    if req_states is None or not isinstance(requests, dict):
        return None

    req_id_to_index = getattr(req_states, "req_id_to_index", None)
    num_computed_gpu = getattr(getattr(req_states, "num_computed_tokens", None), "gpu", None)
    if not isinstance(req_id_to_index, dict) or not isinstance(num_computed_gpu, torch.Tensor):
        return None

    has_active_compressed = _has_active_compressed_requests(state_store)
    if has_active_compressed is False:
        return None

    scheduled_items = get_scheduled_token_items(scheduler_output)
    if not scheduled_items:
        return None

    override = torch.empty_like(num_computed_gpu)
    override.copy_(num_computed_gpu)

    scheduler_step = _scheduler_output_step(scheduler_output)
    idxs: list[int] = []
    vals: list[int] = []
    requests_get = requests.get
    state_get = getattr(state_store, "get", None)
    if not callable(state_get):
        state_get = None
    for _raw_key, req_id, scheduled_tokens in scheduled_items:
        req_idx = req_id_to_index.get(req_id)
        if not isinstance(req_idx, int):
            continue
        state = state_get(req_id) if state_get is not None else None
        if state is not None:
            compression_count = getattr(state, "compression_count", None)
            if isinstance(compression_count, int) and compression_count <= 0:
                continue
        req_state = requests_get(req_id)
        abs_progress = (
            int(getattr(req_state, "num_computed_tokens", 0))
            if req_state is not None
            else 0
        )
        effective_before_step = _effective_base_before_step(
            state=state,
            abs_progress=abs_progress,
            scheduled_tokens=scheduled_tokens,
            scheduler_step=scheduler_step,
        )
        if effective_before_step == abs_progress:
            continue
        idxs.append(req_idx)
        vals.append(effective_before_step)

    if not idxs:
        return override
    idx_tensor = torch.as_tensor(idxs, device=override.device, dtype=torch.long)
    val_tensor = torch.as_tensor(vals, device=override.device, dtype=override.dtype)
    override.index_copy_(0, idx_tensor, val_tensor)
    return override


def build_effective_positions_override_tensor(
    *,
    base_runner: Any,
    state_store: Any,
    scheduler_output: Any,
    compression_events: list[dict[str, Any]] | None = None,
) -> torch.Tensor | None:
    """Build token-level positions for KV slot mappings using effective base len."""
    req_states = getattr(base_runner, "req_states", None)
    requests = getattr(base_runner, "requests", None)
    if req_states is None or not isinstance(requests, dict):
        return None
    num_computed_gpu = getattr(getattr(req_states, "num_computed_tokens", None), "gpu", None)
    if not isinstance(num_computed_gpu, torch.Tensor):
        return None

    has_active_compressed = _has_active_compressed_requests(state_store)
    if has_active_compressed is False:
        return None

    scheduled_items = get_scheduled_token_items(scheduler_output)
    if not scheduled_items:
        return None

    # Preserve scheduler mapping iteration order. Sorting by token count can diverge
    # from vLLM batch packing order and corrupt dense debug/compat overrides.
    ordered_pairs: list[tuple[str, int]] = [(req_id, qlen) for _raw, req_id, qlen in scheduled_items]
    total_tokens = int(sum(qlen for _req_id, qlen in ordered_pairs))
    if total_tokens <= 0:
        return None

    scheduler_step = _scheduler_output_step(scheduler_output)

    out = torch.empty(total_tokens, device=num_computed_gpu.device, dtype=num_computed_gpu.dtype)
    cursor = 0
    for req_id, qlen in ordered_pairs:
        if qlen <= 0:
            continue
        req_state = requests.get(req_id)
        abs_progress = (
            int(getattr(req_state, "num_computed_tokens", 0))
            if req_state is not None
            else 0
        )
        state = (
            state_store.get(req_id)
            if state_store is not None and hasattr(state_store, "get")
            else None
        )
        effective_base = _effective_base_before_step(
            state=state,
            abs_progress=abs_progress,
            scheduled_tokens=qlen,
            scheduler_step=scheduler_step,
        )
        out[cursor : cursor + qlen] = torch.arange(
            effective_base,
            effective_base + qlen,
            device=out.device,
            dtype=out.dtype,
        )
        cursor += qlen
    return out
