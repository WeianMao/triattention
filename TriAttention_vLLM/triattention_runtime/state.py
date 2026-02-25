"""Request-level runtime state for TriAttention v2."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable


@dataclass
class RequestCompressionState:
    req_id: str
    prefill_len: int
    protect_prefill: bool
    current_cache_len: int = 0
    last_absorbed_cache_len: int = 0
    recent_unabsorbed_tokens: int = 0
    compression_count: int = 0
    last_compression_step: int = -1
    pending_triggers: int = 0
    last_trigger_reason: str = "none"
    is_preempted: bool = False
    current_cache_len_semantics: str = "unknown"
    current_cache_len_step: int = -1

    @property
    def mode(self) -> str:
        return "protect_prefill" if self.protect_prefill else "trim_prefill"


class RequestStateStore:
    """Lifecycle-safe request state storage keyed by req_id."""

    def __init__(self) -> None:
        self._states: dict[str, RequestCompressionState] = {}
        self._compressed_req_ids: set[str] = set()

    def ensure(
        self,
        req_id: str,
        prefill_len: int,
        protect_prefill: bool,
    ) -> RequestCompressionState:
        state = self._states.get(req_id)
        if state is None:
            state = RequestCompressionState(
                req_id=req_id,
                prefill_len=prefill_len,
                protect_prefill=protect_prefill,
                last_absorbed_cache_len=max(0, prefill_len),
            )
            self._states[req_id] = state
            return state

        # Preserve most conservative prefill observation and latest mode.
        state.prefill_len = max(state.prefill_len, prefill_len)
        state.protect_prefill = protect_prefill
        if state.compression_count <= 0:
            # Before the first compression, prefill acts as the absorbed baseline.
            state.last_absorbed_cache_len = max(
                state.last_absorbed_cache_len,
                state.prefill_len,
            )
        return state

    def mark_preempted(self, req_id: str) -> None:
        state = self._states.get(req_id)
        if state is not None:
            state.is_preempted = True

    def mark_resumed(self, req_id: str) -> None:
        state = self._states.get(req_id)
        if state is not None:
            state.is_preempted = False

    def update_cache_len(self, req_id: str, cache_len: int, step: int | None = None) -> None:
        state = self._states.get(req_id)
        if state is not None:
            state.current_cache_len = max(0, int(cache_len))
            state.current_cache_len_semantics = "estimated_with_scheduled"
            if isinstance(step, int):
                state.current_cache_len_step = step
            baseline = max(0, int(state.last_absorbed_cache_len))
            state.recent_unabsorbed_tokens = max(0, state.current_cache_len - baseline)

    def mark_trigger(self, req_id: str, reason: str, step: int) -> None:
        state = self._states.get(req_id)
        if state is None:
            return
        state.pending_triggers += 1
        state.last_trigger_reason = reason
        state.last_compression_step = max(state.last_compression_step, step)

    def mark_compressed(self, req_id: str, step: int, cache_len: int) -> None:
        state = self._states.get(req_id)
        if state is None:
            return
        self._compressed_req_ids.add(req_id)
        state.compression_count += 1
        state.pending_triggers = max(state.pending_triggers - 1, 0)
        state.last_compression_step = step
        state.current_cache_len = cache_len
        state.current_cache_len_semantics = "effective_pre_step"
        state.current_cache_len_step = int(step)
        state.last_absorbed_cache_len = max(0, int(cache_len))
        state.recent_unabsorbed_tokens = 0
        state.last_trigger_reason = "applied"

    def mark_compression_skipped(self, req_id: str, reason: str, step: int) -> None:
        state = self._states.get(req_id)
        if state is None:
            return
        state.pending_triggers = max(state.pending_triggers - 1, 0)
        state.last_compression_step = max(state.last_compression_step, step)
        state.last_trigger_reason = f"skipped:{reason}"

    def remove(self, req_id: str) -> None:
        self._states.pop(req_id, None)
        self._compressed_req_ids.discard(req_id)

    def get(self, req_id: str) -> RequestCompressionState | None:
        return self._states.get(req_id)

    def snapshot(self) -> dict[str, RequestCompressionState]:
        return dict(self._states)

    def has_active_compressed_requests(self) -> bool:
        return bool(self._compressed_req_ids)

    def has_compressed_request_in(self, req_ids: Iterable[str]) -> bool:
        for req_id in req_ids:
            if req_id in self._compressed_req_ids:
                return True
        return False
