"""Effective vs logical length tracking for compressed requests.

Each active request has two lengths:

* **logical length** — the total number of tokens generated so far
  (what sglang's ``Req.fill_ids`` reflects).
* **effective length** — the number of KV entries actually present in
  the cache after compression.

This module maintains the mapping and enforces the core invariant:
``effective_len <= logical_len`` at all times.  It also provides:

* A regression guard that crashes on impossible ``effective > logical``
  states.
* Rollback handling when ``logical`` decreases (e.g. sglang's retract
  mechanism).
* A ``sync_kv_metadata`` helper that updates ``Req.kv_committed_len``
  and ``Req.kv_allocated_len`` after compression.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

logger = logging.getLogger(__name__)

# When true, an effective length exceeding logical length causes a hard
# crash instead of a warning + clamp.  Controlled via env var so that
# debugging runs can surface data-corruption bugs immediately.
_FAIL_ON_REGRESSION = os.environ.get(
    "TRIATTN_RUNTIME_FAIL_ON_EFFECTIVE_LEN_REGRESSION", "1"
).strip().lower() in {"1", "true", "yes", "on"}


class EffectiveLengthTracker:
    """Per-request effective-length bookkeeping.

    Tracks the divergence between logical length (monotonic token
    count maintained by sglang) and effective length (actual KV entries
    in the cache after compression).

    Lifecycle for a single request:
        1. ``register_request(req_id, initial_len)`` — called when the
           request first enters the running batch (after prefill).
        2. ``observe_logical_progress(req_id, logical_len)`` — called
           every schedule round to let the tracker infer how many new
           tokens were appended since the last observation.
        3. ``update_after_compression(req_id, new_effective_len)`` —
           called when compression shrinks the KV cache.
        4. ``remove_request(req_id)`` — called when the request
           finishes or is aborted.
    """

    def __init__(self) -> None:
        # Maps req_id -> current effective length.
        # Only present for requests that have been compressed at least
        # once (i.e. effective != logical).
        self._effective: Dict[str, int] = {}

        # Maps req_id -> last observed logical length.
        # Present for ALL tracked requests (compressed or not).
        self._last_logical: Dict[str, int] = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def register_request(self, req_id: str, initial_len: int) -> None:
        """Register a new request with its initial cache length.

        ``initial_len`` is typically the prefill length — the number of
        KV entries present when the request first enters decode phase.
        At this point effective == logical, so we only record the
        logical baseline; the effective override is created lazily on
        the first compression.
        """
        if initial_len < 0:
            raise ValueError(
                f"initial_len must be >= 0, got {initial_len} "
                f"for req {req_id}"
            )
        self._last_logical[req_id] = initial_len
        # Do NOT set _effective here — no compression has happened yet.

    def update_after_compression(
        self, req_id: str, new_effective_len: int
    ) -> None:
        """Record the effective length after a compression event.

        After compression the KV cache shrinks to ``new_effective_len``
        entries.  The logical length is unchanged (tokens were generated;
        they just no longer have KV backing).

        Invariant: ``new_effective_len <= last_logical``.
        """
        if new_effective_len < 0:
            raise ValueError(
                f"new_effective_len must be >= 0, got "
                f"{new_effective_len} for req {req_id}"
            )

        last_logical = self._last_logical.get(req_id)
        if last_logical is not None and new_effective_len > last_logical:
            msg = (
                f"Effective length regression: effective "
                f"({new_effective_len}) > logical ({last_logical}) "
                f"for req {req_id}.  This indicates a bug in the "
                f"compression pipeline."
            )
            if _FAIL_ON_REGRESSION:
                raise RuntimeError(msg)
            logger.error(msg + "  Clamping to logical.")
            new_effective_len = last_logical

        self._effective[req_id] = new_effective_len

    def get_effective_len(self, req_id: str) -> Optional[int]:
        """Return the current effective length, or ``None`` if the
        request has never been compressed (effective == logical).

        Callers that need a concrete number should fall back to the
        request's logical length when this returns ``None``.
        """
        return self._effective.get(req_id)

    def has_override(self, req_id: str) -> bool:
        """Whether the request has an effective-length override
        (i.e. has been compressed at least once)."""
        return req_id in self._effective

    def has_any_overrides(self) -> bool:
        """Whether *any* tracked request has an effective-length
        override.  Used as a fast-path check to skip input-patch
        logic when no requests are compressed."""
        return bool(self._effective)

    def observe_logical_progress(
        self, req_id: str, logical_len: int
    ) -> int:
        """Update the tracker with the latest logical length and
        return the current effective length.

        For uncompressed requests (no override), returns
        ``logical_len`` unchanged.

        For compressed requests, the effective length grows by the
        same delta as the logical length since the last observation::

            delta = logical_len - last_logical
            effective += delta

        This mirrors the fact that new tokens append to the KV cache
        after compression — each new token adds one KV entry.

        If ``logical_len < last_logical`` (rollback / retract), the
        effective length is clamped: ``min(effective, logical_len)``.
        """
        last = self._last_logical.get(req_id, logical_len)
        self._last_logical[req_id] = logical_len

        if req_id not in self._effective:
            # Never compressed — effective tracks logical identically.
            return logical_len

        effective = self._effective[req_id]

        if logical_len >= last:
            # Normal forward progress: effective grows by the same
            # amount as logical.
            # Long prefill progression.
            # For requests with prefill_len >> budget, the first
            # observation after registration may have delta=0 (same
            # step) or delta=1 (next decode step).  Large deltas
            # during prefill are safe because no _effective entry
            # exists yet (returns logical_len above).  After the
            # first compression, _effective is set to budget, and
            # subsequent deltas are small (1 per decode step).
            # The regression guard below catches any anomaly.
            effective += logical_len - last
        else:
            # Rollback (e.g. sglang retract): clamp effective to not
            # exceed the new logical length.
            effective = min(effective, logical_len)

        # Regression guard.
        if effective > logical_len:
            msg = (
                f"Effective length regression after progress sync: "
                f"effective ({effective}) > logical ({logical_len}) "
                f"for req {req_id}."
            )
            if _FAIL_ON_REGRESSION:
                raise RuntimeError(msg)
            logger.error(msg + "  Clamping to logical.")
            effective = logical_len

        effective = max(0, effective)
        self._effective[req_id] = effective
        return effective

    def remove_request(self, req_id: str) -> None:
        """Clean up all state for a completed or aborted request."""
        self._effective.pop(req_id, None)
        self._last_logical.pop(req_id, None)

    def get_offset(self, req_id: str) -> int:
        """Return ``logical - effective`` for a request.

        Returns 0 if the request has never been compressed or is
        not tracked.
        """
        effective = self._effective.get(req_id)
        if effective is None:
            return 0
        logical = self._last_logical.get(req_id)
        if logical is None:
            return 0
        return max(0, logical - effective)

    def snapshot(self) -> Dict[str, Tuple[int, Optional[int]]]:
        """Return a snapshot of all tracked state for debugging.

        Returns a dict mapping ``req_id`` to ``(last_logical,
        effective_or_None)``.
        """
        all_ids = set(self._last_logical) | set(self._effective)
        return {
            rid: (
                self._last_logical.get(rid, -1),
                self._effective.get(rid),
            )
            for rid in all_ids
        }

    # ------------------------------------------------------------------
    # KV metadata sync helper
    # ------------------------------------------------------------------

    @staticmethod
    def sync_kv_metadata(req: object, effective_len: int) -> None:
        """Synchronise ``req.kv_committed_len`` and
        ``req.kv_allocated_len`` to *effective_len*.

        Must be called from the scheduler side after compression events
        are processed (Phase E.3).

        These two fields are incremented by ``+1`` on every decode step
        inside ``prepare_for_decode``.  After compression the physical
        KV cache has shrunk, but these counters still reflect the
        pre-compression logical length.  If left unsynchronised:

        * ``new_tokens_required_next_decode`` page estimation based on
          ``kv_committed_len % page_size`` will be wrong.
        * Memory assertions (sglang schedule_batch.py ~line 902, 914)
          may fire.
        * Any code that indexes ``req_to_token`` via
          ``kv_allocated_len`` may read out-of-bounds.

        The fix is a simple pair of assignments that keep these
        counters aligned with the real KV occupancy.
        """
        # Type narrowing: we expect a sglang Req, but accept any object
        # to avoid importing sglang at module level.
        req.kv_committed_len = effective_len  # type: ignore[attr-defined]
        req.kv_allocated_len = effective_len  # type: ignore[attr-defined]
